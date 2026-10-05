"""NuGraph3 instance decoder"""
from typing import Any
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torchmetrics.functional.clustering import adjusted_rand_score
from torch_geometric.data import Batch
from ....util import ObjConLoss
from ..types import Data, N_IT, E_H_IT, N_IP
from ..materialize import (BACKENDS, MaterializeTimer, materialize_labels,
                           labels_to_particle_graph, has_torch_cluster)

class InstanceDecoder(nn.Module):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit, then (optionally) materialize discrete particle
    instances from that latent space.

    Materialization is factored into selectable backends (see
    ``models/nugraph3/materialize.py``):

    * ``"dbscan"`` -- sklearn CPU DBSCAN (baseline / fallback, default);
    * ``"seed"``   -- Candidate A, beta-seeded condensation-point assignment;
    * ``"graph"``  -- Candidate B, radius graph + connected components;
    * ``"greedy"`` -- Candidate C, sequential greedy condensation (oracle).

    By default materialization runs only during validation/test/inference (the
    original behaviour). Set ``materialize_train=True`` to also run it during
    training (every ``materialize_every_n`` steps). Materialization is detached,
    so enabling it never changes the training loss or gradients.

    Args:
        hit_features: Number of hit node features
        instance_features: Number of instance features
        particle_loss: Enable object-condensation particle loss term
        materializer: Backend name, one of ``materialize.BACKENDS``
        materialize_train: Run materialization during training as well
        materialize_every_n: When training, materialize every N steps
        t_beta: Seed/greedy condensation-score threshold (seed/greedy backends)
        t_d: Latent radius (all GPU backends; analogous to DBSCAN eps)
        min_cluster_size: Drop clusters smaller than this (1 = keep all)
        dbscan_eps: DBSCAN eps (dbscan backend)
        dbscan_min_samples: DBSCAN min_samples (dbscan backend)
        materialize_timing: Enable per-operation timing instrumentation
    """
    def __init__(self, hit_features: int, instance_features: int,
                 particle_loss: bool = False,
                 materializer: str = "dbscan",
                 materialize_train: bool = False,
                 materialize_every_n: int = 1,
                 t_beta: float = 0.1,
                 t_d: float = 0.3,
                 min_cluster_size: int = 1,
                 dbscan_eps: float = 0.3,
                 dbscan_min_samples: int = 15,
                 materialize_timing: bool = False,
                 mat_max_pairs: int = 4_000_000,
                 mat_max_seeds: int = 20_000,
                 mat_max_edges: int = 8_000_000,
                 mat_cc_max_iters: int = 200,
                 mat_max_clusters: int = 20_000,
                 mat_on_unsafe: str = "flag",
                 mat_fallback: str = "dbscan",
                 materialize_metric: bool = True):
        super().__init__()

        if materializer not in BACKENDS:
            raise ValueError(f"materializer '{materializer}' not in {BACKENDS}")

        # loss function
        self.loss = ObjConLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # beta MLP
        self.beta_net = nn.Sequential(
            nn.Linear(hit_features + 1, hit_features),
            nn.Mish(),
            nn.Linear(hit_features, 1),
            nn.Sigmoid(),
        )

        # coordinate MLP
        self.coord_net = nn.Sequential(
            nn.Linear(hit_features + instance_features, hit_features),
            nn.Mish(),
            nn.Linear(hit_features, instance_features),
        )

        # materialization configuration
        self.materializer = materializer
        self.materialize_train = materialize_train
        self.materialize_every_n = max(1, materialize_every_n)
        self.t_beta = t_beta
        self.t_d = t_d
        self.min_cluster_size = min_cluster_size
        self.mat_max_pairs = mat_max_pairs
        self.mat_max_seeds = mat_max_seeds
        self.mat_max_edges = mat_max_edges
        self.mat_cc_max_iters = mat_cc_max_iters
        self.mat_max_clusters = mat_max_clusters
        if mat_on_unsafe not in ("flag", "raise", "skip", "fallback"):
            raise ValueError(
                "mat_on_unsafe must be flag|raise|skip|fallback")
        self.mat_on_unsafe = mat_on_unsafe
        if mat_fallback not in ("dbscan", "seed", "graph", "greedy"):
            raise ValueError("mat_fallback must be a valid backend name")
        self.mat_fallback = mat_fallback
        self.materialize_metric = materialize_metric
        # DBSCAN kept as baseline/fallback backend
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.particle_loss = particle_loss
        self.timer = MaterializeTimer(enabled=materialize_timing)
        # step counter used to schedule train-time materialization
        self._train_fwd = 0

        if materializer == "graph" and not has_torch_cluster():
            # not fatal: graph backend has a size-guarded dense fallback
            print("[InstanceDecoder] torch_cluster not found; the 'graph' "
                  "backend will use a size-guarded dense fallback.")

    def _foreground_mask(self, h) -> torch.Tensor:
        """Foreground hit mask: keep signal, non-diffuse hits when available.

        Starts all-True and, when the filter/semantic heads have run, keeps only
        predicted-signal hits (``x_filter > 0.5``) that are not predicted diffuse
        (semantic class index 6 in the DUNE 7-class scheme). Returns (n_hit,).
        """
        mask = torch.ones_like(h.of, dtype=torch.bool)
        if hasattr(h, "x_filter"):
            mask = mask & (h.x_filter > 0.5)
        if hasattr(h, "x_semantic"):
            mask = mask & (h.x_semantic.argmax(dim=1) != 6)
        return mask

    def _rand_score(self, data: Data, valid=None) -> torch.Tensor:
        """Mean per-graph Adjusted Rand Score of predicted vs true instances.

        Uses the global per-hit predicted (``x_i``) and true (``y_i``) instance
        ids and slices per graph via the hit ``batch`` vector -- avoiding the
        expensive ``to_data_list()`` unbatch used previously. Graphs whose
        materialization was invalid (``valid[g] is False``) are excluded so a
        failed/degraded clustering is never scored as a real prediction.
        """
        h = data["hit"]
        x_i, y_i = data.x_i(), data.y_i()
        if isinstance(data, Batch):
            rand = []
            for g in range(data.num_graphs):
                if valid is not None and not valid[g]:
                    continue
                gm = (h.batch == g) & (h.y_semantic >= 0)
                if gm.any():
                    rand.append(adjusted_rand_score(x_i[gm], y_i[gm]))
            if not rand:
                return torch.tensor(0.0, device=h.of.device)
            return torch.stack(rand).mean()
        gm = h.y_semantic >= 0
        return adjusted_rand_score(x_i[gm], y_i[gm])

    # pylint: disable=arguments-differ
    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        h = data["hit"]
        device = h.x.device

        # run network and add output to graph object
        h.of = self.beta_net(torch.cat((h.x, h.of), dim=1)).squeeze(dim=-1)
        h.ox = self.coord_net(torch.cat((h.x, h.ox), dim=1))

        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["of"] = h.ptr
            data._slice_dict["hit"]["ox"] = h.ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # calculate semantic loss to input to object condensation particle loss
        loss_semantic = None
        if (self.particle_loss):
            loss_semantic = data.hit_loss()

        # calculate loss (differentiable; independent of materialization)
        loss = self.loss(h.ox, h.of, data.y_i(), h.y_semantic,
                         data[N_IT].num_nodes, data[E_H_IT].edge_index,
                         loss_semantic)
        loss *= (-1 * self.temp).exp()
        b, v, p = loss
        loss = loss.sum() + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"instance/loss-{stage}"] = loss
            metrics[f"instance/bkg-loss-{stage}"] = b
            metrics[f"instance/potential-loss-{stage}"] = v
            if self.particle_loss:
                metrics[f"instance/particle-loss-{stage}"] = p

        # decide whether to materialize discrete instances this step
        if self.training:
            do_mat = (self.materialize_train
                      and (self._train_fwd % self.materialize_every_n == 0))
            self._train_fwd += 1
        else:
            do_mat = True  # always materialize during val/test/inference

        if do_mat:
            self.timer.reset()
            with self.timer.section("mask"):
                mask = self._foreground_mask(h)

            # backend -> per-hit event-local labels (-1 = background/noise) plus
            # per-event MatStatus records (memory-bounded; see materialize.py)
            ptr = h.ptr if isinstance(data, Batch) else None
            num_graphs = data.num_graphs if isinstance(data, Batch) else 1
            with self.timer.section("cluster"):
                labels, stats = materialize_labels(
                    self.materializer, h.ox.detach(), h.of.detach(), mask,
                    ptr, num_graphs, dbscan=self.dbscan,
                    t_beta=self.t_beta, t_d=self.t_d,
                    min_cluster_size=self.min_cluster_size,
                    max_pairs=self.mat_max_pairs, max_seeds=self.mat_max_seeds,
                    max_edges=self.mat_max_edges,
                    cc_max_iters=self.mat_cc_max_iters,
                    max_clusters=self.mat_max_clusters, timer=self.timer)

            # surface safety/health aggregates so limited events are observable
            n_invalid = sum(1 for s in stats if not s.valid)
            n_cc_fail = sum(1 for s in stats if s.cc_converged is False)
            max_temp_mb = max((s.largest_temp_mb for s in stats), default=0.0)
            used_fallback = False

            # explicit configured behaviour when a result is invalid (a
            # safety-limited clustering is NOT a real prediction)
            if n_invalid:
                if self.mat_on_unsafe == "raise":
                    bad = next(s for s in stats if not s.valid)
                    raise RuntimeError(
                        f"Materialization invalid ({bad.status}) on event "
                        f"{bad.event}: {bad.note}")
                if self.mat_on_unsafe == "fallback":
                    with self.timer.section("cluster_fallback"):
                        labels, stats = materialize_labels(
                            self.mat_fallback, h.ox.detach(), h.of.detach(),
                            mask, ptr, num_graphs, dbscan=self.dbscan,
                            t_beta=self.t_beta, t_d=self.t_d,
                            min_cluster_size=self.min_cluster_size,
                            max_pairs=self.mat_max_pairs,
                            max_seeds=self.mat_max_seeds,
                            max_edges=self.mat_max_edges,
                            cc_max_iters=self.mat_cc_max_iters,
                            max_clusters=self.mat_max_clusters, timer=self.timer)
                        used_fallback = True
                        n_invalid = sum(1 for s in stats if not s.valid)

            valid_flags = [s.valid for s in stats]
            if stage:
                metrics[f"instance/mat-invalid-events-{stage}"] = torch.tensor(
                    float(n_invalid), device=device)
                metrics[f"instance/mat-cc-not-converged-{stage}"] = torch.tensor(
                    float(n_cc_fail), device=device)
                metrics[f"instance/mat-max-temp-mb-{stage}"] = torch.tensor(
                    float(max_temp_mb), device=device)
                metrics[f"instance/mat-fallback-used-{stage}"] = torch.tensor(
                    float(used_fallback), device=device)

            # "skip" mode: do not build particle objects or score when invalid,
            # just record the failure so it is never mistaken for a prediction
            if self.mat_on_unsafe == "skip" and n_invalid:
                if stage:
                    metrics[f"instance/mat-skipped-{stage}"] = torch.tensor(
                        1.0, device=device)
                if stage == "train":
                    metrics["temperature/instance"] = self.temp
                return loss, metrics

            # labels -> particle nodes + hit->particle edges (shared, batch-safe)
            labels_to_particle_graph(data, labels, timer=self.timer)

            # clustering-quality metric (uses truth); skipped in throughput mode
            # (materialize_metric=False) and always excludes invalid events
            want_metric = self.materialize_metric and (
                (not self.training) or self.materialize_train)
            if want_metric:
                with self.timer.section("metric"):
                    rand = self._rand_score(data, valid=valid_flags)
                if not -1. <= rand <= 1.:
                    raise RuntimeError(
                        f"Adjusted Rand Score metric value {rand} is outside "
                        "allowed range!")
                if stage:
                    metrics[f"instance/adjusted-rand-{stage}"] = rand
            if stage:
                metrics[f"instance/n-particles-{stage}"] = torch.tensor(
                    float(data[N_IP].x.size(0)), device=device)

            # surface timing (seconds) as metrics when instrumentation is on
            if self.timer.enabled and stage:
                total = 0.0
                for name, secs in self.timer.times.items():
                    metrics[f"instance/time-{name}-{stage}"] = torch.tensor(
                        secs, device=device)
                    total += secs
                metrics[f"instance/time-materialize_total-{stage}"] = torch.tensor(
                    total, device=device)

        if stage == "train":
            metrics["temperature/instance"] = self.temp

        return loss, metrics

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
