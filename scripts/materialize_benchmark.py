#!/usr/bin/env python
"""
Standalone quality + timing comparison of instance-materialization backends.

This runs a trained NuGraph3 model forward **once per batch** to obtain the
object-condensation outputs (``ox``, ``beta``, foreground mask, truth), then
materializes those cached outputs with every requested backend and records, for
each backend and each event:

* clustering quality vs truth -- TWO adjusted Rand indices per event:
  ``ari_particle`` (predicted-foreground signal hits that have a truth particle;
  excludes the -1 no-instance population) and ``ari_inclusive`` (all
  semantically-labelled hits with -1 kept on both sides -- reproduces the
  original NuGraph validation mask). ``ari_inclusive`` is typically much higher
  because the large, correctly co-grouped -1 background is included. The two
  answer different questions; neither is "better". Plus homogeneity ("purity"),
  completeness, V-measure, #predicted/#true clusters and noise fraction on the
  particle population.
* timing -- BATCH-LEVEL: each backend's internal timers are measured once for the
  whole batch (``t_*_batch``, ``mat_batch_total_s``); the summary de-duplicates
  one timing record per batch (no per-event timing in this benchmark).

Because all backends see the *same* cached model outputs, the quality/timing
comparison is fair and no retraining is needed. Selected/high-occupancy/ντ event
subsets are chosen with ``--keys-file`` (one ``/dataset`` key per line) or, by
default, the first ``--limit-batches`` validation batches.

Outputs: ``<out>.csv`` (one row per event x backend) and ``<out>.summary.csv``
(per-backend aggregates: mean/median over all finite events and over ok-only,
plus n_all_noise/n_invalid/status counts and batch-timing stats). GPU results
require an EAF GPU; on CPU the timing still runs but is not representative.

Example (EAF) -- must match the real training architecture
(--event --semantic --filter --instance --particle-loss --featext --in-feats 11):
  python scripts/materialize_benchmark.py \
      --checkpoint best.ckpt --data-path $DATA --device 0 \
      --backends dbscan seed graph greedy --limit-batches 20 \
      --event --semantic --filter --instance --particle-loss --featext \
      --in-feats 11 --num-workers 0 --batch-size 8 --out mat_quality
"""
import argparse
import csv
import statistics
from itertools import islice

import torch

import nugraph as ng
from nugraph.models.nugraph3.materialize import (
    materialize_labels, labels_to_particle_graph, MaterializeTimer, BACKENDS,
    has_torch_cluster, preflight)

try:
    from sklearn.metrics import (adjusted_rand_score,
                                 homogeneity_completeness_v_measure)
except Exception:  # pragma: no cover
    adjusted_rand_score = None


def foreground_mask(h):
    """Replicate InstanceDecoder foreground selection (signal, non-diffuse)."""
    mask = torch.ones_like(h.of, dtype=torch.bool)
    if hasattr(h, "x_filter"):
        mask = mask & (h.x_filter > 0.5)
    if hasattr(h, "x_semantic"):
        mask = mask & (h.x_semantic.argmax(dim=1) != 6)
    return mask


NAN = float("nan")


def apply_legacy_featext_zero(data):
    """Zero the final four hit features (compatibility mode for legacy checkpoints).

    Reverts the corrected ``FeatureExtension`` write-back to the pre-fix (5438b1c)
    behaviour, where the 4 extended columns were always zero. Only the last four
    ``data["hit"].x`` columns are touched; the first seven (pos + raw hit/x) are
    untouched. Validates that the input has 11 features (featext enabled) and
    raises otherwise. Returns ``data`` (modified in place).
    """
    n = int(data["hit"].x.size(1))
    if n != 11:
        raise RuntimeError(
            "--legacy-zero-featext expects 11 hit input features (enable "
            f"--featext); got {n}. Refusing to zero features of an unexpected "
            "layout.")
    data["hit"].x[:, -4:] = 0.0
    return data


def event_quality(pred, true, y_semantic, keep_particle):
    """Two per-event ARIs (all tensors are event-local slices for one event).

    Args (all shape (n_hit_in_event,)):
        pred: predicted event-local cluster id per hit (-1 = background/noise/
            unmaterialized). This equals the training ``x_i`` up to a label
            permutation, so the adjusted Rand index is identical.
        true: truth instance id per hit (``y_i``; -1 = hit has no truth particle).
        y_semantic: truth semantic label per hit (-1 = unlabelled).
        keep_particle: foreground (predicted signal, non-diffuse) & y_semantic>=0.

    Returns a dict with:
        ari_particle  -- ARI over ``keep_particle & true>=0`` (real-particle
                         separation; excludes the -1 no-instance population);
        ari_inclusive -- ARI over ``y_semantic>=0`` INCLUDING -1 on both sides,
                         reproducing the original NuGraph validation mask;
        purity/completeness/v_measure/n_pred/n_true/noise_frac -- particle-scope
                         (computed on the ``ari_particle`` population).
    Any metric whose mask is empty (<2 hits) is returned as NaN so the event is
    still recorded and enters aggregates only where a finite value exists. The
    two ARIs answer different questions; neither is "better".
    """
    out = dict(ari_particle=NAN, ari_inclusive=NAN, purity=NAN,
               completeness=NAN, v_measure=NAN, n_pred=NAN, n_true=NAN,
               noise_frac=NAN)

    # particle-only metric: predicted foreground signal hits that have a truth
    # particle. Predicted noise (-1) among these is kept so over-noise is penalised.
    mp = keep_particle & (true >= 0)
    if int(mp.sum()) >= 2:
        p = pred[mp].cpu().numpy()
        t = true[mp].cpu().numpy()
        out["ari_particle"] = adjusted_rand_score(t, p)
        homo, comp, vm = homogeneity_completeness_v_measure(t, p)
        out["purity"], out["completeness"], out["v_measure"] = homo, comp, vm
        out["n_pred"] = int(len({int(x) for x in p if x >= 0}))
        out["n_true"] = int(len(set(int(x) for x in t)))
    if int(keep_particle.sum()):
        out["noise_frac"] = float((pred[keep_particle] < 0).float().mean())

    # NuGraph-compatible inclusive metric: all semantically-labelled hits, with
    # -1 kept as a label on both sides (large no-instance population included).
    mi = y_semantic >= 0
    if int(mi.sum()) >= 2:
        out["ari_inclusive"] = adjusted_rand_score(
            true[mi].cpu().numpy(), pred[mi].cpu().numpy())
    return out


# --------------------------------------------------------------------------- #
# Summary aggregation (pure-python; testable without torch/model)
# --------------------------------------------------------------------------- #
def _finite(vals):
    """Finite numeric values only (drops None and NaN)."""
    out = []
    for v in vals:
        if isinstance(v, (int, float)) and v == v:  # v==v drops NaN
            out.append(v)
    return out


def _mean_median_n(vals):
    f = _finite(vals)
    if not f:
        return float("nan"), float("nan"), 0
    return statistics.mean(f), statistics.median(f), len(f)


def _p95(vals):
    f = sorted(_finite(vals))
    if not f:
        return float("nan")
    import math
    k = 0.95 * (len(f) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    return f[lo] if lo == hi else f[lo] + (f[hi] - f[lo]) * (k - lo)


def summarize(rows, backends):
    """Aggregate per-event benchmark rows into one summary dict per backend.

    Quality is reported both over ALL finite events (valid outcomes incl.
    all_noise) and over finite status=="ok" events only, with explicit names so
    an ok-only value is never mistaken for a general backend value. Timing is
    aggregated from ONE record per batch (batch-level), never once per event row.
    """
    summ = []
    for backend in backends:
        br = [r for r in rows if r["backend"] == backend]
        if not br:
            continue
        ok = [r for r in br if r.get("status") == "ok"]
        status_counts = {}
        for r in br:
            status_counts[r.get("status", "?")] = \
                status_counts.get(r.get("status", "?"), 0) + 1
        # de-duplicate timing to one value per batch (rows of a batch share it)
        batch_time = {}
        for r in br:
            batch_time[r.get("batch")] = r.get("mat_batch_total_s")
        bt = list(batch_time.values())
        bt_mean, bt_med, _ = _mean_median_n(bt)

        s = {
            "backend": backend,
            "feature_mode": br[0].get("feature_mode", "populated"),
            "n_events": len(br),
            "n_ok": len(ok),
            "ok_fraction": (len(ok) / len(br)) if br else float("nan"),
            "n_all_noise": sum(1 for r in br if r.get("status") == "all_noise"),
            "n_invalid": sum(1 for r in br if not r.get("valid", True)),
            "n_safety_limited": sum(1 for r in br if r.get("safety_limit_reached")),
            "n_cc_not_converged": sum(1 for r in br
                                      if r.get("status") == "cc_not_converged"),
            "status_counts": status_counts,
            "max_temp_mb": max((r.get("largest_temp_mb", 0.0) for r in br),
                               default=0.0),
            "max_n_edges": max((r.get("n_edges", -1) for r in br), default=-1),
            # batch-level timing (one record per batch, de-duplicated)
            "n_batches": len(batch_time),
            "mean_mat_batch_total_s": bt_mean,
            "median_mat_batch_total_s": bt_med,
            "p95_mat_batch_total_s": _p95(bt),
            "max_mat_batch_total_s": (max(_finite(bt)) if _finite(bt)
                                      else float("nan")),
        }
        # both ARIs: all-event (finite, incl. all_noise) AND ok-only, named
        for metric in ("ari_particle", "ari_inclusive"):
            a_mean, a_med, a_n = _mean_median_n([r.get(metric) for r in br])
            o_mean, o_med, o_n = _mean_median_n([r.get(metric) for r in ok])
            s[f"mean_{metric}_all"] = a_mean
            s[f"median_{metric}_all"] = a_med
            s[f"n_{metric}_all"] = a_n
            s[f"mean_{metric}_ok"] = o_mean
            s[f"median_{metric}_ok"] = o_med
            s[f"n_{metric}_ok"] = o_n
        # secondary quality: explicit _all and _ok medians (never ambiguous)
        for metric in ("purity", "completeness", "n_pred", "n_true",
                       "noise_frac"):
            s[f"median_{metric}_all"] = _mean_median_n(
                [r.get(metric) for r in br])[1]
            s[f"median_{metric}_ok"] = _mean_median_n(
                [r.get(metric) for r in ok])[1]
        summ.append(s)
    return summ


def configure():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Trained checkpoint (recommended; else fresh weights)')
    p.add_argument('--preflight', action='store_true',
                   help='Print dependency/execution-path/limits report and exit')
    p.add_argument('--max-hits', type=int, default=44000,
                   help='Largest event size to check in --preflight')
    p.add_argument('--device', type=int, default=None)
    p.add_argument('--backends', nargs='+', default=list(BACKENDS),
                   choices=list(BACKENDS))
    p.add_argument('--limit-batches', type=int, default=20)
    p.add_argument('--keys-file', type=str, default=None,
                   help='Optional file of /dataset keys (one per line) to use '
                        'instead of the first N validation batches')
    p.add_argument('--split', type=str, default='validation',
                   choices=('train', 'validation', 'test'))
    p.add_argument('--out', type=str, default='mat_quality')
    p.add_argument('--legacy-zero-featext', action='store_true',
                   help='COMPATIBILITY: zero the final four hit features after '
                        'FeatureExtension, reproducing the pre-fix (5438b1c) '
                        'behaviour. Required to evaluate checkpoints trained '
                        'before that fix (their last 4 input columns were always '
                        'zero). Off by default (corrected populated features); '
                        'requires --featext and 11 input features.')
    # clustering parameters (--mat-t-beta/-t-d/-min-cluster-size, --dbscan-eps/
    # -min-samples, --mat-max-* etc.) are provided by add_model_args below, so
    # they are NOT re-declared here to avoid an argparse conflict.
    p = ng.data.H5DataModule.add_data_args(p)
    p = ng.models.NuGraph3.add_model_args(p)
    return p.parse_args()


def build_dataset(nudata, args):
    """Return an iterable of batches for the chosen subset."""
    if args.keys_file:
        from nugraph.data import NuGraphDataset
        from nugraph.util import FeatureExtension
        from torch_geometric.loader import DataLoader
        from torch_geometric.transforms import Compose
        with open(args.keys_file) as f:
            keys = [ln.strip() for ln in f if ln.strip()]
        # replicate the datamodule transform stack, incl. feature extension so
        # the architecture matches --featext --in-feats 11 exactly
        tfs = [ng.models.NuGraph3.transform(planes=nudata.planes)]
        if args.featext:
            tfs.append(FeatureExtension(planes=nudata.planes))
        ds = NuGraphDataset(nudata.filename, keys, Compose(tfs))
        return DataLoader(ds, batch_size=args.batch_size)
    loader = {"train": nudata.train_dataloader,
              "validation": nudata.val_dataloader,
              "test": nudata.test_dataloader}[args.split]()
    return loader


def main():
    args = configure()

    # dependency / execution-path / limits dry-run
    if args.preflight:
        import json
        rep = preflight(args.backends, max_hits=args.max_hits,
                        max_pairs=args.mat_max_pairs, max_seeds=args.mat_max_seeds,
                        max_edges=args.mat_max_edges,
                        cc_max_iters=args.mat_cc_max_iters,
                        max_clusters=args.mat_max_clusters)
        print(json.dumps(rep, indent=2))
        return

    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)

    nudata = ng.data.H5DataModule(
        args.data_path, batch_size=args.batch_size, model=ng.models.NuGraph3,
        shuffle='random', num_workers=args.num_workers, featext=args.featext)

    if args.checkpoint:
        model = ng.models.NuGraph3.load_from_checkpoint(args.checkpoint)
    else:
        print("WARNING: no --checkpoint; embeddings are untrained, quality "
              "numbers are meaningless (timing still valid).")
        model = ng.models.NuGraph3.from_args(args, nudata)

    accelerator, devices = ng.util.configure_device(args.device)
    device = torch.device("cuda" if (accelerator == "gpu") else "cpu")
    model = model.to(device).eval()
    cuda = device.type == "cuda"
    print(f"device={device}  torch_cluster={has_torch_cluster()}  "
          f"backends={args.backends}")

    timer = MaterializeTimer(enabled=True, cuda=cuda)
    rows = []
    # feature compatibility mode (explicit; default = corrected populated featext)
    feature_mode = "legacy_zero_featext" if args.legacy_zero_featext else "populated"
    if args.legacy_zero_featext and not args.featext:
        raise SystemExit("--legacy-zero-featext requires --featext (there must be "
                         "4 extended columns to zero).")
    print(f"feature_mode = {feature_mode}")

    loader = build_dataset(nudata, args)
    for bi, batch in enumerate(islice(loader, args.limit_batches)):
        data = batch.to(device)
        # zero the final four hit features BEFORE the forward, for legacy
        # checkpoints trained with the pre-fix (zero) FeatureExtension
        if args.legacy_zero_featext:
            apply_legacy_featext_zero(data)
        with torch.no_grad():
            model(data, stage="test")  # populates ox/of/x_filter/x_semantic
        h = data["hit"]
        ox, beta = h.ox.detach(), h.of.detach()
        mask = foreground_mask(h)
        true = data.y_i()
        ng_ = data.num_graphs
        ptr = h.ptr

        for backend in args.backends:
            timer.reset()
            labels, stats = materialize_labels(
                backend, ox, beta, mask, ptr, ng_, dbscan=dbscan,
                t_beta=args.mat_t_beta, t_d=args.mat_t_d,
                min_cluster_size=args.mat_min_cluster_size,
                max_pairs=args.mat_max_pairs, max_seeds=args.mat_max_seeds,
                max_edges=args.mat_max_edges, cc_max_iters=args.mat_cc_max_iters,
                max_clusters=args.mat_max_clusters, timer=timer)
            with timer.section("assemble"):
                labels_to_particle_graph(data, labels, timer=None)
            times = dict(timer.times)
            # NOTE: timing is measured ONCE for the whole backend call on this
            # batch, then copied into every event row -> it is BATCH-level, not
            # per-event. Field names carry the ``_batch`` suffix and the summary
            # de-duplicates one timing record per batch.
            mat_batch_total = sum(times.values())
            # per-event quality + status (one row per event x backend)
            for g in range(ng_):
                gm = h.batch == g
                ysem_g = h.y_semantic[gm]
                # foreground (predicted signal, non-diffuse) & semantically labelled
                keep_particle_g = mask[gm] & (ysem_g >= 0)
                # only score VALID clusterings -- invalid/safety-limited results
                # must never be reported as a real prediction
                q = (event_quality(labels[gm], true[gm], ysem_g, keep_particle_g)
                     if (adjusted_rand_score and stats[g].valid) else None)
                row = {"batch": bi, "event": g, "backend": backend,
                       "feature_mode": feature_mode,
                       "n_hits": int(gm.sum()),
                       "mat_batch_total_s": mat_batch_total}
                # structured status fields (counts, convergence, safety limits)
                row.update(stats[g].as_row())
                for k, v in times.items():
                    row[f"t_{k.replace('/', '_')}_batch"] = v
                if q:
                    row.update(q)
                rows.append(row)
        if bi % 5 == 0:
            print(f"  batch {bi}: {ng_} events")

    # write per-row CSV
    keys = sorted({k for r in rows for k in r})
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # per-backend summary (dual ARIs; all-event vs ok-only; batch-level timing)
    summ = summarize(rows, args.backends)
    with open(f"{args.out}.summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()) if summ else ["backend"])
        w.writeheader()
        w.writerows(summ)

    print(f"\nwrote {args.out}.csv and {args.out}.summary.csv")
    for s in summ:
        print("  ", s)


if __name__ == "__main__":
    main()
