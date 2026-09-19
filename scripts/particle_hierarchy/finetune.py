#!/usr/bin/env python
"""finetune.py -- Stage 1b: end-to-end fine-tuning of NuGraph3 with the Stage-1a soft layer.

Pre-registered design: docs/particle_hierarchy_design.md §5.3 (workspace). Both arms start from
the same NuGraph3 checkpoint (ep29, #170 + LayerNorm) and a copy of the frozen Stage-1a head
(read only), use NuGraph's own losses unchanged plus lambda x the Stage-1a soft loss, the same
data order and the same optimiser. The only difference:

  --arm T   the soft loss's gradient flows into the trunk (h.x, ox, beta not detached)
  --arm C   the soft layer sees detached trunk outputs (trunk trained by NuGraph's losses only)

Soft-loss inputs are built per event by soft_particle.prepare() on detached copies, i.e. the
exact Stage-1a candidate and truth path; for arm T the foreground hits' h.x / ox / beta are then
swapped for the gradient-carrying tensors. Candidate selection is never differentiated.
Monitoring runs on dev-A events only (validation samples 0..1199 at batch size 8), checked
against the dev-A cache. Before training, one dev-A pass (epoch -1) must reproduce Stage 1a.

Outputs in --out: trunk_ep{N}.ckpt (loads with NuGraph3.load_from_checkpoint, so
`mat_bench cache` works unchanged), soft_ep{N}.pt (loads with soft_particle eval/diagnose),
history.csv, dev_events_ep{N}.csv, manifest.json, tb/ (TensorBoard).

  finetune.py --arm T --seed 1 --checkpoint EP29.ckpt --head HEAD.pt --data-path DATA.h5 \\
      --dev-cache CACHE_DEVA --out DIR --device 0
"""
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load("soft_particle", HERE / "soft_particle.py")
mb, s0 = sp.mb, sp.s0
NAN = float("nan")
SOFT_TARGETS = dict(target="own", exist_target="real", merge_scope="local")
DECODE = dict(sp.PRESETS["v2"], allow_null=False, merge_thr=0.5)   # frozen Stage-1a decode
DEV_BATCH = 8                                                     # dev-A cache batch size


# --------------------------------------------------------------------------- #
# data order: identical in both arms, whatever else consumes the global RNGs
# --------------------------------------------------------------------------- #
def seeded_balance_sampler(datasize, batch_size, balance_frac, seed):
    """NuGraph's BalanceSampler, with numpy's global RNG reseeded from (seed, epoch) around each
    epoch's draw and restored afterwards."""
    import numpy as np
    from nugraph.data.BalanceSampler import BalanceSampler

    class SeededBalanceSampler(BalanceSampler):
        def __init__(self):
            super().__init__(datasize, batch_size, balance_frac)
            self.seed, self.epoch = seed, 0

        def __iter__(self):
            state = np.random.get_state()
            np.random.seed((self.seed * 1_000_003 + self.epoch) % 2**32)
            try:
                order = list(super().__iter__())
            finally:
                np.random.set_state(state)
            self.epoch += 1
            return iter(order)

    return SeededBalanceSampler()


# --------------------------------------------------------------------------- #
# soft loss on a NuGraph batch
# --------------------------------------------------------------------------- #
def event_inputs(ox, beta, hx, fg, y_i, y_sem, s, g):
    """Detached CPU copy of one event in the cache's layout (what Stage 1a trained on)."""
    mask = fg[s].detach().cpu()
    return {"ox": ox[s].detach().float().cpu(), "beta": beta[s].detach().float().cpu(),
            "mask": mask, "y_i": y_i[s].detach().cpu(), "y_sem": y_sem[s].detach().cpu(),
            "hit_x_fg": hx[s].detach()[fg[s]].float().cpu(), "batch": -1, "event": g}


def soft_loss_from_hits(soft, ox, beta, hx, fg, y_i, y_sem, ptr, grad_to_trunk, rule_fn, mat,
                        M, t_d, t_assign, dev, checkpoint=True):
    """Mean over the batch's usable events of the Stage-1a (membership, existence, merge) losses.

    grad_to_trunk=False (arm C): the soft layer only sees detached copies, so the loss gives the
    trunk no gradient. True (arm T): foreground h.x / ox / beta are the trunk's own tensors.
    Returns (terms (3,), number of usable events).
    """
    import torch
    from torch.utils.checkpoint import checkpoint as ckpt
    terms = []
    bounds = ptr.tolist()
    for g in range(len(bounds) - 1):
        s = slice(bounds[g], bounds[g + 1])
        d = sp.prepare(event_inputs(ox, beta, hx, fg, y_i, y_sem, s, g), mat, rule_fn, M, t_d,
                       t_assign, dev)
        if d is None:
            continue
        if grad_to_trunk:
            fgi = d["fg_idx"].to(ox.device)
            d["hit_x"], d["ox"], d["beta"] = hx[s][fgi], ox[s][fgi], beta[s][fgi]

        def fn(hit_x, oxx, bb, d=d):
            dd = dict(d)
            dd["hit_x"], dd["ox"], dd["beta"] = hit_x, oxx, bb
            return torch.stack(sp.event_loss_terms(soft, dd, **SOFT_TARGETS))

        args = (d["hit_x"], d["ox"], d["beta"])
        terms.append(ckpt(fn, *args, use_reentrant=False) if checkpoint else fn(*args))
    if not terms:
        return torch.zeros(3, device=dev), 0
    return torch.stack(terms).mean(0), len(terms)


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_trunk(net, base, path, meta):
    """NuGraph3-loadable checkpoint: the base checkpoint's hyper-parameters and Lightning
    version, the fine-tuned weights, no optimiser state."""
    import torch
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "hyper_parameters": base["hyper_parameters"],
                "pytorch-lightning_version": base["pytorch-lightning_version"],
                "epoch": meta.get("epoch"), "global_step": meta.get("global_step"),
                "stage1b": meta}, path)


def save_soft(soft, cfg, path, meta):
    import torch
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in soft.state_dict().items()},
                "cfg": {**cfg, **meta}}, path)


# --------------------------------------------------------------------------- #
def build_module(a, net, soft, head_cfg, base_ckpt, mat, out):
    import numpy as np
    import torch
    import pytorch_lightning as pl

    rule_fn = s0.make_rule(head_cfg["rule"])
    M, t_d, t_assign = head_cfg["m"], head_cfg["t_d"], head_cfg["t_assign"]

    class SoftFinetune(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net, self.soft = net, soft
            self.phase = "baseline"
            self.hist = []
            self._reset_train()
            self._reset_val()

        # ---- accumulators ----
        def _reset_train(self):
            self.tr = {"n": 0, "loss": 0.0, "loss_ng": 0.0, "soft": 0.0, "mem": 0.0,
                       "exist": 0.0, "merge": 0.0, "events": 0, "soft_events": 0}

        def _reset_val(self):
            self.va_metrics, self.va_graphs, self.va_rows = {}, 0, []

        def configure_optimizers(self):
            return torch.optim.AdamW([{"params": self.net.parameters(), "lr": a.lr},
                                      {"params": self.soft.parameters(), "lr": a.head_lr}])

        def on_before_optimizer_step(self, optimizer):
            torch.nn.utils.clip_grad_norm_(self.soft.parameters(), 1.0)   # soft layer only

        def _soft(self, batch, grad_to_trunk):
            h = batch["hit"]
            fg = mb.foreground_mask(h)
            return soft_loss_from_hits(self.soft, h.ox, h.of, h.x, fg, batch.y_i(),
                                       h.y_semantic, h.ptr, grad_to_trunk, rule_fn, mat, M, t_d,
                                       t_assign, self.device, checkpoint=not a.no_soft_checkpoint)

        def training_step(self, batch, batch_idx):
            loss_ng, _ = self.net(batch, "train")
            terms, n_soft = self._soft(batch, a.arm == "T")
            soft_loss = terms.sum()
            loss = loss_ng + a.lam * soft_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {self.global_step}: "
                                   f"nugraph {float(loss_ng)}, soft {terms.tolist()}")
            t = self.tr
            t["n"] += 1
            t["events"] += batch.num_graphs
            t["soft_events"] += n_soft
            for k, v in (("loss", loss), ("loss_ng", loss_ng), ("soft", soft_loss),
                         ("mem", terms[0]), ("exist", terms[1]), ("merge", terms[2])):
                t[k] += float(v.detach())
            self.log("loss/train", loss.detach(), batch_size=batch.num_graphs, prog_bar=True)
            self.log("soft/train", soft_loss.detach(), batch_size=batch.num_graphs, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss_ng, metrics = self.net(batch, "val")
            ng = batch.num_graphs
            self.va_graphs += ng
            for k, v in {"loss/val": loss_ng, **metrics}.items():
                try:
                    self.va_metrics[k] = self.va_metrics.get(k, 0.0) + float(v) * ng
                except (TypeError, ValueError, RuntimeError):
                    pass
            h = batch["hit"]
            fg = mb.foreground_mask(h)
            y_i = batch.y_i()
            bounds = h.ptr.tolist()
            for g in range(len(bounds) - 1):
                s = slice(bounds[g], bounds[g + 1])
                ev = event_inputs(h.ox, h.of, h.x, fg, y_i, h.y_semantic, s, g)
                row = {"batch": a.dev_lo + batch_idx, "event": g, "n_hit": bounds[g + 1] - bounds[g]}
                d = sp.prepare(ev, mat, rule_fn, M, t_d, t_assign, self.device)
                if d is not None:
                    lab_fg, diag = sp.predict_labels(self.soft, d, **DECODE)
                    keep = d["fg"] & (d["y_sem"] >= 0)
                    lab = torch.full((d["n_hit"],), -1, dtype=torch.long)
                    lab[d["fg_idx"]] = lab_fg.cpu()
                    hard = torch.full((d["n_hit"],), -1, dtype=torch.long)
                    hard[d["fg_idx"]] = d["hard_labels"].cpu()
                    q, qh = (mb.event_quality(x, d["y_i"], d["y_sem"], keep) for x in (lab, hard))
                    row.update(ari_soft=q.get("ari_particle", NAN),
                               ari_hard=qh.get("ari_particle", NAN), n_pred=q.get("n_pred", NAN),
                               n_true=q.get("n_true", NAN), n_groups=diag["n_groups"], K=d["K"])
                self.va_rows.append(row)

        def on_train_epoch_start(self):
            self.phase = "train"
            self._reset_train()

        def on_validation_epoch_start(self):
            self._reset_val()

        def on_validation_epoch_end(self):
            epoch = -1 if self.phase == "baseline" else self.current_epoch
            rows = self.va_rows
            if epoch >= 0 and not self.trainer.sanity_checking and self.tr["n"] == 0:
                return                                    # not an end-of-epoch validation
            g = lambda c: np.array([float(r.get(c, NAN)) for r in rows], dtype=float)
            soft_a, hard_a = g("ari_soft"), g("ari_hard")
            ok = np.isfinite(soft_a) & np.isfinite(hard_a)
            dl = soft_a[ok] - hard_a[ok]
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = g("n_pred") / g("n_true")
            peak = (torch.cuda.max_memory_allocated(self.device) / 1e9
                    if self.device.type == "cuda" else NAN)
            row = {"epoch": epoch, "dev_events": len(rows), "gpu_peak_gb": peak,
                   "dev/ari_soft": float(np.nanmean(soft_a)),
                   "dev/ari_hard": float(np.nanmean(hard_a)),
                   "dev/soft_minus_hard": float(dl.mean()) if dl.size else NAN,
                   "dev/soft_minus_hard_se": (float(dl.std(ddof=1) / math.sqrt(dl.size))
                                              if dl.size > 1 else NAN),
                   "dev/pred_true_med": float(np.nanmedian(ratio)),
                   "dev/no_cluster": float(np.mean(g("n_groups") == 0)),
                   **{f"dev_ng/{k}": v / max(self.va_graphs, 1) for k, v in self.va_metrics.items()}}
            if epoch >= 0:
                t = self.tr
                row.update({f"train/{k}": t[k] / max(t["n"], 1)
                            for k in ("loss", "loss_ng", "soft", "mem", "exist", "merge")})
                row.update({"train/steps": t["n"], "train/events": t["events"],
                            "train/soft_events": t["soft_events"],
                            "global_step": int(self.global_step)})
            self.hist.append(row)
            mb.write_rows(out / "history.csv", self.hist)
            mb.write_rows(out / f"dev_events_ep{epoch}.csv", rows)
            for k in ("dev/ari_soft", "dev/ari_hard", "dev/soft_minus_hard", "dev/pred_true_med",
                      "dev/no_cluster"):
                self.log(k, row[k])
            print(f"[{a.arm} s{a.seed}] epoch {epoch}: " + "  ".join(
                f"{k} {v:.4f}" for k, v in row.items()
                if isinstance(v, float) and (k.startswith(("dev/", "train/"))
                                             or k in ("dev_ng/semantic/recall-val",
                                                      "dev_ng/event/recall-val",
                                                      "dev_ng/loss/val"))), flush=True)
            if epoch >= 0:
                meta = {"stage": "1b", "arm": a.arm, "seed": a.seed, "lam": a.lam, "epoch": epoch,
                        "global_step": int(self.global_step)}
                save_trunk(self.net, base_ckpt, out / f"trunk_ep{epoch}.ckpt", meta)
                save_soft(self.soft, head_cfg, out / f"soft_ep{epoch}.pt", meta)

    return SoftFinetune()


def dev_identity(val_dataset, lo, hi, cache):
    """Validation samples [lo*8, hi*8) must be the dev-A cache's events, in order."""
    idx = range(lo * DEV_BATCH, hi * DEV_BATCH)
    cache_n = [int(ev["mask"].numel()) for ev in mb.iter_cache(cache, lo, hi)]
    data_n = [int(val_dataset[i]["hit"].num_nodes) for i in idx]
    if cache_n != data_n:
        bad = next((i for i, (x, y) in enumerate(zip(cache_n, data_n)) if x != y), None)
        raise SystemExit(f"dev subset != dev cache: {len(data_n)} vs {len(cache_n)} events, "
                         f"first differing event {bad}")
    return list(idx)


def git_state(path):
    run = lambda *x: subprocess.run(["git", "-C", str(path), *x], capture_output=True,
                                    text=True).stdout.strip()
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--short"))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("T", "C"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", required=True, help="NuGraph3 start checkpoint (ep29)")
    p.add_argument("--checkpoint-sha256", default=None, help="refuse to run if it differs")
    p.add_argument("--head", required=True, help="frozen Stage-1a head (read only)")
    p.add_argument("--head-sha256", default=None, help="refuse to run if it differs")
    p.add_argument("--data-path", required=True)
    p.add_argument("--dev-cache", required=True, help="dev-A cache (identity check)")
    p.add_argument("--dev-batches", default="0:150")
    p.add_argument("--out", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4, help="trunk (as the ep29 run)")
    p.add_argument("--head-lr", type=float, default=1e-3, help="soft layer (as Stage 1a)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--balance-frac", type=float, default=0.1)
    p.add_argument("--limit-train-batches", type=int, default=None, help="smoke tests only")
    p.add_argument("--no-soft-checkpoint", action="store_true",
                   help="disable per-event activation checkpointing of the soft layer")
    p.add_argument("--no-baseline", action="store_true", help="skip the epoch -1 dev-A pass")
    a = p.parse_args()

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    import nugraph as ng

    out = Path(a.out)
    if (out / "history.csv").exists():
        raise SystemExit(f"{out} already has a history.csv; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    for path, want in ((a.checkpoint, a.checkpoint_sha256), (a.head, a.head_sha256)):
        if want and mb.sha256(path) != want:
            raise SystemExit(f"sha256 mismatch for {path}")
    torch.set_num_threads(4)                                # as tools/nugraph_train.py
    torch.manual_seed(a.seed)
    a.dev_lo, a.dev_hi = mb.parse_range(a.dev_batches)

    mat = sp._import_materialize()
    base = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    net = ng.models.NuGraph3.load_from_checkpoint(a.checkpoint, map_location="cpu")
    ck = torch.load(a.head, map_location="cpu", weights_only=False)      # read only
    head_cfg = dict(ck["cfg"])
    soft = sp.build_model(head_cfg["hit_dim"], head_cfg["lat_dim"], head_cfg["width"],
                          head_cfg["r"])
    soft.load_state_dict(ck["state_dict"])
    soft.overlap = "sparse"
    head_cfg.update(overlap="sparse", init_head=str(a.head), init_head_sha256=mb.sha256(a.head),
                    decode="v2", decode_kwargs=DECODE)

    nudata = ng.data.H5DataModule(a.data_path, batch_size=a.batch_size, model=ng.models.NuGraph3,
                                  shuffle="balance", balance_frac=a.balance_frac, num_workers=0,
                                  featext=False)
    dev_idx = dev_identity(nudata.val_dataset, a.dev_lo, a.dev_hi, a.dev_cache)
    train_loader = DataLoader(nudata.train_dataset, batch_size=a.batch_size, num_workers=0,
                              sampler=seeded_balance_sampler(nudata.train_datasize, a.batch_size,
                                                             a.balance_frac, a.seed),
                              drop_last=True, shuffle=False, pin_memory=True)
    dev_loader = DataLoader(Subset(nudata.val_dataset, dev_idx), batch_size=DEV_BATCH,
                            num_workers=0, shuffle=False, pin_memory=True)

    manifest = {"args": vars(a), "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint_sha256": mb.sha256(a.checkpoint), "head_sha256": mb.sha256(a.head),
                "materialize_sha256": mb.sha256(mat.__file__), "nugraph_pkg": ng.__file__,
                "layernorm_blocks": sum(hasattr(m, "norm") for m in net.core_net.children()),
                "soft_targets": SOFT_TARGETS, "decode": DECODE, "dev_events": len(dev_idx),
                "matmul_precision": {"baseline": "highest", "train_and_monitor": "high"},
                "torch": torch.__version__, "lightning": pl.__version__,
                **git_state(HERE)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    module = build_module(a, net, soft, head_cfg, base, mat, out)
    trainer = pl.Trainer(accelerator="gpu", devices=[a.device], max_epochs=a.epochs,
                         limit_train_batches=a.limit_train_batches, num_sanity_val_steps=0,
                         enable_checkpointing=False, precision="32",
                         logger=pl.loggers.TensorBoardLogger(save_dir=str(out), name="tb",
                                                             default_hp_metric=False))
    if not a.no_baseline:
        # full fp32, as the caches were made: the epoch -1 pass must reproduce Stage 1a
        torch.set_float32_matmul_precision("highest")
        trainer.validate(module, dataloaders=dev_loader, verbose=False)
    torch.set_float32_matmul_precision("high")              # as tools/nugraph_train.py
    module.phase = "train"
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
