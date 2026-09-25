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

Training loop: an explicit loop (zero_grad -> NuGraph loss + lambda soft loss -> backward ->
clip the soft layer -> AdamW step), i.e. what Lightning's automatic optimisation did, with
**exact resume**: resume.pt (atomic, every --resume-every steps and after every epoch) holds
the trunk, soft layer and optimiser state, epoch / batch-in-epoch / global step, the epoch's
running accumulators, the history, the torch (CPU + CUDA), numpy and python RNG states, and the
run configuration + code commit. The data order is a pure function of (seed, epoch), so a
resumed run replays the identical batch sequence from the first unfinished batch; steps.csv logs
a hash of every batch's sample list. `--resume` refuses any configuration or commit mismatch.
(GPU scatter kernels are not bitwise deterministic run to run; that is unchanged by resuming.)

Outputs in --out: trunk_ep{N}.ckpt (loads with NuGraph3.load_from_checkpoint, so
`mat_bench cache` works unchanged), soft_ep{N}.pt (loads with soft_particle eval/diagnose),
history.csv, steps.csv, dev_events_ep{N}.csv, manifest.json, resume.pt.

  finetune.py --arm T --seed 1 --checkpoint EP29.ckpt --head HEAD.pt --data-path DATA.h5 \\
      --dev-cache CACHE_DEVA --out DIR --device 0            # fresh
  finetune.py ...same arguments... --resume                  # continue an interrupted run
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
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
sc = _load("sem_consistency", HERE / "sem_consistency.py")
mb, s0 = sp.mb, sp.s0
NAN = float("nan")
SOFT_TARGETS = dict(target="own", exist_target="real", merge_scope="local")
DECODE = dict(sp.PRESETS["v2"], allow_null=False, merge_thr=0.5)   # frozen Stage-1a decode
DEV_BATCH = 8                                                     # dev-A cache batch size
RESUME_VERSION = 1


# --------------------------------------------------------------------------- #
# data order: a pure function of (seed, epoch), identical in both arms
# --------------------------------------------------------------------------- #
def seeded_balance_sampler(datasize, batch_size, balance_frac, seed):
    """NuGraph's BalanceSampler with numpy's global RNG reseeded from (seed, epoch) around each
    draw and restored afterwards. `epoch` and `skip` (samples already consumed in the epoch) are
    set explicitly by the training loop."""
    import numpy as np
    from nugraph.data.BalanceSampler import BalanceSampler

    class SeededBalanceSampler(BalanceSampler):
        def __init__(self):
            super().__init__(datasize, batch_size, balance_frac)
            self.seed, self.epoch, self.skip = seed, 0, 0

        def epoch_order(self, epoch):
            state = np.random.get_state()
            np.random.seed((self.seed * 1_000_003 + epoch) % 2**32)
            try:
                return [int(i) for i in BalanceSampler.__iter__(self)]
            finally:
                np.random.set_state(state)

        def __iter__(self):
            return iter(self.epoch_order(self.epoch)[self.skip:])

    return SeededBalanceSampler()


def ids_hash(ids):
    return hashlib.sha1(",".join(str(int(i)) for i in ids).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# RNG / atomic save
# --------------------------------------------------------------------------- #
def rng_state():
    import random
    import numpy as np
    import torch
    s = {"python": random.getstate(), "numpy": np.random.get_state(),
         "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    import random
    import numpy as np
    import torch
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class SimulatedCrash(RuntimeError):
    """Raised by the test hooks of ResumableLoop.run()."""


# --------------------------------------------------------------------------- #
# generic epoch/step driver with exact resume
# --------------------------------------------------------------------------- #
class ResumableLoop:
    """Subclasses provide: `optimizer`, modules() -> {name: nn.Module}, batches(epoch, skip) ->
    iterable of (sample_ids, batch) starting at batch `skip`, step(batch) -> {name: float} (does
    the optimiser step), end_epoch(epoch) (validation, history, checkpoints)."""

    def __init__(self, out, epochs, config, resume_every=100, limit_batches=None):
        self.out, self.epochs, self.config = Path(out), epochs, dict(config)
        self.resume_every, self.limit = resume_every, limit_batches
        self.st = {"epoch": 0, "batches_done": 0, "global_step": 0}
        self.tr, self.hist, self.extra = {}, [], {}

    # ---- to implement ----
    def modules(self):
        raise NotImplementedError

    def batches(self, epoch, skip):
        raise NotImplementedError

    def step(self, batch):
        raise NotImplementedError

    def end_epoch(self, epoch):
        pass

    # ---- state ----
    def state(self):
        return {"version": RESUME_VERSION, "config": self.config, "st": dict(self.st),
                "tr": dict(self.tr), "hist": list(self.hist), "extra": dict(self.extra),
                "modules": {k: {n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
                            for k, m in self.modules().items()},
                "optimizer": self.optimizer.state_dict(), "rng": rng_state(),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    @property
    def resume_path(self):
        return self.out / "resume.pt"

    def save_resume(self):
        atomic_save(self.state(), self.resume_path)

    def load_resume(self):
        import torch
        s = torch.load(self.resume_path, map_location="cpu", weights_only=False)
        if s.get("version") != RESUME_VERSION:
            raise SystemExit(f"resume.pt version {s.get('version')} != {RESUME_VERSION}")
        diff = {k: (s["config"].get(k), v) for k, v in self.config.items()
                if s["config"].get(k) != v}
        diff.update({k: (v, None) for k, v in s["config"].items() if k not in self.config})
        if diff:
            raise SystemExit("refusing to resume: configuration differs from the interrupted run "
                             f"(saved, now): {diff}")
        for k, m in self.modules().items():
            m.load_state_dict(s["modules"][k])
        self.optimizer.load_state_dict(s["optimizer"])
        self.st, self.tr, self.hist, self.extra = s["st"], s["tr"], s["hist"], s["extra"]
        set_rng_state(s["rng"])
        self._truncate_steps(self.st["global_step"])
        return s

    def _truncate_steps(self, global_step):
        path = self.out / "steps.csv"
        if not path.exists():
            return
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["global_step"]) <= global_step]
        keys = list(rows[0].keys()) if rows else None
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", newline="") as f:
            if keys:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        os.replace(tmp, path)

    # ---- driver ----
    def run(self, crash_at_step=None, crash_after_epoch=None):
        steps_path = self.out / "steps.csv"
        fields = None
        if steps_path.exists() and steps_path.stat().st_size:
            with open(steps_path) as f:
                fields = next(csv.reader(f), None)
        fh = open(steps_path, "a", newline="")
        writer = None
        try:
            while self.st["epoch"] < self.epochs:
                e, k0 = self.st["epoch"], self.st["batches_done"]
                if k0 == 0:
                    self.tr = {}
                for k, (ids, batch) in enumerate(self.batches(e, k0), start=k0):
                    if self.limit is not None and k >= self.limit:
                        break
                    t0 = time.perf_counter()
                    out = self.step(batch)
                    dt = time.perf_counter() - t0
                    self.tr["n"] = self.tr.get("n", 0) + 1
                    for key, v in out.items():
                        self.tr[key] = self.tr.get(key, 0.0) + v
                    self.st["global_step"] += 1
                    self.st["batches_done"] = k + 1
                    row = {"global_step": self.st["global_step"], "epoch": e, "batch": k,
                           "ids_hash": ids_hash(ids), "n_samples": len(ids), "step_s": dt, **out}
                    if writer is None:
                        fields = fields or list(row.keys())
                        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                        if not steps_path.stat().st_size:
                            writer.writeheader()
                    writer.writerow(row)
                    fh.flush()
                    if self.resume_every and self.st["global_step"] % self.resume_every == 0:
                        self.save_resume()
                    if crash_at_step is not None and self.st["global_step"] == crash_at_step:
                        raise SimulatedCrash(f"crash after step {crash_at_step}")
                self.end_epoch(e)
                if crash_after_epoch is not None and e == crash_after_epoch:
                    raise SimulatedCrash(f"crash after end of epoch {e}, before its resume save")
                self.st.update(epoch=e + 1, batches_done=0)
                self.save_resume()
        finally:
            fh.close()


# --------------------------------------------------------------------------- #
# soft loss on a NuGraph batch
# --------------------------------------------------------------------------- #
def tick(timers, sync, key, t0):
    """Accumulate elapsed time into timers[key]; no-op when timers is None (production path).

    Diagnostic only: when profiling, `sync` forces the device to finish so a stage's time is its own.
    The syncs make a profiled step slightly SLOWER than a normal one, so profiled steps must not be
    used for throughput numbers -- only for the breakdown.
    """
    if timers is None:
        return t0
    if sync:
        sync()
    t1 = time.perf_counter()
    timers[key] = timers.get(key, 0.0) + (t1 - t0)
    return t1


PROFILE_STAGES = ("to_device", "nugraph_fwd", "ev_inputs", "prepare", "swap_grad_tensors",
                  "soft_fwd", "soft_other", "backward", "backward_ng", "backward_soft", "optimizer")


def validate_profile_args(a):
    """--profile-split-backward is only sound in arm C, where the two graphs are disjoint.

    In arm C the soft layer sees detached copies, so the NuGraph loss reaches only the trunk and the
    soft loss only the soft layer: backward on each separately accumulates exactly the same gradients
    as one backward on the sum. In arm T the soft loss reaches the trunk too, so a split would walk
    the trunk twice -- slower AND wrongly attributed.
    """
    if a.profile_split_backward:
        if a.arm != "C":
            raise SystemExit("--profile-split-backward requires --arm C (in arm T the soft loss also "
                             "flows into the trunk, so the split double-counts the trunk backward)")
        if not a.profile_steps:
            raise SystemExit("--profile-split-backward needs --profile-steps N")
        if getattr(a, "sem_cons", 0.0):
            raise SystemExit("--profile-split-backward does not backward the --sem-cons term; not allowed together")


def profile_row(global_step, epoch, timers, step_total, extras):
    """One diagnostic row: every stage, the unattributed remainder, and the step total."""
    row = {"global_step": global_step, "epoch": epoch, "step_total_s": step_total,
           **{f"t_{k}_s": timers.get(k, 0.0) for k in PROFILE_STAGES}, **extras}
    row["t_unattributed_s"] = step_total - sum(timers.get(k, 0.0) for k in PROFILE_STAGES)
    return row


def event_inputs(ox, beta, hx, fg, y_i, y_sem, s, g):
    """Detached CPU copy of one event in the cache's layout (what Stage 1a trained on)."""
    mask = fg[s].detach().cpu()
    return {"ox": ox[s].detach().float().cpu(), "beta": beta[s].detach().float().cpu(),
            "mask": mask, "y_i": y_i[s].detach().cpu(), "y_sem": y_sem[s].detach().cpu(),
            "hit_x_fg": hx[s].detach()[fg[s]].float().cpu(), "batch": -1, "event": g}


def soft_loss_from_hits(soft, ox, beta, hx, fg, y_i, y_sem, ptr, grad_to_trunk, rule_fn, mat,
                        M, t_d, t_assign, dev, checkpoint=True, timers=None, sync=None):
    """Mean over the batch's usable events of the Stage-1a (membership, existence, merge) losses.

    grad_to_trunk=False (arm C): the soft layer only sees detached copies, so the loss gives the
    trunk no gradient. True (arm T): foreground h.x / ox / beta are the trunk's own tensors.
    Returns (terms (3,), number of usable events).
    """
    import torch
    from torch.utils.checkpoint import checkpoint as ckpt
    terms = []
    bounds = ptr.tolist()
    t0 = time.perf_counter()
    for g in range(len(bounds) - 1):
        s = slice(bounds[g], bounds[g + 1])
        ev = event_inputs(ox, beta, hx, fg, y_i, y_sem, s, g)
        t0 = tick(timers, sync, "ev_inputs", t0)
        d = sp.prepare(ev, mat, rule_fn, M, t_d, t_assign, dev)
        t0 = tick(timers, sync, "prepare", t0)
        if d is None:
            continue
        if grad_to_trunk:
            fgi = d["fg_idx"].to(ox.device)
            d["hit_x"], d["ox"], d["beta"] = hx[s][fgi], ox[s][fgi], beta[s][fgi]
        t0 = tick(timers, sync, "swap_grad_tensors", t0)

        def fn(hit_x, oxx, bb, d=d):
            dd = dict(d)
            dd["hit_x"], dd["ox"], dd["beta"] = hit_x, oxx, bb
            return torch.stack(sp.event_loss_terms(soft, dd, **SOFT_TARGETS))

        args = (d["hit_x"], d["ox"], d["beta"])
        terms.append(ckpt(fn, *args, use_reentrant=False) if checkpoint else fn(*args))
        t0 = tick(timers, sync, "soft_fwd", t0)
    if not terms:
        return torch.zeros(3, device=dev), 0
    out = torch.stack(terms).mean(0), len(terms)
    tick(timers, sync, "soft_other", t0)
    return out


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_trunk(net, base, path, meta):
    """NuGraph3-loadable checkpoint: the base checkpoint's hyper-parameters and Lightning
    version, the fine-tuned weights, no optimiser state."""
    atomic_save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                 "hyper_parameters": base["hyper_parameters"],
                 "pytorch-lightning_version": base["pytorch-lightning_version"],
                 "epoch": meta.get("epoch"), "global_step": meta.get("global_step"),
                 "stage1b": meta}, path)


def save_soft(soft, cfg, path, meta):
    atomic_save({"state_dict": {k: v.detach().cpu() for k, v in soft.state_dict().items()},
                 "cfg": {**cfg, **meta}}, path)


# --------------------------------------------------------------------------- #
# the Stage-1b run
# --------------------------------------------------------------------------- #
class Stage1bLoop(ResumableLoop):
    def __init__(self, a, net, soft, head_cfg, base_ckpt, mat, train_ds, datasize, dev_loader,
                 dev, config):
        import torch
        super().__init__(a.out, a.epochs, config, a.resume_every, a.limit_train_batches)
        self.a, self.net, self.soft, self.head_cfg, self.base = a, net, soft, head_cfg, base_ckpt
        self.mat, self.train_ds, self.dev_loader, self.dev = mat, train_ds, dev_loader, dev
        self.rule_fn = s0.make_rule(head_cfg["rule"])
        self.M, self.t_d, self.t_assign = head_cfg["m"], head_cfg["t_d"], head_cfg["t_assign"]
        self.sampler = seeded_balance_sampler(datasize, a.batch_size, a.balance_frac, a.seed)
        self.prof_rows = []
        self.optimizer = torch.optim.AdamW([{"params": net.parameters(), "lr": a.lr},
                                            {"params": soft.parameters(), "lr": a.head_lr}])

    def modules(self):
        return {"net": self.net, "soft": self.soft}

    def batches(self, epoch, skip):
        from torch_geometric.loader import DataLoader
        bs = self.a.batch_size
        order = self.sampler.epoch_order(epoch)
        self.sampler.epoch, self.sampler.skip = epoch, skip * bs
        nw = self.a.num_workers
        extra = dict(persistent_workers=True, prefetch_factor=self.a.prefetch_factor) if nw else {}
        loader = DataLoader(self.train_ds, batch_size=bs, num_workers=nw, sampler=self.sampler,
                            drop_last=True, shuffle=False, pin_memory=True, **extra)
        for j, batch in enumerate(loader):
            k = skip + j
            yield order[k * bs:(k + 1) * bs], batch

    def _soft(self, batch, grad_to_trunk, timers=None, sync=None):
        h = batch["hit"]
        return soft_loss_from_hits(self.soft, h.ox, h.of, h.x, mb.foreground_mask(h), batch.y_i(),
                                   h.y_semantic, h.ptr, grad_to_trunk, self.rule_fn, self.mat,
                                   self.M, self.t_d, self.t_assign, self.dev,
                                   checkpoint=not self.a.no_soft_checkpoint,
                                   timers=timers, sync=sync)

    def step(self, batch):
        import torch
        prof = self.st["global_step"] < (self.a.profile_steps or 0)
        timers = {} if prof else None
        sync = ((lambda: torch.cuda.synchronize(self.dev))
                if prof and self.dev.type == "cuda" else None)
        t_step = time.perf_counter()
        t0 = t_step
        self.net.train()
        self.soft.train()
        batch = batch.to(self.dev)
        t0 = tick(timers, sync, "to_device", t0)
        self.optimizer.zero_grad(set_to_none=True)
        loss_ng, _ = self.net(batch, "train")
        t0 = tick(timers, sync, "nugraph_fwd", t0)
        terms, n_soft = self._soft(batch, self.a.arm == "T", timers=timers, sync=sync)
        t0 = time.perf_counter() if prof else t0          # soft stages timed inside
        soft_loss = terms.sum()
        loss = loss_ng + self.a.lam * soft_loss
        cons = None
        if self.a.sem_cons:                      # Stage 1a-R7: within-particle semantic consistency
            h = batch["hit"]
            cons = sc.sem_consistency_loss(h.x_semantic, batch.y_i(), h.y_semantic, h.batch)
            loss = loss + self.a.sem_cons * cons
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {self.st['global_step']}: "
                               f"nugraph {float(loss_ng)}, soft {terms.tolist()}")
        if prof and self.a.profile_split_backward:              # arm C only; see validate_profile_args
            loss_ng.backward()
            t0 = tick(timers, sync, "backward_ng", t0)
            (self.a.lam * soft_loss).backward()
            t0 = tick(timers, sync, "backward_soft", t0)
        else:
            loss.backward()
            t0 = tick(timers, sync, "backward", t0)
        torch.nn.utils.clip_grad_norm_(self.soft.parameters(), 1.0)       # soft layer only
        self.optimizer.step()
        t0 = tick(timers, sync, "optimizer", t0)
        if prof:
            self.prof_rows.append(profile_row(
                self.st["global_step"], self.st["epoch"], timers,
                time.perf_counter() - t_step,
                {"n_events": int(batch.num_graphs), "n_soft_events": n_soft,
                 "n_hits": int(batch["hit"].num_nodes),
                 "soft_checkpoint": not self.a.no_soft_checkpoint}))
            mb.write_rows(self.out / "profile.csv", self.prof_rows)
            if len(self.prof_rows) == self.a.profile_steps:
                self.print_profile()
        return {"loss": float(loss.detach()), "loss_ng": float(loss_ng.detach()),
                "soft": float(soft_loss.detach()), "mem": float(terms[0].detach()),
                "exist": float(terms[1].detach()), "merge": float(terms[2].detach()),
                "events": float(batch.num_graphs), "soft_events": float(n_soft),
                **({"cons": float(cons.detach())} if cons is not None else {})}

    def print_profile(self):
        """Mean seconds per stage over the profiled steps (diagnostic; syncs included)."""
        import numpy as np
        rows = self.prof_rows
        tot = float(np.mean([r["step_total_s"] for r in rows]))
        print(f"\n[profile] {len(rows)} steps, mean step {tot:.3f} s (with per-stage syncs; a normal "
              f"step is slightly faster). Mean seconds per step:", flush=True)
        for k in PROFILE_STAGES + ("unattributed",):
            key = f"t_{k}_s"
            v = float(np.mean([r.get(key, 0.0) for r in rows]))
            print(f"[profile]   {k:20s} {v:7.3f} s  ({100 * v / tot:5.1f} %)", flush=True)
        ev = float(np.mean([r["n_soft_events"] for r in rows]))
        soft = sum(float(np.mean([r[f"t_{k}_s"] for r in rows]))
                   for k in ("ev_inputs", "prepare", "swap_grad_tensors", "soft_fwd", "soft_other"))
        print(f"[profile]   soft-layer forward total {soft:.3f} s over {ev:.1f} events "
              f"= {1e3 * soft / max(ev, 1):.1f} ms/event; wrote {self.out}/profile.csv\n", flush=True)

    # ---- dev-A monitoring ----
    def validate(self, epoch):
        import numpy as np
        import torch
        self.net.eval()
        self.soft.eval()
        metrics, graphs, rows = {}, 0, []
        sem = [0.0, 0.0, 0, 0]                   # disagreeing pairs, pairs, labelled hits, correct
        with torch.no_grad():
            for bi, batch in enumerate(self.dev_loader):
                batch = batch.to(self.dev)
                loss_ng, m = self.net(batch, "val")
                hs = batch["hit"]
                for i, v in enumerate(sc.pair_counts_torch(hs.x_semantic.argmax(dim=1), batch.y_i(),
                                                           hs.y_semantic, hs.batch)):
                    sem[i] += v
                ng = batch.num_graphs
                graphs += ng
                for k, v in {"loss/val": loss_ng, **m}.items():
                    try:
                        metrics[k] = metrics.get(k, 0.0) + float(v) * ng
                    except (TypeError, ValueError, RuntimeError):
                        pass
                h = batch["hit"]
                fg, y_i = mb.foreground_mask(h), batch.y_i()
                bounds = h.ptr.tolist()
                for g in range(len(bounds) - 1):
                    s = slice(bounds[g], bounds[g + 1])
                    row = {"batch": self.a.dev_lo + bi, "event": g,
                           "n_hit": bounds[g + 1] - bounds[g]}
                    d = sp.prepare(event_inputs(h.ox, h.of, h.x, fg, y_i, h.y_semantic, s, g),
                                   self.mat, self.rule_fn, self.M, self.t_d, self.t_assign,
                                   self.dev)
                    if d is not None:
                        lab_fg, diag = sp.predict_labels(self.soft, d, **DECODE)
                        keep = d["fg"] & (d["y_sem"] >= 0)
                        lab = torch.full((d["n_hit"],), -1, dtype=torch.long)
                        lab[d["fg_idx"]] = lab_fg.cpu()
                        hard = torch.full((d["n_hit"],), -1, dtype=torch.long)
                        hard[d["fg_idx"]] = d["hard_labels"].cpu()
                        q, qh = (mb.event_quality(x, d["y_i"], d["y_sem"], keep)
                                 for x in (lab, hard))
                        row.update(ari_soft=q.get("ari_particle", NAN),
                                   ari_hard=qh.get("ari_particle", NAN),
                                   n_pred=q.get("n_pred", NAN), n_true=q.get("n_true", NAN),
                                   n_groups=diag["n_groups"], K=d["K"])
                    rows.append(row)
        g_ = lambda c: np.array([float(r.get(c, NAN)) for r in rows], dtype=float)
        soft_a, hard_a = g_("ari_soft"), g_("ari_hard")
        ok = np.isfinite(soft_a) & np.isfinite(hard_a)
        dl = soft_a[ok] - hard_a[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = g_("n_pred") / g_("n_true")
        peak = (torch.cuda.max_memory_allocated(self.dev) / 1e9
                if self.dev.type == "cuda" else NAN)
        row = {"epoch": epoch, "dev_events": len(rows), "gpu_peak_gb": peak,
               "dev/ari_soft": float(np.nanmean(soft_a)), "dev/ari_hard": float(np.nanmean(hard_a)),
               "dev/soft_minus_hard": float(dl.mean()) if dl.size else NAN,
               "dev/soft_minus_hard_se": (float(dl.std(ddof=1) / math.sqrt(dl.size))
                                          if dl.size > 1 else NAN),
               "dev/pred_true_med": float(np.nanmedian(ratio)),
               "dev/no_cluster": float(np.mean(g_("n_groups") == 0)),
               "dev_sem/pair_disagree": sem[0] / sem[1] if sem[1] else NAN,
               "dev_sem/acc": sem[3] / sem[2] if sem[2] else NAN,
               **{f"dev_ng/{k}": v / max(graphs, 1) for k, v in metrics.items()}}
        return row, rows

    def record(self, epoch, row, rows):
        self.hist.append(row)
        mb.write_rows(self.out / "history.csv", self.hist)
        mb.write_rows(self.out / f"dev_events_ep{epoch}.csv", rows)
        print(f"[{self.a.arm} s{self.a.seed}] epoch {epoch}: " + "  ".join(
            f"{k} {v:.4f}" for k, v in row.items()
            if isinstance(v, float) and (k.startswith(("dev/", "dev_sem/", "train/", "gpu_"))
                                         or k in ("dev_ng/semantic/recall-val",
                                                  "dev_ng/event/recall-val", "dev_ng/loss/val"))),
            flush=True)

    def baseline(self):
        import torch
        torch.set_float32_matmul_precision("highest")    # as the caches were made
        row, rows = self.validate(-1)
        self.record(-1, row, rows)
        self.extra["baseline_done"] = True

    def end_epoch(self, epoch):
        row, rows = self.validate(epoch)
        n = max(self.tr.get("n", 0), 1)
        row.update({f"train/{k}": self.tr.get(k, 0.0) / n
                    for k in ("loss", "loss_ng", "soft", "mem", "exist", "merge", "cons") if k in self.tr})
        row.update({"train/steps": self.tr.get("n", 0), "train/events": self.tr.get("events", 0),
                    "train/soft_events": self.tr.get("soft_events", 0),
                    "global_step": self.st["global_step"]})
        self.record(epoch, row, rows)
        meta = {"stage": "1b", "arm": self.a.arm, "seed": self.a.seed, "lam": self.a.lam,
                "epoch": epoch, "global_step": self.st["global_step"]}
        save_trunk(self.net, self.base, self.out / f"trunk_ep{epoch}.ckpt", meta)
        save_soft(self.soft, self.head_cfg, self.out / f"soft_ep{epoch}.pt", meta)


# --------------------------------------------------------------------------- #
def n_hits(sample):
    """Hit count of one dataset sample. The sample must stay referenced while its node store is
    read: PyG node stores hold only a weak reference to their HeteroData, and `num_nodes` needs
    it, so `dataset[i]["hit"].num_nodes` fails once the temporary sample is freed."""
    store = sample["hit"]
    n = store.num_nodes
    if n is None:
        raise SystemExit(f"cannot infer the hit count from keys {sorted(store.keys())}")
    return int(n)


def dev_identity(val_dataset, lo, hi, cache):
    """Validation samples [lo*8, hi*8) must be the dev-A cache's events, in order."""
    idx = range(lo * DEV_BATCH, hi * DEV_BATCH)
    cache_n = [int(ev["mask"].numel()) for ev in mb.iter_cache(cache, lo, hi)]
    data_n = [n_hits(val_dataset[i]) for i in idx]
    if cache_n != data_n:
        bad = next((i for i, (x, y) in enumerate(zip(cache_n, data_n)) if x != y), None)
        raise SystemExit(f"dev subset != dev cache: {len(data_n)} vs {len(cache_n)} events, "
                         f"first differing event {bad}")
    return list(idx)


def git_state(path):
    run = lambda *x: subprocess.run(["git", "-C", str(path), *x], capture_output=True,
                                    text=True).stdout.strip()
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--short"))}


def run_config(a, ckpt_sha, head_sha, git):
    """Everything that must be identical for a run to be resumed."""
    return {"arm": a.arm, "seed": a.seed, "lam": a.lam, "lr": a.lr, "head_lr": a.head_lr,
            "epochs": a.epochs, "batch_size": a.batch_size, "balance_frac": a.balance_frac,
            "limit_train_batches": a.limit_train_batches, "dev_batches": a.dev_batches,
            "no_soft_checkpoint": a.no_soft_checkpoint, "resume_every": a.resume_every,
            "sem_cons": a.sem_cons,
            # profile_steps is likewise absent: it only adds timers and device syncs.
            # num_workers / prefetch_factor are deliberately absent: they change who reads the
            # batches, not which batches, so a run may be resumed with a different loader setting.
            "data_path": str(Path(a.data_path).resolve()), "checkpoint_sha256": ckpt_sha,
            "head_sha256": head_sha, "git_commit": git["git_commit"], "git_dirty": git["git_dirty"],
            "soft_targets": SOFT_TARGETS, "decode": DECODE}


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
    p.add_argument("--sem-cons", type=float, default=0.0,
                   help="Stage 1a-R7: weight of the within-particle semantic-consistency loss "
                        "KL(sg(particle-mean p) || p_hit) on the trunk's semantic predictions (0 = off)")
    p.add_argument("--lr", type=float, default=1e-4, help="trunk (as the ep29 run)")
    p.add_argument("--head-lr", type=float, default=1e-3, help="soft layer (as Stage 1a)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--balance-frac", type=float, default=0.1)
    p.add_argument("--limit-train-batches", type=int, default=None, help="smoke tests only")
    p.add_argument("--no-soft-checkpoint", action="store_true",
                   help="disable per-event activation checkpointing of the soft layer")
    p.add_argument("--num-workers", type=int, default=0,
                   help="training DataLoader workers (0 = as all runs so far). Batches and their "
                        "order are identical for any value; only who reads them changes")
    p.add_argument("--prefetch-factor", type=int, default=2, help="batches prefetched per worker")
    p.add_argument("--profile-steps", type=int, default=0,
                   help="diagnostic: record a per-stage timing breakdown for the first N steps into "
                        "profile.csv. Adds device syncs, so those steps are slightly slower and must "
                        "not be used for throughput; the maths and the data order are unchanged")
    p.add_argument("--profile-split-backward", action="store_true",
                   help="diagnostic (arm C + --profile-steps only): backward the NuGraph and soft "
                        "losses separately to split the backward into trunk and soft layer. The two "
                        "graphs are disjoint in arm C, so the gradients are the same. Costs ~4.6 GB of "
                        "extra peak memory: forcing the trunk backward first keeps the whole soft graph "
                        "alive, which a single backward does not -- measurement only, never a real run")
    p.add_argument("--resume-every", type=int, default=100, help="steps between resume saves")
    p.add_argument("--resume", action="store_true", help="continue from --out/resume.pt")
    p.add_argument("--crash-at-step", type=int, default=None,
                   help="testing only: raise after this global step (resume verification)")
    a = p.parse_args()
    validate_profile_args(a)

    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    import nugraph as ng

    out = Path(a.out)
    if a.resume:
        if not (out / "resume.pt").exists():
            raise SystemExit(f"--resume: {out}/resume.pt not found")
    elif (out / "history.csv").exists() or (out / "resume.pt").exists():
        raise SystemExit(f"{out} already holds a run; use --resume or a new --out")
    out.mkdir(parents=True, exist_ok=True)
    ckpt_sha, head_sha = mb.sha256(a.checkpoint), mb.sha256(a.head)
    for path, got, want in ((a.checkpoint, ckpt_sha, a.checkpoint_sha256),
                            (a.head, head_sha, a.head_sha256)):
        if want and got != want:
            raise SystemExit(f"sha256 mismatch for {path}")
    torch.set_num_threads(4)                                # as tools/nugraph_train.py
    torch.manual_seed(a.seed)
    a.dev_lo, a.dev_hi = mb.parse_range(a.dev_batches)
    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)

    mat = sp._import_materialize()
    base = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    net = ng.models.NuGraph3.load_from_checkpoint(a.checkpoint, map_location="cpu").to(dev)
    ck = torch.load(a.head, map_location="cpu", weights_only=False)      # read only
    head_cfg = dict(ck["cfg"])
    soft = sp.build_model(head_cfg["hit_dim"], head_cfg["lat_dim"], head_cfg["width"],
                          head_cfg["r"])
    soft.load_state_dict(ck["state_dict"])
    soft.overlap = "sparse"
    soft = soft.to(dev)
    head_cfg.update(overlap="sparse", init_head=str(a.head), init_head_sha256=head_sha,
                    decode="v2", decode_kwargs=DECODE)

    nudata = ng.data.H5DataModule(a.data_path, batch_size=a.batch_size, model=ng.models.NuGraph3,
                                  shuffle="balance", balance_frac=a.balance_frac, num_workers=0,
                                  featext=False)
    dev_idx = dev_identity(nudata.val_dataset, a.dev_lo, a.dev_hi, a.dev_cache)
    dev_loader = DataLoader(Subset(nudata.val_dataset, dev_idx), batch_size=DEV_BATCH,
                            num_workers=0, shuffle=False, pin_memory=True)
    git = git_state(HERE)
    loop = Stage1bLoop(a, net, soft, head_cfg, base, mat, nudata.train_dataset,
                       nudata.train_datasize, dev_loader, dev, run_config(a, ckpt_sha, head_sha, git))

    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    if a.resume:
        s = loop.load_resume()
        manifest.setdefault("resumes", []).append(
            {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "from_saved_at": s["saved_at"],
             "state": dict(loop.st), "argv": sys.argv})
        print(f"resumed at epoch {loop.st['epoch']}, batch {loop.st['batches_done']}, "
              f"global step {loop.st['global_step']} (saved {s['saved_at']})", flush=True)
    else:
        manifest = {"args": {k: v for k, v in vars(a).items()}, "argv": sys.argv,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": loop.config,
                    "materialize_sha256": mb.sha256(mat.__file__), "nugraph_pkg": ng.__file__,
                    "layernorm_blocks": sum(hasattr(m, "norm") for m in net.core_net.children()),
                    "dev_events": len(dev_idx), "torch": torch.__version__,
                    "matmul_precision": {"baseline": "highest", "train_and_monitor": "high"},
                    "training_loop": "explicit loop with exact resume (no Lightning Trainer)",
                    "loader": {"num_workers": a.num_workers,
                               "prefetch_factor": a.prefetch_factor, "pin_memory": True},
                    "profile_steps": a.profile_steps,
                    "profile_split_backward": a.profile_split_backward}
    man_path.write_text(json.dumps(manifest, indent=2, default=str))

    if not loop.extra.get("baseline_done"):
        loop.baseline()
        loop.save_resume()
    torch.set_float32_matmul_precision("high")              # as tools/nugraph_train.py
    loop.run(crash_at_step=a.crash_at_step)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    man_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
