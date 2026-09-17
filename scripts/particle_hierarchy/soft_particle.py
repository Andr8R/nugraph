#!/usr/bin/env python
"""soft_particle.py -- Stage 1a: learned soft particle decoder on a FROZEN trunk.

Trains and evaluates a small decoder on top of cached NuGraph3 outputs (see
``mat_bench.py cache --store-hit-x``). The trunk, the OC heads and the frozen Stage-0 candidate
rule are fixed, so this isolates the decoder.

Per event:
  candidates   frozen Stage-0 rule (default quant:0.995:0.05:0.5, NMS t_d 0.3) -> K seeds
  membership   each foreground hit scores its M nearest candidates + a null option, from a
               LEARNED function of hit/candidate features (latent distance alone is ambiguous)
  refinement   R rounds: candidates aggregate their members, update state/centroid/existence,
               memberships are recomputed
  merge        learned "same true particle" score for candidate pairs -> connected components
  existence    learned per-candidate score; lets the model switch candidates off
  hard output  only at inference: merge, then assign each hit to its best surviving candidate

  train      train on a TRAIN-split cache, monitor on dev-A
  eval       evaluate a trained head on a cache range, writing the benchmark CSV schema
  selftest   synthetic end-to-end check (CPU, seconds) -- no cache needed

Data protocol (enforced by --batches guards in eval): train on the training split; develop on
dev-A 0:150; freeze; confirm once on dev-B 150:314; held-out 314:627 only afterwards.
"""
import argparse
import csv
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


mb = _load("mat_bench", HERE.parent / "dune_materialization" / "mat_bench.py")
s0 = _load("stage0_seed_diag", HERE / "stage0_seed_diag.py")

NAN = float("nan")
IGNORE = -100
DEFAULT_RULE = "quant:0.995:0.05:0.5"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(hit_dim, lat_dim, width, R, cfg=None):
    import torch
    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.R = R
            self.width = width
            self.hit_net = nn.Sequential(nn.Linear(hit_dim + lat_dim + 1, width), nn.Mish(),
                                         nn.Linear(width, width), nn.Mish())
            self.cand_net = nn.Sequential(nn.Linear(hit_dim + lat_dim + 1, width), nn.Mish(),
                                          nn.Linear(width, width), nn.Mish())
            # membership score of (hit, candidate) pairs
            self.pair_net = nn.Sequential(nn.Linear(2 * width + lat_dim + 2, width), nn.Mish(),
                                          nn.Linear(width, width), nn.Mish(),
                                          nn.Linear(width, 1))
            self.null_net = nn.Sequential(nn.Linear(width, width), nn.Mish(), nn.Linear(width, 1))
            self.update_net = nn.Sequential(nn.Linear(2 * width, width), nn.Mish(),
                                            nn.Linear(width, width), nn.Mish())
            self.exist_net = nn.Sequential(nn.Linear(width, width), nn.Mish(), nn.Linear(width, 1))
            self.offset_net = nn.Sequential(nn.Linear(width, width), nn.Mish(),
                                            nn.Linear(width, lat_dim))
            self.merge_net = nn.Sequential(nn.Linear(2 * width + lat_dim + 2, width), nn.Mish(),
                                           nn.Linear(width, 1))
            self.log_temp = nn.Parameter(torch.zeros(1))
            self.dist_only = False        # ablation switch

        def logits(self, h, c, ox, cen, nb, e):
            """Membership logits (n_hit, M+1): last column is the null option."""
            d = ox[:, None, :] - cen[nb]                      # (n, M, lat)
            dist = d.norm(dim=-1, keepdim=True)               # (n, M, 1)
            temp = self.log_temp.exp().clamp(1e-3, 1e3)
            geo = -(dist.squeeze(-1) ** 2) / temp + torch.nn.functional.logsigmoid(e)[nb]
            if self.dist_only:
                lg = geo
            else:
                feat = torch.cat([h[:, None, :].expand(-1, nb.size(1), -1), c[nb], d, dist,
                                  torch.nn.functional.logsigmoid(e)[nb].unsqueeze(-1)], dim=-1)
                lg = self.pair_net(feat).squeeze(-1) + geo
            return torch.cat([lg, self.null_net(h)], dim=1)

        def forward(self, hit_x, ox, beta, seed_idx, nb):
            """hit_x (n,F) ox (n,lat) beta (n,) seed_idx (K,) nb (n,M) candidate indices."""
            hin = torch.cat([hit_x, ox, beta[:, None]], dim=1)
            h = self.hit_net(hin)
            c = self.cand_net(hin[seed_idx])
            cen = ox[seed_idx].clone()
            e = self.exist_net(c).squeeze(-1)
            for _ in range(max(1, self.R)):
                lg = self.logits(h, c, ox, cen, nb, e)
                w = lg.softmax(dim=1)[:, :-1]                 # (n, M) membership to neighbours
                num = torch.zeros_like(c)
                den = torch.zeros(c.size(0), 1, device=c.device, dtype=c.dtype)
                src = w.unsqueeze(-1) * h[:, None, :]
                num.index_add_(0, nb.reshape(-1), src.reshape(-1, src.size(-1)))
                den.index_add_(0, nb.reshape(-1), w.reshape(-1, 1))
                agg = num / den.clamp(min=1e-6)
                c = c + self.update_net(torch.cat([c, agg], dim=1))
                e = self.exist_net(c).squeeze(-1)
                cen = cen + 0.1 * self.offset_net(c)
            lg = self.logits(h, c, ox, cen, nb, e)
            return lg, c, cen, e

        def merge_logits(self, c, cen, pairs, w=None, nb=None):
            """Symmetric same-particle score for candidate pairs (P,2)."""
            if pairs.numel() == 0:
                return torch.zeros(0, device=c.device)
            a, b = pairs[:, 0], pairs[:, 1]
            d = cen[a] - cen[b]
            ov = torch.zeros(pairs.size(0), 1, device=c.device)
            if w is not None and nb is not None:
                full = torch.zeros(w.size(0), c.size(0), device=c.device)
                full.scatter_(1, nb, w)
                ov = (full[:, a] * full[:, b]).sum(0).unsqueeze(-1)
            feat = torch.cat([c[a] + c[b], (c[a] - c[b]).abs(), d.abs(),
                              d.norm(dim=-1, keepdim=True), ov], dim=-1)
            return self.merge_net(feat).squeeze(-1)

    return Model()


# --------------------------------------------------------------------------- #
# per-event preparation (frozen candidates + targets)
# --------------------------------------------------------------------------- #
def prepare(ev, mat, rule_fn, M, t_d, t_assign, dev, need_hit_x=True):
    """Return a dict of tensors for one cached event, or None if unusable."""
    import torch
    fg, pop, y_i, ysem = s0.event_truth(ev)
    fg_idx = fg.nonzero(as_tuple=False).squeeze(1)
    if fg_idx.numel() == 0:
        return None
    ox = ev["ox"].to(dev)[fg_idx.to(dev)]
    beta = ev["beta"].to(dev)[fg_idx.to(dev)]
    tau = rule_fn(beta.double().cpu())
    cp, lab_fg, D = s0.seeds_and_labels(mat, ox, beta, tau, t_d, t_assign, dev)
    if cp.numel() == 0 or D is None:
        return None
    hit_x = None
    if need_hit_x:
        if "hit_x_fg" not in ev:
            raise SystemExit("cache has no 'hit_x_fg'; regenerate it with --store-hit-x")
        hit_x = ev["hit_x_fg"].to(dev).float()
        if hit_x.size(0) != fg_idx.numel():
            raise SystemExit("hit_x_fg rows do not match the foreground mask")
    K = cp.numel()
    m = min(M, K)
    nb = torch.topk(D, m, dim=1, largest=False).indices           # (n_fg, m)

    # ---- targets from truth (train/eval diagnostics only) ----
    seed_true, primary = s0._seed_truth(fg_idx, cp, y_i, ev["beta"])
    y_fg = y_i[fg_idx].to(dev)
    correct = torch.tensor([primary.get(int(t), -1) for t in y_fg.tolist()],
                           dtype=torch.long, device=dev)
    pop_fg = pop[fg_idx].to(dev)
    in_nb = (nb == correct[:, None])
    target = torch.full((fg_idx.numel(),), IGNORE, dtype=torch.long, device=dev)
    target[pop_fg & in_nb.any(dim=1)] = in_nb.float().argmax(dim=1)[pop_fg & in_nb.any(dim=1)]
    bkg_fg = (~pop_fg) & (y_fg < 0)
    target[bkg_fg] = m                                            # null column
    seed_true_d = seed_true.to(dev)
    exist_t = torch.zeros(K, device=dev)
    if primary:
        exist_t[torch.tensor(sorted(primary.values()), dtype=torch.long, device=dev)] = 1.0
    return {"fg_idx": fg_idx, "ox": ox, "beta": beta, "hit_x": hit_x, "cp": cp, "nb": nb,
            "m": m, "K": K, "D": D, "target": target, "correct": correct, "pop_fg": pop_fg,
            "seed_true": seed_true_d, "exist_t": exist_t, "hard_labels": lab_fg,
            "y_i": y_i, "y_sem": ysem, "fg": fg, "n_hit": fg.numel(),
            "batch": ev["batch"], "event": ev["event"]}


MAX_MERGE_K = 512          # above this, candidate-pair merging is skipped (guard)


def merge_pairs(K, dev, max_pairs=4096):
    import torch
    if K < 2 or K > MAX_MERGE_K:
        return torch.zeros(0, 2, dtype=torch.long, device=dev)
    a, b = torch.triu_indices(K, K, offset=1, device=dev)
    p = torch.stack([a, b], dim=1)
    if p.size(0) > max_pairs:
        p = p[torch.randperm(p.size(0), device=dev)[:max_pairs]]
    return p


def event_loss(model, d, w_exist=1.0, w_merge=1.0):
    import torch
    import torch.nn.functional as F
    lg, c, cen, e = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
    loss_mem = F.cross_entropy(lg, d["target"], ignore_index=IGNORE)
    if torch.isnan(loss_mem):
        loss_mem = lg.sum() * 0.0
    loss_ex = F.binary_cross_entropy_with_logits(e, d["exist_t"])
    pairs = merge_pairs(d["K"], lg.device)
    loss_mg = lg.sum() * 0.0
    if pairs.numel():
        st = d["seed_true"]
        tgt = ((st[pairs[:, 0]] == st[pairs[:, 1]]) & (st[pairs[:, 0]] >= 0)).float()
        w = lg.softmax(dim=1)[:, :-1]
        loss_mg = F.binary_cross_entropy_with_logits(
            model.merge_logits(c, cen, pairs, w, d["nb"]), tgt)
    total = loss_mem + w_exist * loss_ex + w_merge * loss_mg
    return total, {"mem": float(loss_mem.detach()), "exist": float(loss_ex.detach()),
                   "merge": float(loss_mg.detach())}


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def predict_labels(model, d, merge_thr=0.5, exist_thr=0.5, allow_null=True):
    """Hard labels over foreground hits (-1 = unassigned) + diagnostics.

    Inference only: runs under no_grad, so the returned tensors never carry gradients.
    """
    import torch
    with torch.no_grad():
        return _predict_labels(model, d, merge_thr, exist_thr, allow_null)


def _predict_labels(model, d, merge_thr, exist_thr, allow_null):
    import torch
    lg, c, cen, e = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
    w = lg.softmax(dim=1)
    m, K = d["m"], d["K"]
    pick = w.argmax(dim=1)
    top_cand = d["nb"].gather(1, w[:, :m].argmax(dim=1, keepdim=True)).squeeze(1)
    # merge candidates: connected components over accepted pairs
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = merge_pairs(K, lg.device, max_pairs=200_000)
    if pairs.numel():
        acc = torch.sigmoid(model.merge_logits(c, cen, pairs, w[:, :m], d["nb"])) > merge_thr
        for a, b in pairs[acc].tolist():
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    root = torch.tensor([find(k) for k in range(K)], dtype=torch.long, device=lg.device)
    alive = torch.sigmoid(e) > exist_thr
    keep = torch.zeros(K, dtype=torch.bool, device=lg.device)
    for k in range(K):                      # a group survives if any member is alive
        if alive[k]:
            keep[root[k]] = True
    lab = root[top_cand]
    if allow_null:
        lab = torch.where(pick == m, torch.full_like(lab, -1), lab)
    lab = torch.where(keep[lab.clamp(min=0)] | (lab < 0), lab, torch.full_like(lab, -1))
    # renumber to contiguous ids
    out = torch.full_like(lab, -1)
    uniq = torch.unique(lab[lab >= 0])
    for new, old in enumerate(uniq.tolist()):
        out[lab == old] = new
    diag = {"n_cand": K, "n_groups": int(uniq.numel()),
            "frac_null": float((pick == m).float().mean()),
            "pick_correct": float(((d["nb"].gather(1, w[:, :m].argmax(dim=1, keepdim=True)).squeeze(1)
                                    == d["correct"]) & d["pop_fg"]).sum()),
            "pop": int(d["pop_fg"].sum()),
            "oracle_in_nb": float(((d["nb"] == d["correct"][:, None]).any(dim=1) & d["pop_fg"]).sum())}
    return out, diag


# --------------------------------------------------------------------------- #
def to_device(d, dev):
    """Move one prepared event to `dev` (hit features are stored fp16, used fp32)."""
    import torch
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev).float() if k == "hit_x" else v.to(dev)
        else:
            out[k] = v
    return out


def to_store(d):
    """Detach one prepared event to CPU for cheap bulk storage."""
    import torch
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.half().cpu() if k == "hit_x" else v.cpu()
        else:
            out[k] = v
    out.pop("D", None)          # (n_fg x K) distances: recomputed only when needed
    return out


def iter_events(cache, lo, hi, mat, rule_fn, a, dev, need_hit_x=True, store=False):
    for ev in mb.iter_cache(cache, lo, hi):
        d = prepare(ev, mat, rule_fn, a.m, a.t_d, a.t_assign, dev, need_hit_x)
        if d is not None:
            yield to_store(d) if store else d


def cmd_train(a):
    import torch
    mat, dev = _setup(a)
    rule_fn = s0.make_rule(a.rule)
    torch.manual_seed(a.seed)
    train = list(iter_events(a.train_cache, *mb.parse_range(a.train_batches), mat, rule_fn, a,
                             dev, store=True))
    print(f"prepared {len(train)} training events", flush=True)
    if not train:
        raise SystemExit("no usable training events")
    model = build_model(train[0]["hit_x"].size(1), train[0]["ox"].size(1), a.width, a.r).to(dev)
    gb = sum(v.numel() * v.element_size() for d in train for v in d.values()
             if hasattr(v, "numel")) / 1e9
    print(f"  cached training tensors: {gb:.2f} GB (CPU)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    hist = []
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(len(train))
        tot, parts, n = 0.0, {"mem": 0.0, "exist": 0.0, "merge": 0.0}, 0
        opt.zero_grad()
        for j, i in enumerate(perm.tolist()):
            loss, p = event_loss(model, to_device(train[i], dev), a.w_exist, a.w_merge)
            (loss / a.accum).backward()
            tot += float(loss.detach()); n += 1
            for k in p: parts[k] += p[k]
            if (j + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        row = {"epoch": epoch, "loss": tot / n, **{k: v / n for k, v in parts.items()}}
        if a.dev_cache:
            row.update(evaluate(model, a, mat, rule_fn, dev, a.dev_cache, a.dev_batches,
                                out=None, tag=f"dev_epoch{epoch}"))
        hist.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in row.items()), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = {"rule": a.rule, "m": a.m, "r": a.r, "width": a.width, "lr": a.lr, "epochs": a.epochs,
           "w_exist": a.w_exist, "w_merge": a.w_merge, "seed": a.seed, "t_d": a.t_d,
           "t_assign": a.t_assign, "hit_dim": train[0]["hit_x"].size(1),
           "lat_dim": train[0]["ox"].size(1), "accum": a.accum, "max_merge_k": MAX_MERGE_K,
           "train_cache": a.train_cache, "train_batches": a.train_batches,
           "n_train_events": len(train)}
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, out / "head.pt")
    mb.write_rows(out / "train_history.csv", hist)
    (out / "train_meta.json").write_text(json.dumps({**cfg, **_prov(mat)}, indent=2))
    print(f"wrote {out}/head.pt")


def evaluate(model, a, mat, rule_fn, dev, cache, batches, out, tag):
    import torch
    import numpy as np
    model.eval()
    rows = []
    with torch.no_grad():
        for d in iter_events(cache, *mb.parse_range(batches), mat, rule_fn, a, dev):
            lab_fg, diag = predict_labels(model, d, a.merge_thr, a.exist_thr, not a.no_null)
            lab = torch.full((d["n_hit"],), -1, dtype=torch.long)
            lab[d["fg_idx"]] = lab_fg.cpu()
            keep = d["fg"] & (d["y_sem"] >= 0)
            q = mb.event_quality(lab, d["y_i"], d["y_sem"], keep)
            hard = torch.full((d["n_hit"],), -1, dtype=torch.long)
            hard[d["fg_idx"]] = d["hard_labels"].cpu()
            qh = mb.event_quality(hard, d["y_i"], d["y_sem"], keep)
            rows.append({"batch": d["batch"], "event": d["event"], "n_fg": int(d["fg_idx"].numel()),
                         "beta_max": float(d["beta"].max()), "n_true_all": q.get("n_true", NAN),
                         **{k: q.get(k, NAN) for k in ("ari_particle", "ari_inclusive", "purity",
                                                       "completeness", "n_pred", "noise_frac")},
                         "ari_particle_hardseed": qh.get("ari_particle", NAN),
                         "n_pred_hardseed": qh.get("n_pred", NAN), **diag})
    g = lambda c: np.array([float(r.get(c, NAN)) for r in rows], dtype=float)
    pop, corr, orc = g("pop").sum(), g("pick_correct").sum(), g("oracle_in_nb").sum()
    summary = {f"{tag}/ari_particle": float(np.nanmean(g("ari_particle"))),
               f"{tag}/ari_hardseed": float(np.nanmean(g("ari_particle_hardseed"))),
               f"{tag}/n_pred_med": float(np.nanmedian(g("n_pred"))),
               f"{tag}/groups_med": float(np.nanmedian(g("n_groups"))),
               f"{tag}/null_frac": float(np.nanmean(g("frac_null"))),
               f"{tag}/pick_acc": float(corr / max(pop, 1)),
               f"{tag}/pick_acc_of_local": float(corr / max(orc, 1)),
               f"{tag}/no_cluster": float(np.mean(g("n_groups") == 0))}
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        mb.write_rows(Path(out) / f"{tag}_events.csv", rows)
        (Path(out) / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
        low = g("beta_max") <= 0.5
        extra = {f"{tag}/ari_particle_lowbeta": float(np.nanmean(g("ari_particle")[low])) if low.any() else NAN,
                 f"{tag}/ari_hardseed_lowbeta": float(np.nanmean(g("ari_particle_hardseed")[low])) if low.any() else NAN,
                 f"{tag}/n_events": len(rows)}
        summary.update(extra)
        (Path(out) / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    return summary


def cmd_eval(a):
    import torch
    mat, dev = _setup(a)
    ck = torch.load(a.head, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    for k in ("rule", "m", "r", "t_d", "t_assign"):
        setattr(a, k if k != "r" else "r", cfg[k])
    lo, hi = mb.parse_range(a.batches)
    if (hi is None or hi > a.heldout_start) and not a.allow_heldout:
        raise SystemExit(f"--batches {a.batches} reaches the held-out set; use a development "
                         "split or pass --allow-heldout")
    model = build_model(cfg["hit_dim"], cfg["lat_dim"], cfg["width"], cfg["r"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    evaluate(model, a, mat, s0.make_rule(cfg["rule"]), dev, a.cache, a.batches, a.out, a.tag)


def cmd_selftest(a):
    """Synthetic end-to-end check: separable blobs must be learnable and decodable."""
    import torch
    dev = torch.device("cpu")
    torch.manual_seed(0)
    mat = _import_materialize()
    rule_fn = s0.make_rule("fixed1:0.5")
    events = []
    for e in range(12):
        K, n = 3, 60
        cen = torch.randn(K, 8) * 5
        y = torch.arange(K).repeat_interleave(n)
        ox = cen[y] + 0.3 * torch.randn(K * n, 8)
        beta = torch.rand(K * n) * 0.2
        beta[torch.arange(K) * n] = 0.9
        hit = torch.cat([torch.nn.functional.one_hot(y, K).float(), torch.randn(K * n, 5)], 1)
        ev = {"batch": 0, "event": e, "ox": ox, "beta": beta,
              "mask": torch.ones(K * n, dtype=torch.bool), "y_i": y.to(torch.int32),
              "y_sem": torch.zeros(K * n, dtype=torch.int8),
              "x_nugraph": torch.zeros(K * n, dtype=torch.int32), "hit_x_fg": hit.half()}
        d = prepare(ev, mat, rule_fn, 3, 0.3, float("inf"), dev)
        assert d is not None, "prepare returned None on synthetic event"
        events.append(d)
    model = build_model(events[0]["hit_x"].size(1), 8, 32, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for epoch in range(30):
        tot = 0.0
        for d in events:
            loss, _ = event_loss(model, d)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach())
        first = tot / len(events) if first is None else first
        last = tot / len(events)
    model.eval()
    aris = []
    with torch.no_grad():
        for d in events:
            lab, diag = predict_labels(model, d)
            lab_full = torch.full((d["n_hit"],), -1, dtype=torch.long)
            lab_full[d["fg_idx"]] = lab
            q = mb.event_quality(lab_full, d["y_i"], d["y_sem"], d["fg"] & (d["y_sem"] >= 0))
            aris.append(q["ari_particle"])
    ari = sum(aris) / len(aris)
    print(f"selftest: loss {first:.4f} -> {last:.4f}; mean ari_particle {ari:.4f}; "
          f"groups {diag['n_groups']} (truth 3)")
    ok = last < first and ari > 0.8
    print("selftest:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# --------------------------------------------------------------------------- #
def _import_materialize():
    import nugraph.models.nugraph3.materialize as mat
    return mat


def _prov(mat):
    repo = Path(mat.__file__).resolve().parents[4]
    git = lambda *x: subprocess.run(["git", "-C", str(repo), *x],
                                    capture_output=True, text=True).stdout.strip()
    return {"materialize_sha256": mb.sha256(mat.__file__), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--short")), "argv": sys.argv}


def _setup(a):
    import torch
    mat = _import_materialize()
    dev = torch.device(f"cuda:{a.device}" if a.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(json.dumps({**_prov(mat), "device": str(dev)}))
    return mat, dev


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--device", default="0")
        s.add_argument("--rule", default=DEFAULT_RULE)
        s.add_argument("--m", type=int, default=5)
        s.add_argument("--r", type=int, default=2)
        s.add_argument("--t-d", type=float, default=0.3)
        s.add_argument("--t-assign", type=float, default=float("inf"))
        s.add_argument("--merge-thr", type=float, default=0.5)
        s.add_argument("--exist-thr", type=float, default=0.5)
        s.add_argument("--no-null", action="store_true",
                       help="never predict the null option at inference")

    s = sub.add_parser("train"); common(s)
    s.add_argument("--train-cache", required=True)
    s.add_argument("--train-batches", default="0:")
    s.add_argument("--dev-cache", default=None)
    s.add_argument("--dev-batches", default="0:150")
    s.add_argument("--out", required=True)
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--width", type=int, default=64)
    s.add_argument("--accum", type=int, default=8)
    s.add_argument("--w-exist", type=float, default=1.0)
    s.add_argument("--w-merge", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval"); common(s)
    s.add_argument("--head", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:150")
    s.add_argument("--out", required=True)
    s.add_argument("--tag", default="dev")
    s.add_argument("--heldout-start", type=int, default=314)
    s.add_argument("--allow-heldout", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("selftest"); common(s)
    s.set_defaults(fn=cmd_selftest, device="cpu")

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
