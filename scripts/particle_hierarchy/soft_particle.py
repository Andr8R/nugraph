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

  train      train on a TRAIN-split cache, monitor on dev-A (saves last, best-on-dev-A, per epoch)
  eval       evaluate a trained head on a cache range, writing the benchmark CSV schema
  diagnose   dev-A only: decode ablations + membership/null/existence/merge statistics
  time       dev-A only: solo timing of the inference path (no truth work), beside the hard
             seed decoder, the Thread-A seed backend and canonical DBSCAN on the same events;
             checks that the timed labels equal the evaluated labels on every event
  selftest   synthetic end-to-end check (CPU, seconds) -- no cache needed

Targets (Run 2 correction; Run 1 used "primary" for both):
  --target own          a hit's membership is correct on ANY candidate seeded on its own particle
                        (duplicates are resolved by the merge head), not only the primary seed
  --exist-target real   existence = "candidate is seeded on a real particle" (spurious -> 0);
                        duplicates are handled by merge, not by switching them off
Decoders (--decode): v1 = Run 1 behaviour; v2 = hits go to their best SURVIVING candidate (as
documented above), at least one group always survives, merges only between candidates that
share hits' local neighbourhoods (no all-pairs single linkage).

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
COMEM_PAIRS = 2048                  # Stage 1a-R10: positive and negative pairs per event
COMEM_SEED = 20260928               # private pair-sampling generator = COMEM_SEED + head seed
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
            self.overlap = "dense"        # merge overlap Σ_i w_ia·w_ib: "dense" (Stage 1a) | "sparse"
            # Stage 1a-R8: the last 2k hit_x columns are [hit block (k) | candidate block (k)]; hit_net never sees
            # the candidate block, cand_net never sees the hit block. 0 = no split (every earlier head).
            self.split_k = 0

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

        def encode(self, hit_x, ox, beta, seed_idx):
            """Initial hit and candidate embeddings (with the R8 block isolation when split_k > 0)."""
            hin = torch.cat([hit_x, ox, beta[:, None]], dim=1)
            k = self.split_k
            if not k:
                return self.hit_net(hin), self.cand_net(hin[seed_idx])
            F = hit_x.size(1)
            hit_in, cand_in = hin.clone(), hin[seed_idx].clone()
            hit_in[:, F - k:F] = 0                       # hit encoder: no candidate block
            cand_in[:, F - 2 * k:F - k] = 0              # candidate encoder: no hit block
            return self.hit_net(hit_in), self.cand_net(cand_in)

        def forward(self, hit_x, ox, beta, seed_idx, nb):
            """hit_x (n,F) ox (n,lat) beta (n,) seed_idx (K,) nb (n,M) candidate indices."""
            h, c = self.encode(hit_x, ox, beta, seed_idx)
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
                if self.overlap == "sparse":
                    ov = sparse_overlap(w, nb, pairs, c.size(0)).unsqueeze(-1)
                else:
                    full = torch.zeros(w.size(0), c.size(0), device=c.device)
                    full.scatter_(1, nb, w)
                    ov = (full[:, a] * full[:, b]).sum(0).unsqueeze(-1)
            feat = torch.cat([c[a] + c[b], (c[a] - c[b]).abs(), d.abs(),
                              d.norm(dim=-1, keepdim=True), ov], dim=-1)
            return self.merge_net(feat).squeeze(-1)

    return Model()


def sparse_overlap(w, nb, pairs, K):
    """Σ_i w_ia·w_ib for candidate pairs, from each hit's M candidates only.

    Same quantity as the dense (n_hit x K) formula, since w_i is zero outside hit i's top-M,
    but memory O(n_hit·M²) instead of O(n_hit·pairs). Differentiable in w.
    """
    import torch
    m = nb.size(1)
    zero = torch.zeros(pairs.size(0), dtype=w.dtype, device=w.device)
    if m < 2 or pairs.numel() == 0:
        return zero
    i, j = torch.triu_indices(m, m, offset=1, device=nb.device)
    a, b = nb[:, i], nb[:, j]
    key = (torch.minimum(a, b) * K + torch.maximum(a, b)).reshape(-1)
    uk, inv = torch.unique(key, return_inverse=True)
    sums = torch.zeros(uk.numel(), dtype=w.dtype, device=w.device).index_add(
        0, inv, (w[:, i] * w[:, j]).reshape(-1))
    q = torch.minimum(pairs[:, 0], pairs[:, 1]) * K + torch.maximum(pairs[:, 0], pairs[:, 1])
    pos = torch.searchsorted(uk, q).clamp(max=uk.numel() - 1)
    return torch.where(uk[pos] == q, sums[pos], zero)


def primary_lookup(primary, y, dev):
    """correct[i] = primary candidate of hit i's particle, -1 if none (vectorised dict lookup)."""
    import torch
    out = torch.full((y.numel(),), -1, dtype=torch.long, device=dev)
    if not primary or y.numel() == 0:
        return out
    ids = torch.tensor(list(primary.keys()), dtype=torch.long, device=dev)
    vals = torch.tensor(list(primary.values()), dtype=torch.long, device=dev)
    order = ids.argsort()
    ids, vals = ids[order], vals[order]
    yy = y.to(dev).long()
    pos = torch.searchsorted(ids, yy).clamp(max=ids.numel() - 1)
    hit = ids[pos] == yy
    out[hit] = vals[pos][hit]
    return out


# --------------------------------------------------------------------------- #
# per-event preparation (frozen candidates + targets)
# --------------------------------------------------------------------------- #
def infer_inputs(ev, mat, rule_fn, M, t_d, t_assign, dev, need_hit_x=True, mark=None):
    """Inference-only inputs for one event (no truth): foreground, frozen candidates, top-M.

    Returns None if the event is unusable. `mark(name)` is called after each stage (timing).
    """
    import torch
    fg = ev["mask"].bool()
    fg_idx = fg.nonzero(as_tuple=False).squeeze(1)
    if fg_idx.numel() == 0:
        return None
    ox = ev["ox"].to(dev)[fg_idx.to(dev)]
    beta = ev["beta"].to(dev)[fg_idx.to(dev)]
    tau = rule_fn(beta.double().cpu())
    cp, lab_fg, D = s0.seeds_and_labels(mat, ox, beta, tau, t_d, t_assign, dev)
    if cp.numel() == 0 or D is None:
        return None
    if mark:
        mark("candidates")
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
    if mark:
        mark("topm")
    return {"fg": fg, "fg_idx": fg_idx, "ox": ox, "beta": beta, "hit_x": hit_x, "cp": cp,
            "lab_fg": lab_fg, "D": D, "K": K, "m": m, "nb": nb}


def prepare(ev, mat, rule_fn, M, t_d, t_assign, dev, need_hit_x=True):
    """Return a dict of tensors for one cached event (inference inputs + truth targets),
    or None if unusable."""
    import torch
    fg, pop, y_i, ysem = s0.event_truth(ev)
    x = infer_inputs(ev, mat, rule_fn, M, t_d, t_assign, dev, need_hit_x)
    if x is None:
        return None
    fg_idx, ox, beta, hit_x, cp = x["fg_idx"], x["ox"], x["beta"], x["hit_x"], x["cp"]
    lab_fg, D, K, m, nb = x["lab_fg"], x["D"], x["K"], x["m"], x["nb"]

    # ---- targets from truth (train/eval diagnostics only) ----
    seed_true, primary = s0._seed_truth(fg_idx, cp, y_i, ev["beta"])
    y_fg = y_i[fg_idx].to(dev)
    correct = primary_lookup(primary, y_fg, dev)
    pop_fg = pop[fg_idx].to(dev)
    in_nb = (nb == correct[:, None])
    target = torch.full((fg_idx.numel(),), IGNORE, dtype=torch.long, device=dev)
    target[pop_fg & in_nb.any(dim=1)] = in_nb.float().argmax(dim=1)[pop_fg & in_nb.any(dim=1)]
    bkg_fg = (~pop_fg) & (y_fg < 0)
    target[bkg_fg] = m                                            # null column
    seed_true_d = seed_true.to(dev)
    # any candidate seeded on the hit's own particle (primary or duplicate)
    own = (seed_true_d[nb] == y_fg[:, None]) & (pop_fg & (y_fg >= 0))[:, None]
    exist_t = torch.zeros(K, device=dev)
    if primary:
        exist_t[torch.tensor(sorted(primary.values()), dtype=torch.long, device=dev)] = 1.0
    exist_real = (seed_true_d >= 0).float()
    return {"fg_idx": fg_idx, "ox": ox, "beta": beta, "hit_x": hit_x, "cp": cp, "nb": nb,
            "m": m, "K": K, "D": D, "target": target, "correct": correct, "pop_fg": pop_fg,
            "own": own, "exist_real": exist_real,
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


def candidate_pairs(nb, K, scope, dev, max_pairs=4096):
    """Candidate pairs for the merge head.

    all:   every pair (random subset above max_pairs) -- Run 1.
    local: pairs that appear together in some hit's top-M list, i.e. candidates that compete
           for the same hits. Far pairs can then never be merged.
    """
    import torch
    if scope == "all":
        return merge_pairs(K, dev, max_pairs)
    if scope != "local":
        raise ValueError(f"unknown merge scope {scope!r}")
    m = nb.size(1)
    if K < 2 or K > MAX_MERGE_K or m < 2:
        return torch.zeros(0, 2, dtype=torch.long, device=dev)
    i, j = torch.triu_indices(m, m, offset=1, device=nb.device)
    a, b = nb[:, i].reshape(-1), nb[:, j].reshape(-1)
    key = torch.unique(torch.minimum(a, b) * K + torch.maximum(a, b))
    p = torch.stack([key // K, key % K], dim=1).to(dev)
    if p.size(0) > max_pairs:
        p = p[torch.randperm(p.size(0), device=dev)[:max_pairs]]
    return p


def pair_targets(d, pairs):
    """1 where both candidates are seeded on the same real particle."""
    st = d["seed_true"]
    return ((st[pairs[:, 0]] == st[pairs[:, 1]]) & (st[pairs[:, 0]] >= 0)).float()


def membership_loss(lg, d, target):
    import torch
    import torch.nn.functional as F
    if target == "primary":
        loss = F.cross_entropy(lg, d["target"], ignore_index=IGNORE)
        return lg.sum() * 0.0 if torch.isnan(loss) else loss
    if target != "own":
        raise ValueError(f"unknown membership target {target!r}")
    # multi-positive: -log(sum of the probability on the allowed columns)
    pos = torch.zeros_like(lg, dtype=torch.bool)
    pos[:, :-1] = d["own"]
    pos[d["target"] == d["m"], -1] = True                     # background -> null
    valid = pos.any(dim=1)
    if not bool(valid.any()):
        return lg.sum() * 0.0
    lv = lg[valid]
    return (lv.logsumexp(dim=1) - lv.masked_fill(~pos[valid], float("-inf")).logsumexp(dim=1)).mean()


def event_loss(model, d, w_exist=1.0, w_merge=1.0, target="primary", exist_target="primary",
               merge_scope="all", comem=0.0, comem_gen=None, comem_pairs=COMEM_PAIRS):
    """Stage-1a loss of one event. With `comem_gen` (Stage 1a-R10) the co-membership term is computed and logged;
    it is added to the loss only when comem > 0, so a lambda = 0 control follows the identical path."""
    loss_mem, loss_ex, loss_mg, lg = _event_loss_terms(model, d, target, exist_target, merge_scope)
    total = loss_mem + w_exist * loss_ex + w_merge * loss_mg
    parts = {"mem": float(loss_mem.detach()), "exist": float(loss_ex.detach()), "merge": float(loss_mg.detach())}
    if comem_gen is not None:
        i, j, same = comem_sample(d, comem_pairs, comem_gen)
        lc = comem_loss(lg, d, i, j, same)
        parts["comem"] = float(lc.detach())
        if comem:
            total = total + comem * lc
    return total, parts


COMEM_EPS = 1e-6


def comem_sample(d, n_pairs, gen):
    """Stage 1a-R10: same-particle and different-particle pairs of POPULATION hits (reachable or not).

    Positives: anchor uniform among population hits whose particle has >= 2 of them, partner uniform among the rest
    of its particle. Negatives: both uniform among population hits, kept if their particles differ. All draws use the
    private CPU generator `gen`, never the global torch RNG. Returns fg-row indices (i, j) and a bool `same`.
    """
    import torch
    pop = d["pop_fg"].cpu()
    idx = pop.nonzero(as_tuple=False).squeeze(1)
    empty = torch.zeros(0, dtype=torch.long)
    if idx.numel() < 2:
        return empty, empty, torch.zeros(0, dtype=torch.bool)
    pid = d["y_i"].cpu()[d["fg_idx"].cpu()][idx].long()
    pid, order = torch.sort(pid, stable=True)
    idx = idx[order]
    uniq, counts = torch.unique_consecutive(pid, return_counts=True)
    start = counts.cumsum(0) - counts
    part = torch.repeat_interleave(torch.arange(uniq.numel()), counts)       # particle slot of each sorted hit
    elig = (counts[part] >= 2).nonzero(as_tuple=False).squeeze(1)
    pi = pj = empty
    if elig.numel():
        a = elig[torch.randint(elig.numel(), (n_pairs,), generator=gen)]
        c = counts[part[a]]
        off = (torch.rand(n_pairs, generator=gen) * (c - 1)).long().clamp(max=c - 2)
        b = start[part[a]] + off
        b = b + (b >= a).long()                                               # skip the anchor itself
        pi, pj = idx[a], idx[b]
    na, nb_ = (torch.randint(idx.numel(), (2, 2 * n_pairs), generator=gen))
    keep = (pid[na] != pid[nb_]).nonzero(as_tuple=False).squeeze(1)[:n_pairs]
    ni, nj = idx[na[keep]], idx[nb_[keep]]
    same = torch.cat([torch.ones(pi.numel(), dtype=torch.bool), torch.zeros(ni.numel(), dtype=torch.bool)])
    return torch.cat([pi, ni]), torch.cat([pj, nj]), same


def comem_overlap(lg, nb, i, j):
    """s_ij = sum_k w_i(k) w_j(k) over candidates in both hits' neighbour lists (null mass excluded)."""
    w = lg.softmax(dim=1)[:, :nb.size(1)]
    eq = (nb[i][:, :, None] == nb[j][:, None, :]).to(w.dtype)
    return (w[i][:, :, None] * w[j][:, None, :] * eq).sum(dim=(1, 2))


def comem_loss(lg, d, i, j, same):
    """-log s on same-particle pairs + -log(1 - s) on different-particle pairs (each a mean; 0 if none)."""
    if i.numel() == 0:
        return lg.sum() * 0.0
    dev = lg.device
    i, j, same = i.to(dev), j.to(dev), same.to(dev)
    s = comem_overlap(lg, d["nb"], i, j)
    out = lg.sum() * 0.0
    if bool(same.any()):
        out = out - (s[same] + COMEM_EPS).log().mean()
    if bool((~same).any()):
        out = out - (1 - s[~same] + COMEM_EPS).log().mean()
    return out


def event_loss_terms(model, d, target="primary", exist_target="primary", merge_scope="all"):
    """(membership, existence, merge) losses of one event as tensors."""
    return _event_loss_terms(model, d, target, exist_target, merge_scope)[:3]


def _event_loss_terms(model, d, target, exist_target, merge_scope):
    import torch.nn.functional as F
    lg, c, cen, e = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
    loss_mem = membership_loss(lg, d, target)
    ex_t = {"primary": d["exist_t"], "real": d["exist_real"]}[exist_target]
    loss_ex = F.binary_cross_entropy_with_logits(e, ex_t)
    pairs = candidate_pairs(d["nb"], d["K"], merge_scope, lg.device)
    loss_mg = lg.sum() * 0.0
    if pairs.numel():
        tgt = pair_targets(d, pairs)
        w = lg.softmax(dim=1)[:, :-1]
        loss_mg = F.binary_cross_entropy_with_logits(
            model.merge_logits(c, cen, pairs, w, d["nb"]), tgt)
    return loss_mem, loss_ex, loss_mg, lg


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
# decode presets. v1 = Run 1. v2 = the documented decode (best surviving candidate), at least
# one surviving group, local merges.
PRESETS = {
    "v1": dict(merge_thr=0.5, exist_thr=0.5, allow_null=True, merge_scope="all",
               reassign=False, keep_one=False),
    "v2": dict(merge_thr=0.5, exist_thr=0.5, allow_null=True, merge_scope="local",
               reassign=True, keep_one=True),
}


def predict_labels(model, d, merge_thr=0.5, exist_thr=0.5, allow_null=True, merge_scope="all",
                   reassign=False, keep_one=False, nearest=False):
    """Hard labels over foreground hits (-1 = unassigned) + diagnostics.

    merge_thr / exist_thr None disable merging / existence. reassign: a hit goes to its best
    candidate whose group survives (else to the best candidate, dropped if its group died).
    keep_one: if no group survives existence, keep the most confident one. nearest: ignore the
    learned memberships and take the nearest candidate (reproduces the hard seed decoder).
    Inference only: runs under no_grad, so the returned tensors never carry gradients.
    """
    import torch
    with torch.no_grad():
        return _predict_labels(model, d, merge_thr, exist_thr, allow_null, merge_scope,
                               reassign, keep_one, nearest)


def _components(K, pairs, dev):
    """Connected components (union-find, smallest index as root) over accepted pairs."""
    import torch
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return torch.tensor([find(k) for k in range(K)], dtype=torch.long, device=dev)


def _predict_labels(model, d, merge_thr, exist_thr, allow_null, merge_scope, reassign, keep_one,
                    nearest):
    out, x = _decode(model, d, merge_thr, exist_thr, allow_null, merge_scope, reassign, keep_one,
                     nearest)
    m, K, nb, w, raw_best = d["m"], d["K"], d["nb"], x["w"], x["raw_best"]
    pop = d["pop_fg"]
    diag = {"n_cand": K, "n_keep": x["n_keep"], "n_groups": x["n_groups"],
            "frac_null": float((w.argmax(dim=1) == m).float().mean()),
            "pick_correct": float(((nb.gather(1, raw_best).squeeze(1) == d["correct"]) & pop).sum()),
            "pop": int(pop.sum()),
            "oracle_in_nb": float(((nb == d["correct"][:, None]).any(dim=1) & pop).sum())}
    if "own" in d:
        diag["pick_own"] = float(d["own"].gather(1, raw_best).squeeze(1).sum())
        diag["own_in_nb"] = float(d["own"].any(dim=1).sum())
    return out, diag


def _decode(model, d, merge_thr, exist_thr, allow_null, merge_scope, reassign, keep_one, nearest,
            mark=None):
    """Hard labels from inference inputs only (no truth). `mark(name)` is called after the
    forward pass, after merging and at the end (timing)."""
    import torch
    lg, c, cen, e = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
    if mark:
        mark("forward")
    w = lg.softmax(dim=1)
    m, K, nb = d["m"], d["K"], d["nb"]
    dev = lg.device
    raw_best = w[:, :m].argmax(dim=1, keepdim=True)
    root = torch.arange(K, device=dev)
    if merge_thr is not None:
        pairs = candidate_pairs(nb, K, merge_scope, dev, max_pairs=200_000)
        if pairs.numel():
            acc = torch.sigmoid(model.merge_logits(c, cen, pairs, w[:, :m], nb)) > merge_thr
            root = _components(K, pairs[acc], dev)
    if mark:
        mark("merge")
    alive = (torch.ones(K, dtype=torch.bool, device=dev) if exist_thr is None
             else torch.sigmoid(e) > exist_thr)
    keep = torch.zeros(K, dtype=torch.bool, device=dev)
    keep[root[alive]] = True                   # a group survives if any member is alive
    if keep_one and K > 0 and not bool(keep.any()):
        keep[root[e.argmax()]] = True
    score = w[:, :m]
    if nearest:                                 # nb columns are sorted by distance
        score = -torch.arange(m, device=dev, dtype=w.dtype).expand(nb.size(0), m)
    has = torch.ones(nb.size(0), dtype=torch.bool, device=dev)
    if reassign:
        ok = keep[root[nb]]
        score = score.masked_fill(~ok, float("-inf"))
        has = ok.any(dim=1)
    best = score.argmax(dim=1, keepdim=True)
    lab = root[nb.gather(1, best).squeeze(1)]
    lab = torch.where(has & keep[lab], lab, torch.full_like(lab, -1))
    if allow_null:                              # null beats the chosen candidate
        null_wins = w[:, m] > w[:, :m].gather(1, best).squeeze(1)
        lab = torch.where(null_wins, torch.full_like(lab, -1), lab)
    # renumber to contiguous ids
    out = torch.full_like(lab, -1)
    uniq = torch.unique(lab[lab >= 0])
    for new, old in enumerate(uniq.tolist()):
        out[lab == old] = new
    if mark:
        mark("decode")
    return out, {"w": w, "raw_best": raw_best, "n_keep": int(keep.sum()),
                 "n_groups": int(uniq.numel())}


def infer_labels(model, d, mark=None, **decode_kw):
    """Inference-only hard labels over foreground hits (the timed path)."""
    import torch
    kw = dict(merge_thr=0.5, exist_thr=0.5, allow_null=True, merge_scope="all", reassign=False,
              keep_one=False, nearest=False)
    kw.update(decode_kw)
    with torch.no_grad():
        return _decode(model, d, mark=mark, **kw)[0]


def decode_kwargs(a):
    """Decode preset from --decode, with the explicit threshold / null flags as overrides."""
    kw = dict(PRESETS[getattr(a, "decode", "v1")])
    if getattr(a, "merge_thr", None) is not None:
        kw["merge_thr"] = a.merge_thr
    if getattr(a, "exist_thr", None) is not None:
        kw["exist_thr"] = a.exist_thr
    if getattr(a, "no_null", False):
        kw["allow_null"] = False
    if getattr(a, "no_merge", False):          # Stage 1a-R11: the decode without the learned merge step
        kw["merge_thr"] = None
    return kw


# --------------------------------------------------------------------------- #
# diagnostics (dev-A only)
# --------------------------------------------------------------------------- #
_OFF = dict(merge_thr=None, exist_thr=None, allow_null=False)
VARIANTS = [                                   # (name, predict_labels kwargs, force dist_only)
    ("v1", PRESETS["v1"], False),
    ("v1_nonull", {**PRESETS["v1"], "allow_null": False}, False),
    ("v1_noexist", {**PRESETS["v1"], "exist_thr": None}, False),
    ("v1_nomerge", {**PRESETS["v1"], "merge_thr": None}, False),
    ("v1_localmerge", {**PRESETS["v1"], "merge_scope": "local"}, False),
    ("v1_reassign", {**PRESETS["v1"], "reassign": True, "keep_one": True}, False),
    ("v2", PRESETS["v2"], False),
    ("v2_nonull", {**PRESETS["v2"], "allow_null": False}, False),
    ("v2_nomerge", {**PRESETS["v2"], "merge_thr": None}, False),
    ("v2_nonull_nomerge", {**PRESETS["v2"], "allow_null": False, "merge_thr": None}, False),
    ("pick_merge", {**PRESETS["v2"], "allow_null": False, "exist_thr": None}, False),  # pick + merge only
    ("pick_only", _OFF, False),                # learned membership, nothing else
    ("pick_only_dist", _OFF, True),            # same weights, pair_net switched off
    ("nearest", {**_OFF, "nearest": True}, False),   # must equal the hard seed decoder
]


def head_stats(model, d):
    """Per-event counts for membership, null, existence and merge (summed over events)."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        lg, c, cen, e = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
        m, K, nb, own = d["m"], d["K"], d["nb"], d["own"]
        w = lg.softmax(dim=1)
        pop, bkg = d["pop_fg"], d["target"] == m
        best = w[:, :m].argmax(dim=1, keepdim=True)
        null = w.argmax(dim=1) == m
        p = torch.sigmoid(e)
        alive = p > 0.5
        prim_t, real_t = d["exist_t"], d["exist_real"]
        s = {"events": 1, "n_fg": int(pop.numel()), "n_pop": int(pop.sum()), "n_bkg": int(bkg.sum()),
             "prim_local": int(((nb == d["correct"][:, None]).any(dim=1) & pop).sum()),
             "own_local": int(own.any(dim=1).sum()),
             "pick_prim": int(((nb.gather(1, best).squeeze(1) == d["correct"]) & pop).sum()),
             "pick_own": int(own.gather(1, best).squeeze(1).sum()),
             "near_prim": int(((nb[:, 0] == d["correct"]) & pop).sum()),
             "near_own": int(own[:, 0].sum()),
             "null_pop": int((null & pop).sum()), "null_bkg": int((null & bkg).sum()),
             "K": K, "p_sum": float(p.sum()), "alive": int(alive.sum()),
             "prim": float(prim_t.sum()), "real": float(real_t.sum()),
             "alive_prim": int((alive & (prim_t > 0)).sum()),
             "alive_real": int((alive & (real_t > 0)).sum()),
             "bce_prim": float(F.binary_cross_entropy_with_logits(e, prim_t, reduction="sum")),
             "bce_real": float(F.binary_cross_entropy_with_logits(e, real_t, reduction="sum")),
             "all_dead": int(not bool(alive.any())),
             "real_particles": int(torch.unique(d["seed_true"][d["seed_true"] >= 0]).numel())}
        for scope in ("all", "local"):
            pairs = candidate_pairs(nb, K, scope, lg.device, max_pairs=200_000)
            s[f"{scope}_pairs"] = int(pairs.size(0))
            if pairs.numel():
                tgt = pair_targets(d, pairs)
                ml = model.merge_logits(c, cen, pairs, w[:, :m], nb)
                acc = torch.sigmoid(ml) > 0.5
                s[f"{scope}_pos"] = float(tgt.sum())
                s[f"{scope}_acc"] = int(acc.sum())
                s[f"{scope}_acc_pos"] = int((acc & (tgt > 0)).sum())
                s[f"{scope}_bce"] = float(F.binary_cross_entropy_with_logits(ml, tgt, reduction="sum"))
                s[f"{scope}_groups"] = int(torch.unique(_components(K, pairs[acc], lg.device)).numel())
            else:
                s.update({f"{scope}_pos": 0.0, f"{scope}_acc": 0, f"{scope}_acc_pos": 0,
                          f"{scope}_bce": 0.0, f"{scope}_groups": K})
    return s


def summarize_stats(s):
    """Rates from summed head_stats counts; each BCE is paired with its constant-prior value."""
    def r(a, b):
        return a / b if b else NAN

    def h(q):
        return 0.0 if q <= 0 or q >= 1 else -(q * math.log(q) + (1 - q) * math.log(1 - q))

    out = {"events": s["events"],
           "membership": {
               "bkg_share_of_fg": r(s["n_bkg"], s["n_fg"]),
               "primary_in_topM": r(s["prim_local"], s["n_pop"]),
               "own_in_topM": r(s["own_local"], s["n_pop"]),
               "learned_pick_primary_of_local": r(s["pick_prim"], s["prim_local"]),
               "nearest_pick_primary_of_local": r(s["near_prim"], s["prim_local"]),
               "learned_pick_own_of_local": r(s["pick_own"], s["own_local"]),
               "nearest_pick_own_of_local": r(s["near_own"], s["own_local"])},
           "null": {"p_null_given_pop": r(s["null_pop"], s["n_pop"]),
                    "p_null_given_bkg": r(s["null_bkg"], s["n_bkg"])},
           "existence": {
               "cands_per_event": r(s["K"], s["events"]),
               "primary_rate": r(s["prim"], s["K"]), "real_rate": r(s["real"], s["K"]),
               "mean_p": r(s["p_sum"], s["K"]), "alive_rate": r(s["alive"], s["K"]),
               "events_all_dead": r(s["all_dead"], s["events"]),
               "alive_precision_primary": r(s["alive_prim"], s["alive"]),
               "alive_recall_primary": r(s["alive_prim"], s["prim"]),
               "alive_precision_real": r(s["alive_real"], s["alive"]),
               "alive_recall_real": r(s["alive_real"], s["real"]),
               "bce_primary": r(s["bce_prim"], s["K"]), "prior_bce_primary": h(r(s["prim"], s["K"])),
               "bce_real": r(s["bce_real"], s["K"]), "prior_bce_real": h(r(s["real"], s["K"]))},
           "merge": {"real_particles_per_event": r(s["real_particles"], s["events"])}}
    for scope in ("all", "local"):
        n = s[f"{scope}_pairs"]
        out["merge"][scope] = {
            "pairs_per_event": r(n, s["events"]), "pos_rate": r(s[f"{scope}_pos"], n),
            "accept_rate": r(s[f"{scope}_acc"], n),
            "precision": r(s[f"{scope}_acc_pos"], s[f"{scope}_acc"]),
            "recall": r(s[f"{scope}_acc_pos"], s[f"{scope}_pos"]),
            "bce": r(s[f"{scope}_bce"], n), "prior_bce": h(r(s[f"{scope}_pos"], n)),
            "groups_per_event_after_merge": r(s[f"{scope}_groups"], s["events"])}
    return out


def diagnose_event(model, d, variants=None):
    """One row per decode variant for this event, each paired with the hard seed decoder."""
    import torch
    keep = d["fg"] & (d["y_sem"] >= 0)
    mp = keep & (d["y_i"] >= 0)
    hard = torch.full((d["n_hit"],), -1, dtype=torch.long)
    hard[d["fg_idx"]] = d["hard_labels"].cpu()
    qh = mb.event_quality(hard, d["y_i"], d["y_sem"], keep)
    base = {"batch": d["batch"], "event": d["event"], "n_fg": int(d["fg_idx"].numel()),
            "beta_max": float(d["beta"].max()), "K": d["K"],
            "ari_hardseed": qh.get("ari_particle", NAN), "n_pred_hardseed": qh.get("n_pred", NAN)}
    rows, trained = [], model.dist_only
    try:
        for name, kw, dist in (variants or VARIANTS):
            model.dist_only = trained or dist
            lab_fg, diag = predict_labels(model, d, **kw)
            lab = torch.full((d["n_hit"],), -1, dtype=torch.long)
            lab[d["fg_idx"]] = lab_fg.cpu()
            q = mb.event_quality(lab, d["y_i"], d["y_sem"], keep)
            rows.append({**base, "variant": name, "ari_particle": q.get("ari_particle", NAN),
                         "n_pred": q.get("n_pred", NAN), "n_true": q.get("n_true", NAN),
                         "n_groups": diag["n_groups"], "n_keep": diag["n_keep"],
                         "pop_unassigned": (float((lab[mp] < 0).float().mean())
                                            if bool(mp.any()) else NAN)})
    finally:
        model.dist_only = trained
    return rows


def summarize_variants(rows):
    import numpy as np
    out = {}
    for name in dict.fromkeys(r["variant"] for r in rows):
        rr = [r for r in rows if r["variant"] == name]
        g = lambda c: np.array([float(r[c]) for r in rr], dtype=float)
        ari, hs, npred, ntrue = g("ari_particle"), g("ari_hardseed"), g("n_pred"), g("n_true")
        ok = np.isfinite(ari) & np.isfinite(hs)
        dl = ari[ok] - hs[ok]
        groups, nkeep = g("n_groups"), g("n_keep")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = npred / ntrue
        out[name] = {"ari_particle": float(np.nanmean(ari)), "ari_hardseed": float(np.nanmean(hs)),
                     "delta": float(dl.mean()) if dl.size else NAN,
                     "delta_se": float(dl.std(ddof=1) / np.sqrt(dl.size)) if dl.size > 1 else NAN,
                     "pred_true_med": float(np.nanmedian(ratio)),
                     "groups_med": float(np.nanmedian(groups)),
                     "no_cluster": float(np.mean(groups == 0)),
                     "no_cluster_all_dead": float(np.mean((groups == 0) & (nkeep == 0))),
                     "pop_unassigned": float(np.nanmean(g("pop_unassigned"))),
                     "n_events": int(ok.sum())}
    return out


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


CLASS_BUCKETS = 8                   # 7 semantic classes + "unlabelled" (y_sem < 0)
CLASS_MODES = ("none", "zeros", "oracle", "pred",
               "s_zeros", "s_oracle", "s_oracle_seed", "s_oracle_hit", "s_pred_hard",    # s_*: R8 split blocks
               "trueclass", "cons", "fix50", "fixtrack")                                 # R9 label arms
LABEL_MODES = ("trueclass", "cons", "fix50", "fixtrack")
READOUT_MODES = ("pred", "s_pred_hard") + LABEL_MODES
ORACLE_MODES = ("oracle", "s_oracle", "s_oracle_seed", "s_oracle_hit") + LABEL_MODES     # truth-derived inputs
TRACK_CLASSES = (0, 1, 2, 3)        # pion, muon, kaon, hadron
FIX_SEED = 20260927
SPLIT_MODES = {  # (hit block, candidate block): None = zeros, "true" = true one-hot, "hard" = argmax one-hot
    "s_zeros": (None, None), "s_oracle": ("true", "true"), "s_oracle_seed": (None, "true"),
    "s_oracle_hit": ("true", None), "s_pred_hard": ("hard", "hard")}
SEMANTIC_KEYS = ("semantic_decoder.net.weight", "semantic_decoder.net.bias")
DIFFUSE = 6
_READOUT = {}


def semantic_readout(cache):
    """(W, b, sha256) of the semantic head of the checkpoint that produced `cache`.

    NuGraph3's semantic head is one Linear on h.x, and the cache stores h.x of foreground hits, so the
    trunk's predicted class logits are exactly W @ hit_x + b. The checkpoint is verified against the
    cache manifest's checkpoint_sha256 (loaded once per process).
    """
    import torch
    man = json.loads((Path(cache) / "manifest.json").read_text())
    ckpt, want = man["checkpoint"], man["checkpoint_sha256"]
    if ckpt not in _READOUT:
        got = mb.sha256(ckpt)
        if got != want:
            raise SystemExit(f"{ckpt}: sha256 {got} != cache manifest {want}")
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        _READOUT[ckpt] = (sd[SEMANTIC_KEYS[0]].float(), sd[SEMANTIC_KEYS[1]].float(), got)
    return _READOUT[ckpt]


def _manifest_sha(cache):
    """checkpoint_sha256 recorded in a cache manifest (None if the cache has no manifest, e.g. tests)."""
    p = Path(cache) / "manifest.json"
    return json.loads(p.read_text()).get("checkpoint_sha256") if p.exists() else None


def predicted_logits(hit_x, readout):
    W, b, _ = readout
    return hit_x.float() @ W.to(hit_x.device).T + b.to(hit_x.device)


def class_channels(d, mode, readout=None):
    """Extra per-hit input channels for the class-information probe (Stage 1a-R5).

    zeros  -- CTRL arm: the channels exist (same architecture, same init draws) but carry nothing.
    oracle -- ORACLE UPPER BOUND, not deployable: one-hot of each foreground hit's TRUE semantic
              class; candidates inherit their seed hit's class through the candidate encoder.
    pred   -- deployable (Stage 1a-R6): softmax of the trunk's own semantic logits, recomputed from the
              cached hit embedding (`readout` = semantic_readout(cache)), plus a zero 8th channel.
    """
    import torch
    n = d["fg_idx"].numel()
    dev = d["hit_x"].device
    if mode == "zeros":
        return torch.zeros(n, CLASS_BUCKETS, device=dev, dtype=d["hit_x"].dtype)
    if mode == "oracle":
        ys = d["y_sem"][d["fg_idx"].cpu()].long()
        ys = torch.where(ys < 0, torch.full_like(ys, CLASS_BUCKETS - 1), ys)
        return torch.nn.functional.one_hot(ys, CLASS_BUCKETS).to(device=dev, dtype=d["hit_x"].dtype)
    if mode in SPLIT_MODES:
        return split_channels(d, mode, readout)
    if mode in LABEL_MODES:
        ys = label_classes(d, mode, readout)
        return torch.nn.functional.one_hot(ys, CLASS_BUCKETS).to(device=dev, dtype=d["hit_x"].dtype)
    if mode == "pred":
        if readout is None:
            raise ValueError("class_channels('pred') needs the cache's semantic read-out")
        p = predicted_logits(d["hit_x"], readout).softmax(dim=1)
        pad = torch.zeros(n, CLASS_BUCKETS - p.size(1), device=dev, dtype=p.dtype)
        return torch.cat([p, pad], dim=1).to(d["hit_x"].dtype)
    raise ValueError(f"unknown class_channels mode {mode!r}")


def label_classes(d, mode, readout=None):
    """Stage 1a-R9 channel class per foreground hit (CPU long tensor; ORACLE UPPER BOUNDS, all use truth).

    argmax = the trunk's own class (exact read-out of h.x). Hits with y_sem < 0 keep their argmax in every mode,
    so the unlabelled bucket is never used.
    trueclass  true class on labelled hits
    cons       the argmax majority over the hit's TRUE particle (ties -> lowest class); no particle -> argmax
    fix50      floor(n_err / 2) of the event's labelled wrong hits corrected, chosen by a local RNG seeded by
               (FIX_SEED, batch, event) -- the global torch RNG is never touched
    fixtrack   every labelled wrong hit of a track class (pion / muon / kaon / hadron) corrected
    """
    import numpy as np
    import torch
    if readout is None:
        raise ValueError(f"class_channels({mode!r}) needs the cache's semantic read-out")
    fgc = d["fg_idx"].cpu()
    ys = d["y_sem"][fgc].long().numpy()
    yi = d["y_i"][fgc].long().numpy()
    am = predicted_logits(d["hit_x"], readout).argmax(dim=1).cpu().numpy()
    out = am.copy()
    err = (ys >= 0) & (am != ys)
    if mode == "trueclass":
        lab = ys >= 0
        out[lab] = ys[lab]
    elif mode == "cons":
        part = yi >= 0
        if part.any():
            _, inv = np.unique(yi[part], return_inverse=True)
            counts = np.zeros((inv.max() + 1, CLASS_BUCKETS), dtype=np.int64)
            np.add.at(counts, (inv, am[part]), 1)
            out[part] = counts.argmax(axis=1)[inv]
    elif mode == "fix50":
        idx = np.flatnonzero(err)
        rng = np.random.default_rng([FIX_SEED, int(d["batch"]), int(d["event"])])
        pick = rng.permutation(idx)[:idx.size // 2]
        out[pick] = ys[pick]
    elif mode == "fixtrack":
        sel = err & np.isin(ys, TRACK_CLASSES)
        out[sel] = ys[sel]
    else:
        raise ValueError(f"unknown label mode {mode!r}")
    return torch.as_tensor(out, dtype=torch.long)


def split_channels(d, mode, readout=None):
    """Stage 1a-R8: 16 channels = [hit block (8) | candidate block (8)].

    The candidate block is filled on SEED rows only (cand_net reads only those; hit_net never sees the block), so
    ORACLE-SEED gives truth to the candidate encoder alone. PRED-HARD is the one-hot of the trunk's argmax — a hardened
    function of h.x, no new information.
    """
    import torch
    n, k = d["fg_idx"].numel(), CLASS_BUCKETS
    dev, dt = d["hit_x"].device, d["hit_x"].dtype
    hit_spec, cand_spec = SPLIT_MODES[mode]

    def block(spec):
        if spec is None:
            return torch.zeros(n, k, device=dev, dtype=dt)
        if spec == "true":
            ys = d["y_sem"][d["fg_idx"].cpu()].long()
            ys = torch.where(ys < 0, torch.full_like(ys, k - 1), ys)
        else:
            if readout is None:
                raise ValueError("s_pred_hard needs the cache's semantic read-out")
            ys = predicted_logits(d["hit_x"], readout).argmax(dim=1).cpu()
        return torch.nn.functional.one_hot(ys, k).to(device=dev, dtype=dt)

    hb, cb = block(hit_spec), block(cand_spec)
    seed = torch.zeros(n, dtype=torch.bool, device=dev)
    seed[d["cp"].to(dev)] = True
    cb = cb * seed[:, None].to(dt)
    return torch.cat([hb, cb], dim=1)


def iter_events(cache, lo, hi, mat, rule_fn, a, dev, need_hit_x=True, store=False):
    mode = getattr(a, "class_channels", "none")
    readout = semantic_readout(cache) if mode in READOUT_MODES else None
    for ev in mb.iter_cache(cache, lo, hi):
        d = prepare(ev, mat, rule_fn, a.m, a.t_d, a.t_assign, dev, need_hit_x)
        if d is not None:
            if mode != "none":
                d["hit_x"] = torch_cat(d["hit_x"], class_channels(d, mode, readout))
            yield to_store(d) if store else d


def torch_cat(x, y):
    import torch
    return torch.cat([x, y], dim=1)


def apply_head_cfg(a, cfg):
    """Everything a tool must take from a trained head's config to rebuild its inputs exactly."""
    for k in ("rule", "m", "r", "t_d", "t_assign"):
        setattr(a, k, cfg[k])
    a.class_channels = cfg.get("class_channels", "none")


def state_sha256(model):
    """Hash of every parameter/buffer, in key order (initialisation identity check)."""
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cmd_train(a):
    import hashlib
    import torch
    mat, dev = _setup(a)
    rule_fn = s0.make_rule(a.rule)
    if a.class_channels in READOUT_MODES and a.dev_cache:
        if _manifest_sha(a.train_cache) is None or _manifest_sha(a.train_cache) != _manifest_sha(a.dev_cache):
            raise SystemExit("pred class channels need train and dev caches from the same checkpoint")
    torch.manual_seed(a.seed)
    train = list(iter_events(a.train_cache, *mb.parse_range(a.train_batches), mat, rule_fn, a,
                             dev, store=True))
    print(f"prepared {len(train)} training events", flush=True)
    if not train:
        raise SystemExit("no usable training events")
    model = build_model(train[0]["hit_x"].size(1), train[0]["ox"].size(1), a.width, a.r).to(dev)
    if a.class_channels in SPLIT_MODES:
        model.split_k = CLASS_BUCKETS
    init_sha = state_sha256(model)
    if a.class_channels in ORACLE_MODES:
        print("  *** ORACLE UPPER BOUND: the head sees TRUE semantic classes; not a deployable model ***",
              flush=True)
    gb = sum(v.numel() * v.element_size() for d in train for v in d.values()
             if hasattr(v, "numel")) / 1e9
    print(f"  cached training tensors: {gb:.2f} GB (CPU)", flush=True)
    model.dist_only = a.dist_only
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "epochs").mkdir(exist_ok=True)
    cfg = {"rule": a.rule, "m": a.m, "r": a.r, "width": a.width, "lr": a.lr, "epochs": a.epochs,
           "w_exist": a.w_exist, "w_merge": a.w_merge, "seed": a.seed, "t_d": a.t_d,
           "t_assign": a.t_assign, "hit_dim": train[0]["hit_x"].size(1),
           "lat_dim": train[0]["ox"].size(1), "accum": a.accum, "max_merge_k": MAX_MERGE_K,
           "train_cache": a.train_cache, "train_batches": a.train_batches,
           "n_train_events": len(train), "target": a.target, "exist_target": a.exist_target,
           "merge_scope": a.merge_scope, "dist_only": a.dist_only, "decode": a.decode,
           "decode_kwargs": decode_kwargs(a), "selection": "dev-A ari_particle, decode above",
           "class_channels": a.class_channels, "oracle_upper_bound": a.class_channels in ORACLE_MODES,
           "latent_checkpoint_sha256": _manifest_sha(a.train_cache),
           "init_sha256": init_sha, "perm_sha256": [],
           "split_channels": CLASS_BUCKETS if a.class_channels in SPLIT_MODES else 0,
           "comem": a.comem, "comem_pairs": a.comem_pairs}
    # R10: pairs come from a private generator, so the global RNG (init, permutations) is untouched in every arm
    comem_gen = torch.Generator().manual_seed(COMEM_SEED + a.seed)
    hist, best, best_epoch = [], -math.inf, None
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(len(train))
        cfg["perm_sha256"].append(hashlib.sha256(perm.numpy().tobytes()).hexdigest())
        tot, parts, n = 0.0, {"mem": 0.0, "exist": 0.0, "merge": 0.0, "comem": 0.0}, 0
        opt.zero_grad()
        for j, i in enumerate(perm.tolist()):
            loss, p = event_loss(model, to_device(train[i], dev), a.w_exist, a.w_merge,
                                 a.target, a.exist_target, a.merge_scope, comem=a.comem,
                                 comem_gen=comem_gen, comem_pairs=a.comem_pairs)
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
                                out=None, tag="dev"))
        hist.append(row)
        print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                               for k, v in row.items()), flush=True)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"state_dict": state, "cfg": {**cfg, "epoch": epoch}},
                   out / "epochs" / f"head_ep{epoch}.pt")
        score = row.get("dev/ari_particle", NAN)
        if a.dev_cache and math.isfinite(score) and score > best:
            best, best_epoch = score, epoch
            torch.save({"state_dict": state, "cfg": {**cfg, "epoch": epoch}}, out / "head_best.pt")
        mb.write_rows(out / "train_history.csv", hist)
    cfg.update({"best_epoch": best_epoch, "best_dev_ari": best if best_epoch is not None else NAN})
    # head.pt = LAST epoch (as in Run 1); head_best.pt = best dev-A epoch under the decode above
    torch.save({"state_dict": model.state_dict(), "cfg": {**cfg, "epoch": a.epochs - 1}},
               out / "head.pt")
    (out / "train_meta.json").write_text(json.dumps({**cfg, **_prov(mat)}, indent=2))
    print(f"wrote {out}/head.pt (last epoch) and head_best.pt (dev-A epoch {best_epoch}, "
          f"ari_particle {best:.4f})")


def evaluate(model, a, mat, rule_fn, dev, cache, batches, out, tag):
    import torch
    import numpy as np
    model.eval()
    rows = []
    with torch.no_grad():
        for d in iter_events(cache, *mb.parse_range(batches), mat, rule_fn, a, dev):
            lab_fg, diag = predict_labels(model, d, **decode_kwargs(a))
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
            mp = keep & (d["y_i"] >= 0)
            rows[-1]["pop_unassigned"] = float((lab[mp] < 0).float().mean()) if bool(mp.any()) else NAN
    g = lambda c: np.array([float(r.get(c, NAN)) for r in rows], dtype=float)
    pop, corr, orc = g("pop").sum(), g("pick_correct").sum(), g("oracle_in_nb").sum()
    summary = {f"{tag}/ari_particle": float(np.nanmean(g("ari_particle"))),
               f"{tag}/ari_hardseed": float(np.nanmean(g("ari_particle_hardseed"))),
               f"{tag}/n_pred_med": float(np.nanmedian(g("n_pred"))),
               f"{tag}/groups_med": float(np.nanmedian(g("n_groups"))),
               f"{tag}/null_frac": float(np.nanmean(g("frac_null"))),
               f"{tag}/pick_acc": float(corr / max(pop, 1)),
               f"{tag}/pick_acc_of_local": float(corr / max(orc, 1)),
               f"{tag}/pick_own_of_local": float(g("pick_own").sum() / max(g("own_in_nb").sum(), 1)),
               f"{tag}/pop_unassigned": float(np.nanmean(g("pop_unassigned"))),
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
    apply_head_cfg(a, cfg)
    lo, hi = mb.parse_range(a.batches)
    if (hi is None or hi > a.heldout_start) and not a.allow_heldout:
        raise SystemExit(f"--batches {a.batches} reaches the held-out set; use a development "
                         "split or pass --allow-heldout")
    model = load_head(ck, dev)
    evaluate(model, a, mat, s0.make_rule(cfg["rule"]), dev, a.cache, a.batches, a.out, a.tag)
    # provenance for confirmation reports (R11): which head, which cache, which decode
    meta = {"head": str(a.head), "head_sha256": mb.sha256(a.head), "cache": str(a.cache), "batches": a.batches,
            "cache_checkpoint_sha256": _manifest_sha(a.cache),
            "head_latent_checkpoint_sha256": cfg.get("latent_checkpoint_sha256"),
            "decode_kwargs": decode_kwargs(a), "head_cfg_comem": cfg.get("comem"), **_prov(mat)}
    (Path(a.out) / f"{a.tag}_eval_meta.json").write_text(json.dumps(meta, indent=2))


def load_head(ck, dev):
    cfg = ck["cfg"]
    model = build_model(cfg["hit_dim"], cfg["lat_dim"], cfg["width"], cfg["r"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.dist_only = bool(cfg.get("dist_only", False))
    model.overlap = cfg.get("overlap", "dense")
    model.split_k = int(cfg.get("split_channels", 0))
    model.eval()
    return model


def cmd_diagnose(a):
    """Decode ablations + head statistics for one trained head, on dev-A only."""
    import torch
    lo, hi = mb.parse_range(a.batches)
    if hi is None or hi > a.deva_end:
        raise SystemExit(f"diagnose is dev-A only (batches < {a.deva_end}); got {a.batches}")
    mat, dev = _setup(a)
    ck = torch.load(a.head, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    apply_head_cfg(a, cfg)
    model = load_head(ck, dev)
    stats, rows = {}, []
    t0 = time.perf_counter()
    for d in iter_events(a.cache, lo, hi, mat, s0.make_rule(cfg["rule"]), a, dev):
        for k, v in head_stats(model, d).items():
            stats[k] = stats.get(k, 0) + v
        rows += diagnose_event(model, d)
    if not rows:
        raise SystemExit("no usable events")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mb.write_rows(out / "diagnose_events.csv", rows)
    res = {"head": str(a.head), "head_epoch": cfg.get("epoch"), "batches": a.batches,
           "variants": summarize_variants(rows), "heads": summarize_stats(stats),
           "wall_s": time.perf_counter() - t0}
    (out / "diagnose_summary.json").write_text(json.dumps(res, indent=2))
    print(f"{'variant':16s} {'ARI':>7s} {'d_vs_hard':>10s} {'+-SE':>7s} {'pred/true':>9s} "
          f"{'groups':>6s} {'no_cl':>6s} {'dead':>6s} {'pop_-1':>7s}")
    for name, v in res["variants"].items():
        print(f"{name:16s} {v['ari_particle']:7.4f} {v['delta']:+10.4f} {v['delta_se']:7.4f} "
              f"{v['pred_true_med']:9.2f} {v['groups_med']:6.1f} {v['no_cluster']:6.3f} "
              f"{v['no_cluster_all_dead']:6.3f} {v['pop_unassigned']:7.3f}")
    print(json.dumps(res["heads"], indent=2))
    print(f"wrote {out}/diagnose_summary.json")


def cmd_readout_check(a):
    """Gate P0 (Stage 1a-R6): does the read-out recomputed from the fp16 cached embedding reproduce the
    cache's foreground definition (argmax != diffuse, computed in fp32 at cache time)?"""
    import torch
    readout = semantic_readout(a.cache)
    lo, hi = mb.parse_range(a.batches)
    n_fg = n_diffuse = n_pop = n_pop_ok = 0
    per_class = {}
    for ev in mb.iter_cache(a.cache, lo, hi):
        fg = ev["mask"].bool()
        if not bool(fg.any()):
            continue
        pred = predicted_logits(ev["hit_x_fg"], readout).argmax(dim=1)
        n_fg += int(fg.sum())
        n_diffuse += int((pred == DIFFUSE).sum())
        ys, yi = ev["y_sem"].long()[fg], ev["y_i"].long()[fg]
        pop = (ys >= 0) & (yi >= 0)
        n_pop += int(pop.sum())
        n_pop_ok += int((pred[pop] == ys[pop]).sum())
        for c in torch.unique(ys[pop]).tolist():
            sel = pop & (ys == c)
            acc = per_class.setdefault(int(c), [0, 0])
            acc[0] += int(sel.sum()); acc[1] += int((pred[sel] == c).sum())
    res = {"cache": a.cache, "batches": a.batches, "checkpoint_sha256": readout[2],
           "n_fg": n_fg, "fg_reclassified_diffuse": n_diffuse / max(n_fg, 1),
           "pop_accuracy": n_pop_ok / max(n_pop, 1),
           "pop_recall_by_class": {c: v[1] / max(v[0], 1) for c, v in sorted(per_class.items())},
           "pass": n_diffuse / max(n_fg, 1) <= 1e-3}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if not res["pass"]:
        raise SystemExit("P0 FAIL: the recomputed read-out does not reproduce the cached foreground mask")


def cmd_time(a):
    """Solo timing of a trained head's inference path on dev-A.

    Timing boundary as in mat_bench.py bench: the event's tensors are on the device before the
    clock starts; the clock stops after a device synchronize, before labels are copied back.
    Timed per event: (1) the decoder end to end (foreground, frozen candidates, top-M, head
    forward, decode); (2) the same with a synchronize after each stage (breakdown only; the
    extra syncs make it slightly slower); (3) the hard seed decoder on the same candidates;
    (4) the Thread-A seed backend; (5) canonical DBSCAN. Untimed: the labels of (1) are
    compared with the evaluated path (prepare + predict_labels) on every event.
    """
    import os
    import numpy as np
    import torch
    lo, hi = mb.parse_range(a.batches)
    if hi is None or hi > a.deva_end:
        raise SystemExit(f"time is dev-A only (batches < {a.deva_end}); got {a.batches}")
    mat, dev = _setup(a)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)
    sync = (lambda: torch.cuda.synchronize(dev)) if cuda else (lambda: None)
    ck = torch.load(a.head, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    apply_head_cfg(a, cfg)
    if a.class_channels != "none":
        raise SystemExit("time: class-probe heads are diagnostic (oracle / control) and not timed")
    model = load_head(ck, dev)
    rule_fn = s0.make_rule(cfg["rule"])
    kw = decode_kwargs(a)
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=a.dbscan_eps, min_samples=a.dbscan_min_samples)
    seed_kw = dict(t_beta=a.seed_t_beta, t_d=a.seed_t_d, t_assign=a.seed_t_assign)

    def timed(fn):
        base = 0
        if cuda:
            sync()
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
        t0 = time.perf_counter()
        res = fn()
        sync()
        dt = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_allocated(dev) - base) / 1e6 if cuda else NAN
        return res, dt, peak

    def decoder(evd, mark=None):
        x = infer_inputs(evd, mat, rule_fn, a.m, a.t_d, a.t_assign, dev, mark=mark)
        return None if x is None else infer_labels(model, x, mark=mark, **kw)

    def hardseed(evd):
        fg_idx = evd["mask"].bool().nonzero(as_tuple=False).squeeze(1)
        if fg_idx.numel() == 0:
            return None
        ox, beta = evd["ox"][fg_idx], evd["beta"][fg_idx]
        return s0.seeds_and_labels(mat, ox, beta, rule_fn(beta.double().cpu()), a.t_d,
                                   a.t_assign, dev)[1]

    def backend(evd, name, **bkw):
        return mat.materialize_labels(name, evd["ox"], evd["beta"], evd["mask"], None, 1, **bkw)[0]

    stages = ["candidates", "topm", "forward", "merge", "decode"]
    rows, n_warm, mismatches, n_seen = [], 0, 0, 0
    t_start = time.perf_counter()
    for ev in mb.iter_cache(a.cache, lo, hi):
        if a.limit and n_seen >= a.limit:
            break
        d = prepare(ev, mat, rule_fn, a.m, a.t_d, a.t_assign, dev)     # evaluated path (untimed)
        if d is None:
            continue
        with torch.no_grad():
            evd = {k: ev[k].to(dev) for k in ("ox", "beta", "mask", "hit_x_fg")}
            ref, _ = predict_labels(model, d, **kw)
            lab, t_dec, pk_dec = timed(lambda: decoder(evd))
            match = lab is not None and torch.equal(lab.cpu(), ref.cpu())
            marks = []

            def mark(name):
                sync()
                marks.append((name, time.perf_counter()))
            sync()
            t0 = time.perf_counter()
            decoder(evd, mark)
            _, t_hard, pk_hard = timed(lambda: hardseed(evd))
            _, t_seed, pk_seed = timed(lambda: backend(evd, "seed", **seed_kw))
            t_db = NAN
            if not a.no_dbscan:
                _, t_db, _ = timed(lambda: backend(evd, "dbscan", dbscan=dbscan))
        if n_warm < a.warmup:
            n_warm += 1
            continue
        n_seen += 1
        mismatches += int(not match)
        st, prev = {}, t0
        for name, t in marks:
            st[f"t_{name}_s"] = t - prev
            prev = t
        rows.append({"batch": ev["batch"], "event": ev["event"], "n_hit": int(ev["mask"].numel()),
                     "n_fg": int(d["fg_idx"].numel()), "K": d["K"], "labels_match": int(match),
                     "t_decoder_s": t_dec, "t_decoder_staged_s": prev - t0, **st,
                     "t_hardseed_s": t_hard, "t_seed_backend_s": t_seed, "t_dbscan_s": t_db,
                     "peak_decoder_mb": pk_dec, "peak_hardseed_mb": pk_hard,
                     "peak_seed_backend_mb": pk_seed})
    if not rows:
        raise SystemExit("no timed events")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mb.write_rows(out / "timing_events.csv", rows)
    col = lambda c: np.array([r.get(c, NAN) for r in rows], dtype=float)

    def dist(c):
        v = col(c)
        v = v[np.isfinite(v)]
        if not v.size:
            return {}
        return {"median_ms": 1e3 * float(np.median(v)), "mean_ms": 1e3 * float(v.mean()),
                "p90_ms": 1e3 * float(np.percentile(v, 90)),
                "p99_ms": 1e3 * float(np.percentile(v, 99)),
                "max_ms": 1e3 * float(v.max()), "total_s": float(v.sum())}

    methods = {"decoder": "t_decoder_s", "hard_seed": "t_hardseed_s",
               "seed_backend": "t_seed_backend_s", "dbscan": "t_dbscan_s"}
    summ = {m: dist(c) for m, c in methods.items()}
    staged_mean = col("t_decoder_staged_s").mean()
    breakdown = {s_: {**dist(f"t_{s_}_s"),
                      "share_of_staged_mean": float(col(f"t_{s_}_s").mean() / staged_mean)}
                 for s_ in stages}
    ratio = lambda num, den: (summ[num]["total_s"] / summ[den]["total_s"]
                              if summ.get(num) and summ.get(den) else NAN)
    res = {"head": str(a.head), "head_sha256": mb.sha256(a.head), "head_epoch": cfg.get("epoch"),
           "cache": str(a.cache), "batches": a.batches, "decode_kwargs": kw,
           "seed_backend_kwargs": seed_kw,
           "dbscan": None if a.no_dbscan else {"eps": a.dbscan_eps,
                                               "min_samples": a.dbscan_min_samples},
           "events_timed": len(rows), "warmup_events": n_warm,
           "labels_match_evaluated_path": len(rows) - mismatches, "labels_mismatch": mismatches,
           "methods": summ, "decoder_breakdown": breakdown,
           "total_ratio": {"decoder_over_seed_backend": ratio("decoder", "seed_backend"),
                           "decoder_over_hard_seed": ratio("decoder", "hard_seed"),
                           "dbscan_over_decoder": ratio("dbscan", "decoder")},
           "peak_mb": {k: {"max": float(np.nanmax(col(c))), "p95": float(np.nanpercentile(col(c), 95))}
                       for k, c in (("decoder", "peak_decoder_mb"), ("hard_seed", "peak_hardseed_mb"),
                                    ("seed_backend", "peak_seed_backend_mb"))} if cuda else None,
           "env": {"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if cuda else None,
                   "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                   "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "nproc": os.cpu_count(), "cgroup_cpu_max": mb.cgroup_cpu_max()},
           "wall_s": time.perf_counter() - t_start, **_prov(mat)}
    (out / "timing_summary.json").write_text(json.dumps(res, indent=2))
    print(f"{'method':14s} {'median ms':>10s} {'mean ms':>9s} {'p90 ms':>8s} {'p99 ms':>8s} {'total s':>8s}")
    for m_, v in summ.items():
        if v:
            print(f"{m_:14s} {v['median_ms']:10.2f} {v['mean_ms']:9.2f} {v['p90_ms']:8.2f} "
                  f"{v['p99_ms']:8.2f} {v['total_s']:8.2f}")
    print("decoder stages (with per-stage sync):")
    for s_, v in breakdown.items():
        print(f"  {s_:11s} median {v['median_ms']:8.2f} ms  mean {v['mean_ms']:8.2f} ms  "
              f"share {100 * v['share_of_staged_mean']:5.1f} %")
    print(json.dumps(res["total_ratio"]), f"events {len(rows)}, label mismatches {mismatches}")
    print(f"wrote {out}/timing_summary.json")
    if mismatches:
        raise SystemExit(f"FAIL: timed labels differ from the evaluated path on {mismatches} events")


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
        s.add_argument("--decode", choices=sorted(PRESETS), default="v2",
                       help="decode preset (v1 = Run 1); the flags below override it")
        s.add_argument("--merge-thr", type=float, default=None)
        s.add_argument("--exist-thr", type=float, default=None)
        s.add_argument("--no-null", action="store_true",
                       help="never predict the null option at inference")
        s.add_argument("--no-merge", action="store_true",
                       help="skip the learned merge step at inference (merge_thr None; R11)")

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
    s.add_argument("--target", choices=["own", "primary"], default="own")
    s.add_argument("--exist-target", choices=["real", "primary"], default="real")
    s.add_argument("--merge-scope", choices=["local", "all"], default="local")
    s.add_argument("--class-channels", choices=CLASS_MODES, default="none",
                   help="class probes: 'zeros' = matched control (same architecture, no information), "
                        "'oracle' = TRUE semantic class one-hot (ORACLE UPPER BOUND, not deployable, "
                        "R5), 'pred' = the trunk's own semantic probabilities recomputed from hit_x (R6)")
    s.add_argument("--dist-only", action="store_true",
                   help="ablation: membership from latent distance + existence only (no pair_net)")
    s.add_argument("--comem", type=float, default=0.0,
                   help="R10: weight of the co-membership term over same / different-particle population hit "
                        "pairs (always computed and logged; added to the loss only when > 0)")
    s.add_argument("--comem-pairs", type=int, default=COMEM_PAIRS, help="R10: positive and negative pairs per event")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("readout-check")
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_readout_check)

    s = sub.add_parser("eval"); common(s)
    s.add_argument("--head", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:150")
    s.add_argument("--out", required=True)
    s.add_argument("--tag", default="dev")
    s.add_argument("--heldout-start", type=int, default=314)
    s.add_argument("--allow-heldout", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("diagnose"); common(s)
    s.add_argument("--head", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:150")
    s.add_argument("--out", required=True)
    s.add_argument("--deva-end", type=int, default=150)
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("time"); common(s)
    s.add_argument("--head", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:150")
    s.add_argument("--out", required=True)
    s.add_argument("--deva-end", type=int, default=150)
    s.add_argument("--warmup", type=int, default=8, help="untimed warm-up events (as mat_bench)")
    s.add_argument("--limit", type=int, default=0, help="stop after N timed events (0 = all)")
    s.add_argument("--seed-t-beta", type=float, default=0.5)
    s.add_argument("--seed-t-d", type=float, default=0.3)
    s.add_argument("--seed-t-assign", type=float, default=float("inf"))
    s.add_argument("--dbscan-eps", type=float, default=0.3)
    s.add_argument("--dbscan-min-samples", type=int, default=15)
    s.add_argument("--no-dbscan", action="store_true")
    s.set_defaults(fn=cmd_time)

    s = sub.add_parser("selftest"); common(s)
    s.set_defaults(fn=cmd_selftest, device="cpu")

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
