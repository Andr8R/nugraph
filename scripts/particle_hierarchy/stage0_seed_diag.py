#!/usr/bin/env python
"""stage0_seed_diag.py -- Stage-0 diagnostics for adaptive seeding and soft membership.

Offline only: reads the per-event cache written by
``scripts/dune_materialization/mat_bench.py cache`` (latent coords ``ox``, beta, predicted
foreground, truth). No training, no model forward.

  spectrum   event-wise beta spectrum of foreground hits, and where the TRUE condensation
             points (highest-beta hit of each true particle) sit in it
  rules      candidate event-wise seed thresholds -> seed/NMS/assignment (the validated seed
             decoder, t_d / t_assign fixed) -> no-cluster rate, seed recall, duplicates,
             spurious seeds, multiplicity, particle ARI
  headroom   for a given rule: is the correct seed the nearest (top-1), or among the 2/3/5
             nearest? plus top-M "oracle" particle ARI -> headroom for soft membership

Held-out protection: batches >= --heldout-start (default 314, the published held-out set)
are refused unless --allow-heldout is given. Development splits: 0:150 (explore) and
150:314 (confirm).

Definitions (identical to the benchmark): population = predicted-foreground hits with
y_semantic >= 0 and a true particle (the ``ari_particle`` population). Correct seed of a
population hit = the surviving seed whose seed hit belongs to the same true particle
(highest beta if several); "unreachable" if the hit's particle has no surviving seed.
"""
import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "mat_bench", HERE.parent / "dune_materialization" / "mat_bench.py")
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)

NAN = float("nan")
TOPK = (1, 2, 3, 5, 10, 20, 50)   # the neighbourhood widths the soft head could use (M)
LOGIT_EPS = 1e-6


# --------------------------------------------------------------------------- #
def guard(a):
    lo, hi = mb.parse_range(a.batches)
    if (hi is None or hi > a.heldout_start) and not a.allow_heldout:
        raise SystemExit(f"--batches {a.batches} reaches the held-out set (>= {a.heldout_start}); "
                         "use a development split (e.g. 0:150 or 150:314) or pass --allow-heldout")
    return lo, hi


def logit(b):
    import torch
    b = b.clamp(LOGIT_EPS, 1 - LOGIT_EPS)
    return torch.log(b) - torch.log1p(-b)


def otsu(values, bins=64):
    """Otsu threshold of a 1-D tensor (returns a value in its range)."""
    import torch
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo
    hist = torch.histc(values.double(), bins=bins, min=lo, max=hi)
    centers = torch.linspace(lo, hi, bins + 1, dtype=torch.double)
    centers = 0.5 * (centers[1:] + centers[:-1])
    w0 = torch.cumsum(hist, 0)
    w1 = w0[-1] - w0
    m0 = torch.cumsum(hist * centers, 0) / w0.clamp(min=1e-12)
    m1 = (torch.sum(hist * centers) - torch.cumsum(hist * centers, 0)) / w1.clamp(min=1e-12)
    var = w0 * w1 * (m0 - m1) ** 2
    k = int(torch.argmax(var[:-1])) if bins > 1 else 0
    return float(centers[k] + 0.5 * (centers[1] - centers[0]))


def event_truth(ev):
    """Population mask, true labels, foreground mask (all event-local torch tensors)."""
    y_i = ev["y_i"].long()
    ysem = ev["y_sem"].long()
    fg = ev["mask"].bool()
    pop = fg & (ysem >= 0) & (y_i >= 0)
    return fg, pop, y_i, ysem


# --------------------------------------------------------------------------- #
# threshold rules: each maps the foreground beta vector to one event threshold tau_e.
# Seeds are hits with beta > tau_e (strict, as in materialize.py). "Guaranteed" rules
# cap tau_e just below the event maximum so at least one candidate exists.
# --------------------------------------------------------------------------- #
def make_rule(spec):
    parts = spec.split(":")
    kind, args = parts[0], [float(x) for x in parts[1:]]
    import torch

    def below_max(tau, bf):
        return min(tau, float(bf.max()) - 1e-6)

    if kind == "fixed":                     # fixed:T      (reference, no guarantee)
        (t,) = args
        return lambda bf: t
    if kind == "fixed1":                    # fixed1:T     fixed + guaranteed top-1
        (t,) = args
        return lambda bf: below_max(t, bf)
    if kind == "rel":                       # rel:K:FLOOR:CAP   tau = clip(K*max, FLOOR, CAP)
        k, floor, cap = args
        return lambda bf: below_max(min(cap, max(floor, k * float(bf.max()))), bf)
    if kind == "quant":                     # quant:Q:FLOOR:CAP  tau = clip(quantile_Q, FLOOR, CAP)
        q, floor, cap = args
        return lambda bf: below_max(
            min(cap, max(floor, float(torch.quantile(bf.double(), q)))), bf)
    if kind == "topn":                      # topn:F:NMIN  keep the top max(NMIN, F*n_fg) hits
        f, nmin = args
        def rule(bf):
            n = max(int(nmin), int(math.ceil(f * bf.numel())))
            n = min(n, bf.numel())
            v = torch.topk(bf, n).values
            return float(v[-1]) - 1e-9 if n < bf.numel() else float(bf.min()) - 1e-9
        return rule
    if kind == "gap":                       # gap:N:CAP    largest logit gap among top-N
        n, cap = args
        def rule(bf):
            v = torch.sort(logit(bf.double()), descending=True).values[: int(n)]
            if v.numel() < 2:
                return below_max(cap, bf)
            gaps = v[:-1] - v[1:]
            j = int(torch.argmax(gaps))
            tau = float(torch.sigmoid((v[j] + v[j + 1]) / 2))
            return below_max(min(cap, tau), bf)
        return rule
    if kind == "otsu":                      # otsu:CAP     Otsu on logit(beta)
        (cap,) = args
        return lambda bf: below_max(min(cap, float(torch.sigmoid(
            torch.tensor(otsu(logit(bf.double())))))), bf)
    raise SystemExit(f"unknown rule {spec}")


DEFAULT_RULES = ["fixed:0.5", "fixed1:0.5", "rel:0.5:0.05:0.5", "rel:0.7:0.05:0.5",
                 "rel:0.9:0.05:0.5", "quant:0.995:0.05:0.5", "topn:0.005:1",
                 "gap:50:0.5", "otsu:0.5"]


# --------------------------------------------------------------------------- #
def seeds_and_labels(mat, ox, bf, tau, t_d, t_assign, dev):
    """Surviving seed indices (into fg) and hard labels, replicating materialize._seed_one
    exactly (same chunking and cdist kernels); verified against materialize_labels."""
    import torch
    n = ox.size(0)
    seed_idx = (bf > tau).nonzero(as_tuple=False).squeeze(1)
    if seed_idx.numel() == 0:
        return seed_idx, torch.full((n,), -1, dtype=torch.long, device=dev), None
    if seed_idx.numel() > mat.DEFAULT_MAX_SEEDS:
        seed_idx = seed_idx[torch.topk(bf[seed_idx], mat.DEFAULT_MAX_SEEDS).indices]
    S, bS = ox[seed_idx], bf[seed_idx]
    ks = S.size(0)
    idx = torch.arange(ks, device=dev)
    chunk = mat._chunk_for(ks, mat.DEFAULT_MAX_PAIRS)
    dominated = torch.zeros(ks, dtype=torch.bool, device=dev)
    for i in range(0, ks, chunk):
        q = slice(i, min(i + chunk, ks))
        qi = idx[q]
        within = torch.cdist(S[q], S) <= t_d
        better = (bS[None, :] > bS[q][:, None]) | (
            (bS[None, :] == bS[q][:, None]) & (idx[None, :] < qi[:, None]))
        dominated[q] = (within & better & (idx[None, :] != qi[:, None])).any(dim=1)
    cp = seed_idx[~dominated]
    C = ox[cp]
    k = C.size(0)
    mode = "donot_use_mm_for_euclid_dist" if k <= 25 else "use_mm_for_euclid_dist"
    step = max(1, min(n, mat.DEFAULT_MAX_PAIRS // max(1, k)))
    dists = []
    for i in range(0, n, step):
        dists.append(torch.cdist(ox[i:i + step], C, compute_mode=mode))
    D = torch.cat(dists)                                     # (n_fg, K)
    nn_d, nn_j = D.min(dim=1)
    ta = t_d if t_assign is None else t_assign
    labels = torch.where(nn_d <= ta, nn_j, torch.full_like(nn_j, -1))
    return cp, labels, D


def full_labels(n_hit, fg_idx, lab_fg, dev):
    import torch
    lab = torch.full((n_hit,), -1, dtype=torch.long, device=dev)
    lab[fg_idx] = lab_fg
    return lab


# --------------------------------------------------------------------------- #
def cmd_spectrum(a):
    import torch
    lo, hi = guard(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev_rows, p_rows = [], []
    for ev in mb.iter_cache(a.cache, lo, hi):
        fg, pop, y_i, ysem = event_truth(ev)
        beta = ev["beta"].double()
        bf = beta[fg]
        n_true = int(torch.unique(y_i[pop]).numel()) if pop.any() else 0
        row = {"batch": ev["batch"], "event": ev["event"], "n_hits": int(fg.numel()),
               "n_fg": int(fg.sum()), "n_pop": int(pop.sum()), "n_true": n_true}
        if bf.numel():
            srt = torch.sort(bf, descending=True).values
            row.update(beta_max=float(srt[0]), beta_p50=float(bf.median()),
                       beta_p90=float(torch.quantile(bf, 0.9)),
                       beta_p99=float(torch.quantile(bf, 0.99)))
            for r in (2, 3, 5, 10, 20):
                row[f"beta_top{r}"] = float(srt[r - 1]) if srt.numel() >= r else NAN
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
                row[f"n_above_{t:g}"] = int((bf > t).sum())
            lg = logit(srt[:50])
            if lg.numel() >= 2:
                gaps = lg[:-1] - lg[1:]
                j = int(torch.argmax(gaps))
                row.update(max_logit_gap_top50=float(gaps[j]), gap_rank_top50=j + 1)
            row["otsu_beta"] = float(torch.sigmoid(torch.tensor(otsu(logit(bf)))))
            # rank of every true particle's condensation point among foreground hits
            rank_of = torch.empty_like(bf, dtype=torch.long)
            order = torch.argsort(bf, descending=True)
            rank_of[order] = torch.arange(1, bf.numel() + 1)
            fg_idx = fg.nonzero(as_tuple=False).squeeze(1)
            pos = torch.full((fg.numel(),), -1, dtype=torch.long)
            pos[fg_idx] = torch.arange(fg_idx.numel())
            ranks = []
            for t in torch.unique(y_i[pop]).tolist():
                hits = (pop & (y_i == t)).nonzero(as_tuple=False).squeeze(1)
                cp = hits[torch.argmax(beta[hits])]
                r = int(rank_of[pos[cp]])
                ranks.append(r)
                sem = int(torch.mode(ysem[hits]).values)
                p_rows.append({"batch": ev["batch"], "event": ev["event"], "particle": t,
                               "n_pop_hits": int(hits.numel()), "semantic": sem,
                               "beta_cp": float(beta[cp]), "beta_cp_over_max": float(beta[cp] / srt[0]),
                               "cp_rank": r, "event_beta_max": float(srt[0]), "event_n_true": n_true})
            if ranks:
                rk = torch.tensor(sorted(ranks))
                row["frac_cp_in_top_ntrue"] = float((rk <= n_true).float().mean())
                row["frac_cp_in_top_2ntrue"] = float((rk <= 2 * n_true).float().mean())
                row["cp_rank_median"] = float(rk.float().median())
        ev_rows.append(row)
    mb.write_rows(out / "spectrum_events.csv", ev_rows)
    mb.write_rows(out / "spectrum_particles.csv", p_rows)
    summarize_spectrum(ev_rows, p_rows, out, a.batches)


def summarize_spectrum(ev_rows, p_rows, out, batches):
    import numpy as np
    f = lambda rows, c: np.array([float(r.get(c, NAN)) for r in rows], dtype=float)
    q = lambda x, p: float(np.nanpercentile(x, p)) if np.isfinite(x).any() else NAN
    L = [f"# Stage 0 — beta spectrum (batches {batches})", ""]
    ev = [r for r in ev_rows if r["n_fg"] > 0]
    low = [r for r in ev if r.get("beta_max", 0) <= 0.5]
    L.append(f"events: {len(ev_rows)} (with foreground {len(ev)}); beta_max <= 0.5: "
             f"{len(low)} ({len(low) / max(len(ev), 1):.1%})")
    for name, rows in (("all", ev), ("beta_max<=0.5", low)):
        if not rows:
            continue
        L.append(f"\n## events: {name} (n={len(rows)})")
        for c in ("n_fg", "n_true", "beta_max", "beta_top2", "beta_top5", "beta_p99",
                  "n_above_0.1", "n_above_0.3", "n_above_0.5", "max_logit_gap_top50",
                  "gap_rank_top50", "otsu_beta", "frac_cp_in_top_ntrue",
                  "frac_cp_in_top_2ntrue", "cp_rank_median"):
            x = f(rows, c)
            L.append(f"  {c:24s} p10 {q(x,10):9.4g}  p50 {q(x,50):9.4g}  p90 {q(x,90):9.4g}")
    keys = {(r["batch"], r["event"]) for r in low}
    for name, rows in (("all", p_rows),
                       ("in beta_max<=0.5 events", [r for r in p_rows if (r["batch"], r["event"]) in keys])):
        if not rows:
            continue
        L.append(f"\n## true particles: {name} (n={len(rows)})")
        for c in ("n_pop_hits", "beta_cp", "beta_cp_over_max", "cp_rank"):
            x = f(rows, c)
            L.append(f"  {c:24s} p10 {q(x,10):9.4g}  p50 {q(x,50):9.4g}  p90 {q(x,90):9.4g}")
        bc = f(rows, "beta_cp")
        L.append("  frac beta_cp > t: " + "  ".join(
            f"{t:g}:{np.mean(bc > t):.3f}" for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)))
    text = "\n".join(L)
    (out / "spectrum_summary.txt").write_text(text + "\n")
    print(text)


# --------------------------------------------------------------------------- #
def _event_eval(mat, ev, rule_fn, t_d, t_assign, dev, check_against_materialize):
    import torch
    fg, pop, y_i, ysem = event_truth(ev)
    fg_idx = fg.nonzero(as_tuple=False).squeeze(1)
    if fg_idx.numel() == 0:
        return None
    ox = ev["ox"].to(dev)[fg_idx.to(dev)]
    bf = ev["beta"].to(dev)[fg_idx.to(dev)]
    tau = rule_fn(bf.double().cpu())
    cp, lab_fg, D = seeds_and_labels(mat, ox, bf, tau, t_d, t_assign, dev)
    if check_against_materialize:
        ref, _ = mat.materialize_labels("seed", ev["ox"].to(dev), ev["beta"].to(dev),
                                        fg.to(dev), None, 1, t_beta=tau, t_d=t_d,
                                        t_assign=t_assign)
        mine = full_labels(fg.numel(), fg_idx.to(dev), lab_fg, dev)
        if not torch.equal(ref, mine):
            raise SystemExit(f"replica != materialize_labels on batch {ev['batch']} event {ev['event']}")
    return fg, fg_idx, pop, y_i, ysem, tau, cp, lab_fg.cpu(), (None if D is None else D.cpu())


def _seed_truth(fg_idx, cp, y_i, beta):
    """map each surviving seed to its true particle; pick the primary seed per particle."""
    seed_hits = fg_idx[cp.cpu()]
    seed_true = y_i[seed_hits]
    primary = {}
    for s, t in enumerate(seed_true.tolist()):
        if t < 0:
            continue
        if t not in primary or beta[seed_hits[s]] > beta[seed_hits[primary[t]]]:
            primary[t] = s
    return seed_true, primary


def cmd_rules(a):
    import torch
    lo, hi = guard(a)
    mat, dev = _setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.rules:
        fn = make_rule(spec)
        t0 = time.perf_counter()
        for n, ev in enumerate(mb.iter_cache(a.cache, lo, hi)):
            r = _event_eval(mat, ev, fn, a.t_d, a.t_assign, dev, a.check and n < a.check_events)
            base = {"rule": spec, "batch": ev["batch"], "event": ev["event"]}
            if r is None:
                rows.append({**base, "n_fg": 0})
                continue
            fg, fg_idx, pop, y_i, ysem, tau, cp, lab_fg, _ = r
            beta = ev["beta"]
            seed_true, primary = _seed_truth(fg_idx, cp, y_i, beta)
            true_ids = set(torch.unique(y_i[pop]).tolist())
            lab = full_labels(fg.numel(), fg_idx, lab_fg, "cpu")
            q = mb.event_quality(lab, y_i, ysem, fg & (ysem >= 0))
            n_seed_true = torch.bincount(seed_true[seed_true >= 0]) if (seed_true >= 0).any() else torch.zeros(0)
            rows.append({**base, "n_fg": int(fg_idx.numel()), "beta_max": float(beta[fg].max()),
                         "tau": tau, "n_candidates": int((beta[fg] > tau).sum()),
                         "n_seeds": int(cp.numel()), "no_cluster": int(cp.numel() == 0),
                         "n_true": len(true_ids),
                         "seed_recall": (len(set(primary) & true_ids) / len(true_ids)) if true_ids else NAN,
                         "n_duplicate_seeds": int((n_seed_true[n_seed_true > 1] - 1).sum()) if n_seed_true.numel() else 0,
                         "n_spurious_seeds": int((seed_true < 0).sum()),
                         **{k: q.get(k, NAN) for k in ("ari_particle", "ari_inclusive", "purity",
                                                       "completeness", "n_pred", "noise_frac")}})
        print(f"  {spec}: {time.perf_counter() - t0:.1f}s", flush=True)
    mb.write_rows(out / "rules_events.csv", rows)
    summarize_rules(rows, out, a)


def summarize_rules(rows, out, a):
    import numpy as np
    L = [f"# Stage 0 — seed threshold rules (batches {a.batches}; t_d {a.t_d}, t_assign {a.t_assign})", "",
         "| rule | subset | events | no-cluster | mean tau | seeds med | seed recall | dup/event | spurious/event | particle ARI | n_pred/n_true med | noise frac med |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    g = lambda rs, c: np.array([float(r.get(c, NAN)) for r in rs], dtype=float)
    for spec in a.rules:
        rs = [r for r in rows if r["rule"] == spec and int(r["n_fg"]) > 0]
        for sub, sel in (("all", rs), ("beta_max<=0.5", [r for r in rs if float(r["beta_max"]) <= 0.5])):
            if not sel:
                continue
            ratio = g(sel, "n_pred") / np.where(g(sel, "n_true") > 0, g(sel, "n_true"), np.nan)
            L.append(f"| {spec} | {sub} | {len(sel)} | {np.mean(g(sel,'no_cluster')):.3f} | "
                     f"{np.nanmean(g(sel,'tau')):.3f} | {np.nanmedian(g(sel,'n_seeds')):.0f} | "
                     f"{np.nanmean(g(sel,'seed_recall')):.3f} | {np.nanmean(g(sel,'n_duplicate_seeds')):.2f} | "
                     f"{np.nanmean(g(sel,'n_spurious_seeds')):.2f} | {np.nanmean(g(sel,'ari_particle')):.4f} | "
                     f"{np.nanmedian(ratio):.2f} | {np.nanmedian(g(sel,'noise_frac')):.3f} |")
    text = "\n".join(L)
    (out / "rules_summary.md").write_text(text + "\n")
    print(text)


# --------------------------------------------------------------------------- #
def cmd_headroom(a):
    import torch
    import numpy as np
    lo, hi = guard(a)
    mat, dev = _setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fn = make_rule(a.rule)
    rows = []
    SIZE_BINS = [1, 5, 20, 100, 500, 10 ** 9]
    by_sem = {}
    by_size = {}
    margins = []
    for n, ev in enumerate(mb.iter_cache(a.cache, lo, hi)):
        r = _event_eval(mat, ev, fn, a.t_d, a.t_assign, dev, a.check and n < a.check_events)
        if r is None:
            continue
        fg, fg_idx, pop, y_i, ysem, tau, cp, lab_fg, D = r
        row = {"batch": ev["batch"], "event": ev["event"], "n_fg": int(fg_idx.numel()),
               "n_seeds": int(cp.numel()), "n_pop": int(pop.sum()),
               "beta_max": float(ev["beta"][fg].max())}
        if D is None or not pop.any():
            rows.append(row)
            continue
        beta = ev["beta"]
        seed_true, primary = _seed_truth(fg_idx, cp, y_i, beta)
        pop_fg = pop[fg_idx]                                  # population among fg hits
        yt = y_i[fg_idx][pop_fg]                              # true particle per pop hit
        Dp = D[pop_fg]                                        # (n_pop, K)
        K = Dp.size(1)
        kmax = min(max(TOPK), K)
        order = torch.topk(Dp, kmax, dim=1, largest=False).indices
        correct = torch.tensor([primary.get(t, -1) for t in yt.tolist()], dtype=torch.long)
        reach = correct >= 0
        hit_at = {}
        for m in TOPK:
            mm = min(m, K)
            hit_at[m] = (order[:, :mm] == correct[:, None]).any(dim=1) & reach
        row.update(n_unreachable=int((~reach).sum()),
                   **{f"n_top{m}": int(hit_at[m].sum()) for m in TOPK})
        # "own particle": ANY surviving seed of the hit's true particle (primary or duplicate)
        st_order = seed_true[order]                            # (n_pop, kmax) truth of nearest seeds
        for m in TOPK:
            row[f"n_own_top{m}"] = int((st_order[:, :min(m, K)] == yt[:, None]).any(dim=1).sum())
        # what is the nearest seed of a hit whose nearest seed is NOT its primary seed?
        miss = ~hit_at[1]
        near_t = st_order[:, 0]
        row.update(n_near_dup=int((miss & reach & (near_t == yt)).sum()),
                   n_near_other=int((miss & (near_t >= 0) & (near_t != yt)).sum()),
                   n_near_spurious=int((miss & (near_t < 0)).sum()))
        if K >= 2:
            d2 = torch.topk(Dp, 2, dim=1, largest=False).values
            margins.append(((d2[:, 1] - d2[:, 0]) / d2[:, 1].clamp(min=1e-9)).numpy())
        # oracle labels: correct seed if it is within the top-M nearest, else nearest
        nearest = order[:, 0]
        oracle = {}
        for m in list(TOPK) + ["all"]:
            ok = reach if m == "all" else hit_at[m]
            lab_pop = torch.where(ok, correct, nearest)
            lab_fg_m = lab_fg.clone()
            lab_fg_m[pop_fg] = lab_pop
            lab = full_labels(fg.numel(), fg_idx, lab_fg_m, "cpu")
            q = mb.event_quality(lab, y_i, ysem, fg & (ysem >= 0))
            row[f"ari_particle_top{m}"] = q.get("ari_particle", NAN)
        lab = full_labels(fg.numel(), fg_idx, lab_fg, "cpu")
        row["ari_particle_decoder"] = mb.event_quality(lab, y_i, ysem, fg & (ysem >= 0)).get("ari_particle", NAN)
        # breakdowns (hit-weighted)
        sem = ysem[fg_idx][pop_fg]
        sizes = torch.bincount(yt)[yt]
        for key, grp in (("sem", sem), ("size", torch.bucketize(sizes, torch.tensor(SIZE_BINS)))):
            tgt = by_sem if key == "sem" else by_size
            for gv in torch.unique(grp).tolist():
                sel = grp == gv
                acc = tgt.setdefault(gv, {"n": 0, "unreach": 0, **{m: 0 for m in TOPK}})
                acc["n"] += int(sel.sum())
                acc["unreach"] += int((~reach & sel).sum())
                for m in TOPK:
                    acc[m] += int((hit_at[m] & sel).sum())
        rows.append(row)
    mb.write_rows(out / f"headroom_events_{a.rule.replace(':', '_')}.csv", rows)
    marg = np.concatenate(margins) if margins else np.array([])
    summarize_headroom(rows, by_sem, by_size, SIZE_BINS, marg, out, a)


def summarize_headroom(rows, by_sem, by_size, size_bins, marg, out, a):
    import numpy as np
    g = lambda rs, c: np.array([float(r.get(c, NAN)) for r in rs], dtype=float)
    rs = [r for r in rows if "n_top1" in r]
    tot = {c: np.nansum(g(rs, c)) for c in ["n_pop", "n_unreachable", "n_near_dup", "n_near_other",
                                               "n_near_spurious"]
           + [f"n_top{m}" for m in TOPK] + [f"n_own_top{m}" for m in TOPK]}
    npop = max(tot["n_pop"], 1)
    L = [f"# Stage 0 — membership headroom (rule {a.rule}; batches {a.batches}; "
         f"t_d {a.t_d}, t_assign {a.t_assign})", "",
         f"events scored: {len(rs)}; population hits: {int(npop)}", "",
         "**Hit-weighted correct-seed rates**",
         f"- unreachable (true particle has no surviving seed): {tot['n_unreachable'] / npop:.3f}"]
    for m in TOPK:
        L.append(f"- correct seed within top-{m}: {tot[f'n_top{m}'] / npop:.3f}"
                 f"  (of reachable: {tot[f'n_top{m}'] / max(npop - tot['n_unreachable'], 1):.3f})")
    L += ["", "**Own-particle view (any surviving seed of the hit's particle, primary or duplicate)**"]
    for m in TOPK:
        L.append(f"- an own-particle seed within top-{m}: {tot[f'n_own_top{m}'] / npop:.3f}")
    L += ["", "**Nearest seed of hits whose nearest seed is not their primary seed (fractions of all population hits)**",
          f"- duplicate seed of the same particle: {tot['n_near_dup'] / npop:.3f}",
          f"- seed of another particle: {tot['n_near_other'] / npop:.3f}",
          f"- spurious (background-hit) seed: {tot['n_near_spurious'] / npop:.3f}",
          "(primary-seed rates above count 'duplicate' as a miss; ARI does not, until duplicates are separated.)", ""]
    if marg.size:
        L.append("- relative top-1/top-2 distance margin (d2-d1)/d2: " + "  ".join(
            f"p{p}:{np.percentile(marg, p):.3f}" for p in (10, 25, 50, 75, 90)))
    L += ["", "**Event-mean particle ARI**",
          f"- hard decoder (as deployed): {np.nanmean(g(rs, 'ari_particle_decoder')):.4f}"]
    for m in list(TOPK) + ["all"]:
        L.append(f"- oracle within top-{m}: {np.nanmean(g(rs, f'ari_particle_top{m}')):.4f}")
    low = [r for r in rs if float(r["beta_max"]) <= 0.5]
    if low:
        L.append(f"- (beta_max<=0.5 events, n={len(low)}: decoder {np.nanmean(g(low,'ari_particle_decoder')):.4f}, "
                 f"top-3 oracle {np.nanmean(g(low,'ari_particle_top3')):.4f})")
    L += ["", "**By true semantic class (hit-weighted)**", "",
          "| class | hits | unreachable | " + " | ".join(f"top-{m}" for m in TOPK) + " |",
          "|---|---|" + "---|" * (len(TOPK) + 1)]
    names = ["pion", "muon", "kaon", "hadron", "shower", "michel", "diffuse"]
    for k in sorted(by_sem):
        s = by_sem[k]
        L.append(f"| {names[k] if 0 <= k < len(names) else k} | {s['n']} | {s['unreach']/s['n']:.3f} | "
                 + " | ".join(f"{s[m]/s['n']:.3f}" for m in TOPK) + " |")
    L += ["", "**By true particle size (population hits per particle)**", "",
          "| size bin | hits | unreachable | " + " | ".join(f"top-{m}" for m in TOPK) + " |",
          "|---|---|" + "---|" * (len(TOPK) + 1)]
    for k in sorted(by_size):
        s = by_size[k]
        lo_ = size_bins[k - 1] if k >= 1 else 0
        hi_ = size_bins[k] if k < len(size_bins) else "inf"
        L.append(f"| ({lo_}, {hi_}] | {s['n']} | {s['unreach']/s['n']:.3f} | "
                 + " | ".join(f"{s[m]/s['n']:.3f}" for m in TOPK) + " |")
    L += ["", "Reading: top-M >> top-1 with low 'unreachable' => soft membership / refinement has "
          "headroom; low top-M or high 'unreachable' => the seed set or the latent is the bottleneck. "
          "'Seed of another particle' is what soft membership can fix; 'duplicate seed' is a "
          "duplicate-suppression problem. Oracle ARI is not monotonic in M: moving part of a "
          "particle from a duplicate seed to its primary seed splits it."]
    text = "\n".join(L)
    (out / f"headroom_summary_{a.rule.replace(':', '_')}.md").write_text(text + "\n")
    print(text)


# --------------------------------------------------------------------------- #
def _setup(a):
    import torch
    import nugraph.models.nugraph3.materialize as mat
    dev = torch.device(f"cuda:{a.device}" if a.device != "cpu" and torch.cuda.is_available() else "cpu")
    meta = {"materialize_py": mat.__file__, "materialize_sha256": mb.sha256(mat.__file__),
            "nugraph_pkg": sys.modules["nugraph"].__file__, "device": str(dev),
            "cache_manifest": json.loads((Path(a.cache) / "manifest.json").read_text()),
            "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"meta_{a.cmd}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("materialize_sha256", "nugraph_pkg", "device")}))
    return mat, dev


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--cache", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--batches", default="0:150")
        s.add_argument("--heldout-start", type=int, default=314)
        s.add_argument("--allow-heldout", action="store_true")

    def seeding(s):
        s.add_argument("--device", default="0")
        s.add_argument("--t-d", type=float, default=0.3)
        s.add_argument("--t-assign", type=float, default=float("inf"))
        s.add_argument("--check", action="store_true",
                       help="verify the replica against materialize_labels on the first events")
        s.add_argument("--check-events", type=int, default=200)

    s = sub.add_parser("spectrum"); common(s); s.set_defaults(fn=cmd_spectrum)
    s = sub.add_parser("rules"); common(s); seeding(s)
    s.add_argument("--rules", nargs="+", default=DEFAULT_RULES)
    s.set_defaults(fn=cmd_rules)
    s = sub.add_parser("headroom"); common(s); seeding(s)
    s.add_argument("--rule", default="fixed:0.5")
    s.set_defaults(fn=cmd_headroom)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
