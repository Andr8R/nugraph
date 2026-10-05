#!/usr/bin/env python
"""coverage_anatomy.py -- why population hits cannot reach their own particle (T1 diagnostic).

Pre-registration: experiments/particle_hierarchy/coverage_anatomy.md. Offline only: reads dev-A caches written
by ``mat_bench.py cache``; no training, no model forward.

  run     one cache -> per-event rows (every candidate set), per-particle rows for FROZEN-seedless
          particles, per-class / per-size hit counts, gate records
  report  several run outputs -> gates A1-A4, the cause split, counterfactuals, readings

Candidate sets (all on the same events): FROZEN (quant:0.995:0.05:0.5, deployed NMS t_d 0.3), GREEDY (same seed
list, greedy NMS), TD015 (deployed NMS t_d 0.15), NONMS (every above-threshold seed), LOWTAU (fixed1:0.3, deployed
NMS), ORACLE-ADD (FROZEN + one oracle seed per FROZEN-seedless particle; upper bound), ORACLE-SEEDS (one oracle seed
per true particle; upper bound). Oracle seed = the particle's highest-beta population hit.

The pure decision logic (deployed / greedy NMS from a 'within t_d' matrix, suppressor attribution, categories,
readings) is numpy-only and unit-tested without torch. The geometry (the 'within' matrix, k-NN) is computed in torch
with exactly the chunking and cdist calls of the deployed replica (stage0_seed_diag.seeds_and_labels), so the
attribution reproduces the deployed survivor set by construction (gate A2 checks it on every event).
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAN = float("nan")
FROZEN_RULE = "quant:0.995:0.05:0.5"
LOWTAU_RULE = "fixed1:0.3"
SETS = ["FROZEN", "GREEDY", "TD015", "NONMS", "LOWTAU", "ORACLE-ADD", "ORACLE-SEEDS"]
ORACLE_SETS = {"ORACLE-ADD", "ORACLE-SEEDS"}
M_REACH = 20
M_OWN5 = 5
SIZE_BINS = [1, 5, 20, 100, 500, 10 ** 9]
SEM_NAMES = ["pion", "muon", "kaon", "hadron", "shower", "michel", "diffuse"]
# hit categories under FROZEN (mutually exclusive, sum = population hits)
CAUSES = ["sub", "cap", "supp_foreign_direct", "supp_foreign_chain", "supp_spurious_direct",
          "supp_spurious_chain", "supp_self", "unattributed"]
CATS = ["reach", "oor"] + [f"seedless_{c}" for c in CAUSES]
ANCHOR = {("FROZEN", "seedless"): 0.595, ("FROZEN", "own5"): 0.286, ("FROZEN", "ari_dec"): 0.167,
          ("FROZEN", "ari_all"): 0.620, ("LOWTAU", "seedless"): 0.336, ("LOWTAU", "ari_all"): 0.781}
ANCHOR_TOL = 0.003
A2_MAX_EVENTS = 2


# =========================================================================== #
# numpy core (no torch)
# =========================================================================== #
def better_matrix(bS, rows=None):
    """better[i, j] = seed j outranks seed i (higher beta, ties -> lower position), as in materialize."""
    bS = np.asarray(bS)
    k = bS.size
    rows = np.arange(k) if rows is None else np.asarray(rows)
    pos = np.arange(k)
    bi = bS[rows][:, None]
    return (bS[None, :] > bi) | ((bS[None, :] == bi) & (pos[None, :] < rows[:, None]))


def deployed_from_within(within, bS, chunk_rows=1024):
    """Deployed NMS: a seed is suppressed if ANY outranking seed (survivor or not) is within t_d.

    Returns (dominated bool[k], dominator int[k]): the dominator is the highest-ranked outranking seed within
    t_d (-1 if none)."""
    within = np.asarray(within, dtype=bool)
    bS = np.asarray(bS)
    k = bS.size
    dominated = np.zeros(k, bool)
    dominator = np.full(k, -1, np.int64)
    if k == 0:
        return dominated, dominator
    pos = np.arange(k)
    order = np.lexsort((pos, -bS))                    # best first: higher beta, then lower position
    rank = np.empty(k, np.int64)
    rank[order] = np.arange(k)
    for i in range(0, k, chunk_rows):                 # row chunks keep memory O(chunk * k)
        q = pos[i:i + chunk_rows]
        m = within[q] & better_matrix(bS, q) & (pos[None, :] != q[:, None])
        dom = m.any(axis=1)
        dominated[q] = dom
        best = np.argmin(np.where(m, rank[None, :], k), axis=1)   # outranking seed with the best rank
        dominator[q] = np.where(dom, best, -1)
    return dominated, dominator


def greedy_from_within(within, bS):
    """Greedy NMS: visit seeds best-first; keep one unless a KEPT seed lies within t_d."""
    within = np.asarray(within, dtype=bool)
    bS = np.asarray(bS)
    k = bS.size
    keep = np.zeros(k, bool)
    for i in np.lexsort((np.arange(k), -bS)):
        if not (within[i] & keep).any():
            keep[i] = True
    return keep


def primary_of(seed_true, seed_beta):
    """{particle: position of its highest-beta candidate}; ties -> lower position; spurious (<0) skipped."""
    out = {}
    for s, (t, b) in enumerate(zip(np.asarray(seed_true).tolist(), np.asarray(seed_beta).tolist())):
        if t < 0:
            continue
        if t not in out or b > seed_beta[out[t]]:
            out[t] = s
    return out


def oracle_seeds(y, pop, beta):
    """{particle: fg position of its highest-beta population hit} (ties -> lower position)."""
    out = {}
    y, pop, beta = np.asarray(y), np.asarray(pop, bool), np.asarray(beta)
    for i in np.nonzero(pop & (y >= 0))[0]:
        t = int(y[i])
        if t not in out or beta[i] > beta[out[t]]:
            out[t] = int(i)
    return out


def set_stats(cand, y, pop, beta, topm, m_reach=M_REACH, m_own5=M_OWN5):
    """Coverage statistics of one candidate set for the population hits of one event.

    cand: fg positions of the candidates; topm: (n_pop, min(m_reach, K)) candidate positions (into cand) of each
    population hit's nearest candidates, nearest first. Returns (stats dict, per-hit arrays dict)."""
    cand = np.asarray(cand, np.int64)
    y, pop, beta = np.asarray(y), np.asarray(pop, bool), np.asarray(beta)
    yp = y[pop]
    n = yp.size
    K = cand.size
    st = {"K": K, "n_pop": n}
    if K == 0:
        st.update(n_seedless=n, n_oor=0, n_reach=0, n_own5=0, n_dup=0, n_spur=0)
        return st, {"seedless": np.ones(n, bool), "reach": np.zeros(n, bool),
                    "lab_dec": np.full(n, -1), "lab_or20": np.full(n, -1), "lab_all": np.full(n, -1),
                    "nearest_own": np.zeros(n, bool)}
    seed_true = y[cand]
    prim = primary_of(seed_true, beta[cand])
    correct = np.array([prim.get(int(t), -1) for t in yp], np.int64)
    has_own = correct >= 0
    topm = np.asarray(topm, np.int64)
    own_in = seed_true[topm] == yp[:, None]
    reach = own_in[:, :m_reach].any(axis=1)
    own5 = own_in[:, :m_own5].any(axis=1)
    nearest = topm[:, 0]
    prim20 = (topm[:, :m_reach] == correct[:, None]).any(axis=1) & has_own
    counts = np.bincount(seed_true[seed_true >= 0]) if (seed_true >= 0).any() else np.zeros(0, np.int64)
    st.update(n_seedless=int((~has_own).sum()), n_oor=int((has_own & ~reach).sum()),
              n_reach=int(reach.sum()), n_own5=int(own5.sum()),
              n_dup=int((counts[counts > 1] - 1).sum()) if counts.size else 0,
              n_spur=int((seed_true < 0).sum()))
    return st, {"seedless": ~has_own, "reach": reach,
                "lab_dec": nearest, "lab_or20": np.where(prim20, correct, nearest),
                "lab_all": np.where(has_own, correct, nearest), "nearest_own": seed_true[nearest] == yp}


def classify_seedless(y, pop, beta, tau, seed_list, dominated, dominator, survivor_mask, frozen_seeded):
    """Cause of every FROZEN-seedless particle.

    seed_list: fg positions of the above-threshold seeds after the cap (NMS order); dominated / dominator: from
    deployed_from_within (None if not attributable); survivor_mask: bool over seed_list (deployed survivors);
    frozen_seeded: set of particles with a FROZEN candidate. Returns {particle: record}."""
    y, pop, beta = np.asarray(y), np.asarray(pop, bool), np.asarray(beta)
    seed_list = np.asarray(seed_list, np.int64)
    out = {}
    parts = sorted(set(y[pop & (y >= 0)].tolist()) - set(frozen_seeded))
    for p in parts:
        own_fg = np.nonzero(y == p)[0]
        bmax = float(beta[own_fg].max())
        rec = {"beta_max": bmax, "tau": float(tau), "gap": float(tau - bmax)}
        if not (beta[own_fg] > tau).any():
            rec["cause"] = "sub"
        else:
            in_list = np.nonzero(y[seed_list] == p)[0] if seed_list.size else np.zeros(0, np.int64)
            if in_list.size == 0:
                rec["cause"] = "cap"
            elif dominated is None:
                rec["cause"] = "unattributed"
            else:
                s = in_list[np.lexsort((in_list, -beta[seed_list[in_list]]))[0]]
                d = int(dominator[s])
                if not dominated[s] or d < 0:
                    rec["cause"] = "unattributed"           # deployed survivors disagree (gate A2)
                else:
                    owner = int(y[seed_list[d]])
                    kind = "self" if owner == p else ("spurious" if owner < 0 else "foreign")
                    rec.update(suppressor_particle=owner, suppressor_beta=float(beta[seed_list[d]]),
                               suppressor_survived=bool(survivor_mask[d]))
                    rec["cause"] = "supp_self" if kind == "self" else \
                        f"supp_{kind}_{'direct' if survivor_mask[d] else 'chain'}"
        out[p] = rec
    return out


def hit_categories(y, pop, frozen_hits, causes):
    """Per population hit: one of CATS (reach / oor / seedless_<cause>)."""
    yp = np.asarray(y)[np.asarray(pop, bool)]
    cat = np.where(frozen_hits["reach"], "reach", "oor").astype(object)
    sl = frozen_hits["seedless"]
    for i in np.nonzero(sl)[0]:
        cat[i] = "seedless_" + causes.get(int(yp[i]), {}).get("cause", "unattributed")
    return cat


def size_bin(n):
    return int(np.searchsorted(SIZE_BINS, n, side="left"))


def reading(shares):
    """Pre-registered reading for one cache from hit-weighted shares of U = seedless + out-of-reach."""
    U = shares["seedless"] + shares["oor"]
    if U <= 0:
        return "MIXED", {}
    f = {"supp": shares["seedless_supp"] / U, "sub": shares["seedless_sub"] / U, "oor": shares["oor"] / U}
    if f["supp"] >= 0.5:
        return "SUPPRESSION", f
    if f["sub"] >= 0.5:
        return "THRESHOLD", f
    if f["oor"] >= 0.5:
        return "REACH", f
    return "MIXED", f


def combine_readings(rs):
    return rs[0] if rs and all(r == rs[0] for r in rs) else "MIXED"


# =========================================================================== #
# torch / data layer
# =========================================================================== #
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def within_matrix(mat, ox, bf, tau, t_d, dev):
    """Seed list and the 'within t_d' matrix, computed with exactly the deployed replica's chunking and cdist."""
    import torch
    seed_idx = (bf > tau).nonzero(as_tuple=False).squeeze(1)
    capped = 0
    if seed_idx.numel() > mat.DEFAULT_MAX_SEEDS:
        capped = int(seed_idx.numel() - mat.DEFAULT_MAX_SEEDS)
        seed_idx = seed_idx[torch.topk(bf[seed_idx], mat.DEFAULT_MAX_SEEDS).indices]
    ks = seed_idx.numel()
    S = ox[seed_idx]
    within = torch.zeros(ks, ks, dtype=torch.bool, device=dev)
    if ks:
        chunk = mat._chunk_for(ks, mat.DEFAULT_MAX_PAIRS)
        for i in range(0, ks, chunk):
            q = slice(i, min(i + chunk, ks))
            within[q] = torch.cdist(S[q], S) <= t_d
    return seed_idx, within, capped


def knn(mat, Q, C, m):
    """(n, min(m, K)) nearest-candidate indices, nearest first, with the deployed replica's cdist mode rule."""
    import torch
    K = C.size(0)
    mm = min(m, K)
    mode = "donot_use_mm_for_euclid_dist" if K <= 25 else "use_mm_for_euclid_dist"
    step = max(1, min(max(Q.size(0), 1), mat.DEFAULT_MAX_PAIRS // max(1, K)))
    out = []
    for i in range(0, Q.size(0), step):
        d = torch.cdist(Q[i:i + step], C, compute_mode=mode)
        out.append(torch.topk(d, mm, dim=1, largest=False).indices)
    return torch.cat(out) if out else torch.zeros(0, mm, dtype=torch.long, device=Q.device)


def ari(pred, true):
    """ARI of the population hits -- identical to mat_bench.event_quality's ari_particle (n >= 2 else NaN)."""
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(true, pred)) if len(true) >= 2 else NAN


def analyse_event(ev, mat, s0, dev, a, max_attr_seeds):
    import torch
    fg, pop, y_i, ysem = s0.event_truth(ev)
    fg_idx = fg.nonzero(as_tuple=False).squeeze(1)
    if fg_idx.numel() == 0 or not bool(pop.any()):
        return None
    ox = ev["ox"].to(dev)[fg_idx.to(dev)]
    bf = ev["beta"].to(dev)[fg_idx.to(dev)]
    y = y_i[fg_idx].numpy().astype(np.int64)
    popm = pop[fg_idx].numpy().astype(bool)
    sem = ysem[fg_idx].numpy().astype(np.int64)
    b = bf.float().cpu().numpy()
    tau = s0.make_rule(FROZEN_RULE)(bf.double().cpu())
    cp, lab_fg, D = s0.seeds_and_labels(mat, ox, bf, tau, a.t_d, float("inf"), dev)
    if cp.numel() == 0:
        return {"no_cluster": True}
    pop_pos = torch.as_tensor(np.nonzero(popm)[0], device=dev)
    Qp = ox[pop_pos]

    # ---- FROZEN + attribution ----
    seed_idx, within, capped = within_matrix(mat, ox, bf, tau, a.t_d, dev)
    seed_list = seed_idx.cpu().numpy()
    attributable = seed_list.size <= max_attr_seeds
    if attributable:
        W = within.cpu().numpy()
        dominated, dominator = deployed_from_within(W, b[seed_list])
        survivors = seed_list[~dominated]
        a2_ok = set(survivors.tolist()) == set(cp.cpu().numpy().tolist())
        greedy = seed_list[greedy_from_within(W, b[seed_list])]
    else:
        dominated = dominator = None
        a2_ok, greedy = None, None
    del within
    cand = {"FROZEN": cp.cpu().numpy()}
    topm = {"FROZEN": torch.topk(D[pop_pos], min(M_REACH, D.size(1)), dim=1, largest=False).indices.cpu().numpy()}
    if greedy is not None:
        cand["GREEDY"] = greedy
    cand["NONMS"] = seed_list
    c15, _, _ = s0.seeds_and_labels(mat, ox, bf, tau, 0.15, float("inf"), dev)
    cand["TD015"] = c15.cpu().numpy()
    tau_low = s0.make_rule(LOWTAU_RULE)(bf.double().cpu())
    cl, _, _ = s0.seeds_and_labels(mat, ox, bf, tau_low, a.t_d, float("inf"), dev)
    cand["LOWTAU"] = cl.cpu().numpy()
    orc = oracle_seeds(y, popm, b)
    cand["ORACLE-SEEDS"] = np.array(sorted(orc.values()), np.int64)
    frozen_seeded = set(y[cand["FROZEN"]][y[cand["FROZEN"]] >= 0].tolist())
    missing = [orc[p] for p in sorted(orc) if p not in frozen_seeded]
    cand["ORACLE-ADD"] = np.concatenate([cand["FROZEN"], np.array(missing, np.int64)])
    for name, c in cand.items():
        if name not in topm:
            topm[name] = (knn(mat, Qp, ox[torch.as_tensor(c, device=dev)], M_REACH).cpu().numpy()
                          if c.size else np.zeros((popm.sum(), 0), np.int64))

    yp = y[popm]
    row = {"batch": int(ev["batch"]), "event": int(ev["event"]), "n_fg": int(fg_idx.numel()),
           "n_pop": int(popm.sum()), "n_true": int(len(set(yp.tolist()))), "tau": float(tau),
           "n_above": int((b > tau).sum()), "capped": capped, "attributable": int(attributable),
           "a2_ok": (NAN if a2_ok is None else int(a2_ok))}
    hits = {}
    for name in SETS:
        if name not in cand:
            continue
        if cand[name].size and topm[name].shape[1] == 0:
            continue
        st, h = set_stats(cand[name], y, popm, b, topm[name])
        hits[name] = h
        for k, v in st.items():
            row[f"{name}.{k}"] = v
        for lab in ("lab_dec", "lab_or20", "lab_all"):
            row[f"{name}.ari_{lab[4:]}"] = ari(h[lab], yp)

    # ---- causes (FROZEN), categories, representation ----
    causes = classify_seedless(y, popm, b, tau, seed_list,
                               dominated if (attributable and a2_ok) else None,
                               dominator if (attributable and a2_ok) else None,
                               (~dominated) if attributable else None, frozen_seeded)
    cat = hit_categories(y, popm, hits["FROZEN"], causes)
    for c in CATS:
        row[f"cat.{c}"] = int((cat == c).sum())
    sl = hits["FROZEN"]["seedless"]
    row["rep.seedless_nearest_own_oracle_seeds"] = int((hits["ORACLE-SEEDS"]["nearest_own"] & sl).sum())
    row["rep.seedless_nearest_own_oracle_add"] = int((hits["ORACLE-ADD"]["nearest_own"] & sl).sum())
    row["rep.all_nearest_own_oracle_seeds"] = int(hits["ORACLE-SEEDS"]["nearest_own"].sum())
    if "GREEDY" in hits:
        supp = np.array([c.startswith("seedless_supp") for c in cat])
        row["greedy.supp_hits_recovered"] = int((supp & ~hits["GREEDY"]["seedless"]).sum())
        row["greedy.supp_hits"] = int(supp.sum())

    # per-class / per-size counts and per-particle records
    sizes = np.bincount(yp)[yp]
    groups = {}
    for key, g in (("sem", sem[popm]), ("size", np.array([size_bin(s) for s in sizes]))):
        for gv in np.unique(g):
            sel = g == gv
            groups[(key, int(gv))] = {c: int(((cat == c) & sel).sum()) for c in CATS}
    prows = []
    greedy_parts = set(y[cand["GREEDY"]].tolist()) if "GREEDY" in cand else None
    for p, rec in causes.items():
        own = yp == p
        prows.append({"batch": row["batch"], "event": row["event"], "particle": p,
                      "n_pop_hits": int(own.sum()), "semantic": int(np.bincount(sem[popm][own]).argmax()),
                      "nearest_own_oracle_seeds": float(hits["ORACLE-SEEDS"]["nearest_own"][own].mean()),
                      "greedy_seeded": (int(p in greedy_parts) if greedy_parts is not None else NAN),
                      **rec})
    return {"row": row, "groups": groups, "particles": prows}


def check_replica(mat, ev, s0, dev, a):
    """A1: the torch replica equals materialize_labels (as stage0 --check)."""
    import torch
    fg, _, _, _ = s0.event_truth(ev)
    fg_idx = fg.nonzero(as_tuple=False).squeeze(1)
    if fg_idx.numel() == 0:
        return True
    ox = ev["ox"].to(dev)[fg_idx.to(dev)]
    bf = ev["beta"].to(dev)[fg_idx.to(dev)]
    tau = s0.make_rule(FROZEN_RULE)(bf.double().cpu())
    _, lab_fg, _ = s0.seeds_and_labels(mat, ox, bf, tau, a.t_d, float("inf"), dev)
    ref, _ = mat.materialize_labels("seed", ev["ox"].to(dev), ev["beta"].to(dev), fg.to(dev), None, 1,
                                    t_beta=tau, t_d=a.t_d, t_assign=float("inf"))
    return bool(torch.equal(ref, s0.full_labels(fg.numel(), fg_idx.to(dev), lab_fg, dev)))


def cmd_run(a):
    import torch
    s0 = _load("stage0_seed_diag", HERE / "stage0_seed_diag.py")
    mb = s0.mb
    lo, hi = mb.parse_range(a.batches)
    if hi is None or hi > a.deva_end:
        raise SystemExit(f"coverage_anatomy is dev-A only (batches < {a.deva_end}); got {a.batches}")
    import nugraph.models.nugraph3.materialize as mat
    dev = torch.device(f"cuda:{a.device}" if a.device != "cpu" and torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"cache": a.cache, "cache_manifest": json.loads((Path(a.cache) / "manifest.json").read_text()),
            "batches": a.batches, "t_d": a.t_d, "device": str(dev), "frozen_rule": FROZEN_RULE,
            "lowtau_rule": LOWTAU_RULE, "materialize_sha256": mb.sha256(mat.__file__),
            "script_sha256": mb.sha256(__file__), "git": _git(), "argv": sys.argv,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rows, prows, groups = [], [], {}
    a1_checked = a1_fail = n_nocluster = 0
    t0 = time.perf_counter()
    for n, ev in enumerate(mb.iter_cache(a.cache, lo, hi)):
        if n < a.check_events:
            a1_checked += 1
            a1_fail += int(not check_replica(mat, ev, s0, dev, a))
        r = analyse_event(ev, mat, s0, dev, a, a.max_attr_seeds)
        if r is None:
            continue
        if r.get("no_cluster"):
            n_nocluster += 1
            continue
        rows.append(r["row"])
        prows.extend(r["particles"])
        for k, v in r["groups"].items():
            acc = groups.setdefault(f"{k[0]}:{k[1]}", {c: 0 for c in CATS})
            for c in CATS:
                acc[c] += v[c]
        if (n + 1) % 200 == 0:
            print(f"  {n + 1} events  {time.perf_counter() - t0:.0f}s", flush=True)
    mb.write_rows(out / "events.csv", rows)
    mb.write_rows(out / "seedless_particles.csv", prows)
    (out / "groups.json").write_text(json.dumps(groups, indent=1))
    meta.update(n_events=len(rows), n_no_cluster=n_nocluster, a1_checked=a1_checked, a1_failed=a1_fail,
                wall_s=round(time.perf_counter() - t0, 1), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("n_events", "n_no_cluster", "a1_checked", "a1_failed", "wall_s")}))


def _git():
    import subprocess
    try:
        c = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(HERE), "status", "--porcelain"], text=True).strip())
        return {"commit": c, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


# =========================================================================== #
# report (numpy + csv only)
# =========================================================================== #
def read_rows(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else NAN
    except (TypeError, ValueError):
        return NAN


def summarize(root):
    """Hit-weighted shares and event-mean ARIs per candidate set, causes, gates A2/A4 for one run dir."""
    root = Path(root)
    rows = read_rows(root / "events.csv")
    meta = json.loads((root / "meta.json").read_text())
    tot = lambda c: float(np.nansum([fnum(r.get(c)) for r in rows]))
    npop = tot("n_pop")
    s = {"meta": meta, "n_events": len(rows), "n_pop": npop, "sets": {}}
    for name in SETS:
        if f"{name}.K" not in rows[0]:
            continue
        sel = [r for r in rows if r.get(f"{name}.K", "") != ""]
        Ks = np.array([fnum(r[f"{name}.K"]) for r in sel])
        np_ = float(np.nansum([fnum(r["n_pop"]) for r in sel]))
        d = {"events": len(sel), "K_median": float(np.nanmedian(Ks)), "K_p90": float(np.nanpercentile(Ks, 90))}
        for k in ("seedless", "oor", "reach", "own5"):
            d[k] = float(np.nansum([fnum(r[f"{name}.n_{k}"]) for r in sel])) / max(np_, 1)
        d["dup_per_event"] = float(np.nanmean([fnum(r[f"{name}.n_dup"]) for r in sel]))
        d["spur_per_event"] = float(np.nanmean([fnum(r[f"{name}.n_spur"]) for r in sel]))
        for k in ("dec", "or20", "all"):
            d[f"ari_{k}"] = float(np.nanmean([fnum(r[f"{name}.ari_{k}"]) for r in sel]))
        s["sets"][name] = d
    cats = {c: tot(f"cat.{c}") / max(npop, 1) for c in CATS}
    cats["seedless"] = sum(cats[f"seedless_{c}"] for c in CAUSES)
    cats["seedless_supp"] = sum(cats[f"seedless_{c}"] for c in CAUSES if c.startswith("supp"))
    s["cats"] = cats
    s["reading"], s["fractions_of_U"] = reading(cats)
    s["greedy_recovers_share"] = (tot("greedy.supp_hits_recovered") / tot("greedy.supp_hits")
                                  if tot("greedy.supp_hits") else NAN)
    k_ratio = (s["sets"].get("GREEDY", {}).get("K_median", NAN) / s["sets"]["FROZEN"]["K_median"])
    s["greedy_K_ratio"] = k_ratio
    s["greedy_recovers"] = bool(s["greedy_recovers_share"] >= 0.25 and k_ratio <= 1.5)
    sl = tot("cat.seedless_sub") + sum(tot(f"cat.seedless_{c}") for c in CAUSES if c != "sub")
    s["rep_seedless_nearest_own_oracle_seeds"] = tot("rep.seedless_nearest_own_oracle_seeds") / max(sl, 1)
    s["rep_seedless_nearest_own_oracle_add"] = tot("rep.seedless_nearest_own_oracle_add") / max(sl, 1)
    s["rep_all_nearest_own_oracle_seeds"] = tot("rep.all_nearest_own_oracle_seeds") / max(npop, 1)
    s["representation_limited"] = bool(s["rep_seedless_nearest_own_oracle_seeds"] < 0.5)
    # gates
    a2_bad = sum(1 for r in rows if fnum(r.get("a2_ok")) == 0)
    s["A1"] = {"checked": meta.get("a1_checked"), "failed": meta.get("a1_failed"),
               "pass": bool(meta.get("a1_checked", 0) > 0 and meta.get("a1_failed", 1) == 0)}
    s["A2"] = {"mismatch_events": a2_bad, "not_attributable": int(sum(1 for r in rows if r["attributable"] == "0")),
               "pass": a2_bad <= A2_MAX_EVENTS}
    bad4 = 0
    for r in rows:
        f = fnum(r["FROZEN.n_seedless"]) + fnum(r["FROZEN.n_oor"]) + fnum(r["FROZEN.n_reach"])
        c = sum(fnum(r[f"cat.{x}"]) for x in CATS)
        cs = sum(fnum(r[f"cat.seedless_{x}"]) for x in CAUSES)
        if not (f == fnum(r["n_pop"]) == c and cs == fnum(r["FROZEN.n_seedless"])):
            bad4 += 1
    s["A4"] = {"bad_events": bad4, "pass": bad4 == 0}
    s["groups"] = json.loads((root / "groups.json").read_text())
    return s


def anchor_check(s):
    res = {}
    for (name, k), ref in ANCHOR.items():
        got = s["sets"][name][k]
        res[f"{name}.{k}"] = {"ref": ref, "got": got, "ok": abs(got - ref) <= ANCHOR_TOL}
    return {"checks": res, "pass": all(v["ok"] for v in res.values())}


def cmd_report(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = {Path(p).name: summarize(p) for p in a.primary + a.descriptive + ([a.anchor] if a.anchor else [])}
    prim = [Path(p).name for p in a.primary]
    gates = {n: {g: S[n][g] for g in ("A1", "A2", "A4")} for n in S}
    anc = anchor_check(S[Path(a.anchor).name]) if a.anchor else None
    ok = all(v[g]["pass"] for v in gates.values() for g in v) and (anc is None or anc["pass"])
    reading_all = combine_readings([S[n]["reading"] for n in prim])
    L = ["# Coverage anatomy (T1 diagnostic, dev-A) — `coverage_anatomy.md`", "",
         f"**Gates:** {'PASS' if ok else 'FAIL'}" + ("" if ok else " — readings are NOT interpreted"), ""]
    L += ["| run | A1 replica | A2 attribution (mismatch / not attributable) | A4 bookkeeping |", "|---|---|---|---|"]
    for n, g in gates.items():
        L.append(f"| {n} | {g['A1']['failed']}/{g['A1']['checked']} {'PASS' if g['A1']['pass'] else 'FAIL'} | "
                 f"{g['A2']['mismatch_events']} / {g['A2']['not_attributable']} {'PASS' if g['A2']['pass'] else 'FAIL'} | "
                 f"{g['A4']['bad_events']} {'PASS' if g['A4']['pass'] else 'FAIL'} |")
    if anc:
        L += ["", "**A3 anchor (Stage 0 D3, ep29 dev-A):** " + ("PASS" if anc["pass"] else "FAIL"), ""]
        L += [f"- {k}: {v['got']:.4f} (ref {v['ref']}) {'ok' if v['ok'] else 'OUT OF TOLERANCE'}"
              for k, v in anc["checks"].items()]
    L += ["", f"## Reading (primary runs {', '.join(prim)}): **{reading_all if ok else 'NOT READ (gate failure)'}**", ""]
    L += ["| run | U = seedless + out-of-reach | threshold / U | suppression / U | out-of-reach / U | reading | "
          "GREEDY recovers (share, K ratio) | seedless hits nearest-own under ORACLE-SEEDS / ORACLE-ADD |",
          "|---|---|---|---|---|---|---|---|"]
    for n, s in S.items():
        c, f = s["cats"], s["fractions_of_U"]
        L.append(f"| {n} | {c['seedless'] + c['oor']:.3f} | {f.get('sub', NAN):.3f} | {f.get('supp', NAN):.3f} | "
                 f"{f.get('oor', NAN):.3f} | {s['reading']} | {s['greedy_recovers_share']:.3f}, "
                 f"{s['greedy_K_ratio']:.2f} ({'yes' if s['greedy_recovers'] else 'no'}) | "
                 f"{s['rep_seedless_nearest_own_oracle_seeds']:.3f} / {s['rep_seedless_nearest_own_oracle_add']:.3f}"
                 f"{' (REPRESENTATION-LIMITED)' if s['representation_limited'] else ''} |")
    L += ["", "## FROZEN hit categories (hit-weighted shares of population hits)", "",
          "| run | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
    for n, s in S.items():
        L.append(f"| {n} | " + " | ".join(f"{s['cats'][c]:.3f}" for c in CATS) + " |")
    L += ["", "## Candidate sets (ORACLE-* are truth upper bounds; ARIs are ceilings / distance decoders, not heads)", ""]
    for n, s in S.items():
        L += [f"**{n}** ({s['n_events']} events)", "",
              "| set | K med / p90 | seedless | out-of-reach@20 | reach@20 | own@5 | dup/ev | spur/ev | "
              "ARI nearest | ARI top-20 oracle | ARI all-cand oracle |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in s["sets"].items():
            L.append(f"| {name} | {d['K_median']:.0f} / {d['K_p90']:.0f} | {d['seedless']:.3f} | {d['oor']:.3f} | "
                     f"{d['reach']:.3f} | {d['own5']:.3f} | {d['dup_per_event']:.1f} | {d['spur_per_event']:.1f} | "
                     f"{d['ari_dec']:.4f} | {d['ari_or20']:.4f} | {d['ari_all']:.4f} |")
        L.append("")
    L += ["## FROZEN categories by true class and particle size (primary runs pooled; hit shares within group)", "",
          "| group | hits | " + " | ".join(CATS) + " |", "|---|---|" + "---|" * len(CATS)]
    pooled = {}
    for n in prim:
        for g, v in S[n]["groups"].items():
            acc = pooled.setdefault(g, {c: 0 for c in CATS})
            for c in CATS:
                acc[c] += v[c]
    for g in sorted(pooled, key=lambda x: (x.split(":")[0], int(x.split(":")[1]))):
        v = pooled[g]
        h = sum(v.values())
        k, i = g.split(":")
        lab = (SEM_NAMES[int(i)] if k == "sem" and 0 <= int(i) < len(SEM_NAMES) else
               f"size ({SIZE_BINS[int(i) - 1] if int(i) else 0}, {SIZE_BINS[int(i)] if int(i) < len(SIZE_BINS) else 'inf'}]")
        L.append(f"| {lab} | {h} | " + " | ".join(f"{v[c] / max(h, 1):.3f}" for c in CATS) + " |")
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text)
    (out / "report.json").write_text(json.dumps({"gates_pass": ok, "reading": reading_all, "anchor": anc,
                                                 "runs": {n: {k: v for k, v in s.items() if k != "groups"}
                                                          for n, s in S.items()}}, indent=1, default=str))
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batches", default="0:150")
    s.add_argument("--deva-end", type=int, default=150)
    s.add_argument("--device", default="0")
    s.add_argument("--t-d", type=float, default=0.3)
    s.add_argument("--check-events", type=int, default=200)
    s.add_argument("--max-attr-seeds", type=int, default=8000)
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("--primary", nargs="+", required=True)
    s.add_argument("--descriptive", nargs="*", default=[])
    s.add_argument("--anchor", default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
