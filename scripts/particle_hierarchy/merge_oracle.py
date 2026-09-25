#!/usr/bin/env python3
"""Stage 1a-R4: how much of the oracle@1 gain can candidate-level merging recover? Diagnostic only, dev-A.

Pre-registration: experiments/particle_hierarchy/merge_oracle.md (written before this code ran).

All oracles start from the head's own picks (pick_only) and never re-discriminate a hit:
  P   pick_only                      L   deployed learned merge (pick_merge variant)
  ML  perfect merge over the deployed merge's reachable (local) pairs
  M   perfect merge of every candidate of a particle
  H1  Stage 1a-R3 oracle@1: correct hits -> their primary, others keep their pick
Decomposition of H1 - P: merging = M - P, un-sharing = H1 - M, scope limit = M - ML.

  run     (EAF, torch) one M=20 head on a dev-A cache -> merge_events.csv + merge_summary.json
  report  (pure python) six heads -> gates Q1-Q4 and the pre-registered MERGE / UNSHARE reading
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ranking_diag", HERE / "ranking_diag.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)

HEADS = rd.HEADS
EP29_DIAG = rd.EP29_DIAG
MAX_FLIPS = rd.MAX_FLIPS
KS_RESOLVE = (2, 3, 5)
ORACLES = ("P", "L", "ML", "M", "H1") + tuple(f"M+H@{k}" for k in KS_RESOLVE)


# --------------------------------------------------------------------------- #
# per-event oracles (numpy)
# --------------------------------------------------------------------------- #
def merge_oracles(W, nb, own, pop, owner, cand_pid, cand_primary, local_root, ari_fn, learned=None):
    """Oracle labels, ARIs, hit-level counts and construction checks for one event.

    W, nb, own, pop, owner as in ranking_diag.analyse_event; cand_pid (K,) particle of each candidate
    (-1 spurious); cand_primary (K,) primary candidate of that particle (-1 for spurious); local_root (K,)
    component root after merging same-particle pairs in the deployed (local) pair set; learned: labels of
    the deployed learned merge over the same hits (or None). ari_fn(labels) -> particle ARI.
    """
    import numpy as np
    n = W.shape[0]
    rows = np.arange(n)
    order_h = np.argsort(-W, axis=1, kind="stable")
    rank_h = rd.first_true(np.take_along_axis(own, order_h, axis=1))
    avail = pop & own.any(axis=1)
    correct = avail & (rank_h == 1)
    top1 = nb[rows, order_h[:, 0]]
    has_pid = cand_pid[top1] >= 0
    lab = {"P": top1,
           "ML": local_root[top1],
           "M": np.where(has_pid, cand_primary[top1], top1),
           "H1": np.where(correct, owner, top1)}
    for k in KS_RESOLVE:
        lab[f"M+H@{k}"] = np.where(avail & (rank_h >= 1) & (rank_h <= k), owner, lab["M"])
    if learned is not None:
        lab["L"] = learned
    ev = {name: ari_fn(v) for name, v in lab.items()}
    wrong = pop & ~correct
    K = cand_pid.shape[0]
    spur = cand_pid < 0
    checks = {
        "q3_m_eq_h1_on_correct": bool((lab["M"][correct] == lab["H1"][correct]).all()),
        "q3_ml_within_particle": bool((cand_pid[local_root] == cand_pid)[~spur].all()),
        "q3_spurious_singletons": bool((local_root[spur] == np.arange(K)[spur]).all()),
        "q4_primary": bool(((cand_primary >= 0) & (cand_pid[np.clip(cand_primary, 0, None)] == cand_pid))[~spur].all()),
    }
    st = {"n_pop": int(pop.sum()), "n_correct": int(correct.sum()),
          "n_wrong_on_particle": int((wrong & has_pid).sum()),
          "n_wrong_on_spurious": int((wrong & ~has_pid).sum()),
          "n_correct_on_duplicate": int((correct & (top1 != cand_primary[top1])).sum())}
    return ev, st, checks, lab


def local_components(cand_pid, pairs, K):
    """Union-find roots over the same-particle pairs among `pairs` (P, 2); smallest index is root."""
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        if cand_pid[a] >= 0 and cand_pid[a] == cand_pid[b]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return [find(k) for k in range(K)]


# --------------------------------------------------------------------------- #
# run (torch)
# --------------------------------------------------------------------------- #
def cmd_run(a):
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("soft_particle", HERE / "soft_particle.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    lo, hi = sp.mb.parse_range(a.batches)
    if hi is None or hi > a.deva_end:
        raise SystemExit(f"merge_oracle is dev-A only (batches < {a.deva_end}); got {a.batches}")
    mat, dev = sp._setup(a)
    ck = torch.load(a.head, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    sp.apply_head_cfg(a, cfg)
    if cfg["m"] != 20:
        raise SystemExit(f"registered for M=20 heads; this head has m={cfg['m']}")
    model = sp.load_head(ck, dev)
    pick_merge = dict(next(kw for name, kw, _ in sp.VARIANTS if name == "pick_merge"))
    rows, total, bad = [], {}, []
    for d in sp.iter_events(a.cache, lo, hi, mat, sp.s0.make_rule(cfg["rule"]), a, dev):
        with torch.no_grad():
            lg, _, _, _ = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
            W = lg.softmax(dim=1)[:, :d["m"]]
            learned, _ = sp.predict_labels(model, d, **pick_merge)
        fg_idx, ysem, yi = d["fg_idx"].cpu(), d["y_sem"], d["y_i"]
        keep = d["fg"] & (ysem >= 0)
        K = d["K"]
        cand_pid = d["seed_true"].cpu().numpy()
        corr = d["correct"].cpu().numpy()
        cand_primary = np.where(cand_pid >= 0, corr[d["cp"].cpu().numpy()], -1)
        # deployed reachable pairs: local scope, K <= MAX_MERGE_K guard, no truncation
        pairs = sp.candidate_pairs(d["nb"], K, "local", dev, max_pairs=10**12).cpu().tolist()
        local_root = np.array(local_components(cand_pid, pairs, K), dtype=np.int64)
        own = d["own"].cpu().numpy()

        def ari_fn(lab_fg):
            lab = torch.full((d["n_hit"],), -1, dtype=torch.long)
            lab[fg_idx] = torch.as_tensor(lab_fg, dtype=torch.long)
            return sp.mb.event_quality(lab, yi, ysem, keep).get("ari_particle", math.nan)

        ev, st, checks, _ = merge_oracles(
            W.cpu().numpy(), d["nb"].cpu().numpy(), own, d["pop_fg"].cpu().numpy(),
            np.where(own.any(axis=1), corr, -1), cand_pid, cand_primary, local_root, ari_fn,
            learned=learned.cpu().numpy())
        key = (int(d["batch"]), int(d["event"]))
        if not all(checks.values()):
            bad.append({"event": key, **{k: v for k, v in checks.items() if not v}})
        rows.append({"batch": key[0], "event": key[1], "K": K, "n_fg": int(fg_idx.numel()),
                     "n_local_pairs": len(pairs), **ev})
        rd.merge_stats(total, st)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp.mb.write_rows(out / "merge_events.csv", rows)
    res = {"head": a.head, "cache": a.cache, "batches": a.batches, "n_events": len(rows),
           "ari": {o: rd._nanmean([r[o] for r in rows]) for o in ORACLES},
           "stats": total, "gates": {"construction_fail": bad}}
    (out / "merge_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"ari": res["ari"], "construction_fail": len(bad)}, indent=2))
    if bad:
        raise SystemExit(f"FAIL: construction checks failed on {len(bad)} events")


# --------------------------------------------------------------------------- #
# report (pure python)
# --------------------------------------------------------------------------- #
def per_event(path, cols):
    out = {}
    for r in csv.DictReader(open(path)):
        out[(int(r["batch"]), int(r["event"]))] = {c: rd_float(r[c]) for c in cols}
    return out


def rd_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def compare(ref, got, tol=1e-9):
    """(n differing events, |mean difference|, same event set)."""
    if set(ref) != set(got):
        return None, None, False
    bad = [k for k in ref if not (math.isnan(ref[k]) and math.isnan(got[k])) and not abs(ref[k] - got[k]) <= tol]
    md = abs(rd._nanmean(list(ref.values())) - rd._nanmean(list(got.values())))
    return len(bad), md, True


def gates(merge_root, rank_root, mb_root, sums):
    fails, flips = [], {}
    for h, s in sums.items():
        if s["gates"]["construction_fail"]:
            fails.append(f"Q3/Q4 {h}: {len(s['gates']['construction_fail'])} events")
        mine = per_event(Path(merge_root) / h / "merge_events.csv", ("P", "H1", "L"))
        r3 = per_event(Path(rank_root) / h / "rank_events.csv", ("pick_only", "head@1"))
        for mine_col, r3_col in (("P", "pick_only"), ("H1", "head@1")):
            n, md, same = compare({k: v[r3_col] for k, v in r3.items()}, {k: v[mine_col] for k, v in mine.items()})
            if not same:
                fails.append(f"Q1 {h} {mine_col}: event set differs from R3")
                continue
            flips[f"{h}:{mine_col}"] = n
            if n > MAX_FLIPS:
                fails.append(f"Q1 {h} {mine_col}: differs from R3 on {n} events")
        if h in EP29_DIAG:
            p = Path(mb_root) / EP29_DIAG[h] / "diagnose_events.csv"
            ref = {(int(r["batch"]), int(r["event"])): rd_float(r["ari_particle"])
                   for r in csv.DictReader(open(p)) if r["variant"] == "pick_merge"}
            n, md, same = compare(ref, {k: v["L"] for k, v in mine.items()})
            if not same or n > MAX_FLIPS or md > 1e-4:
                fails.append(f"Q2 {h}: learned merge vs diagnose pick_merge: {n} events differ, mean diff {md}")
            flips[f"{h}:L"] = n
    return fails, flips


def head_numbers(s):
    A, st = s["ari"], s["stats"]
    P, L, ML, M, H1 = (A[k] for k in ("P", "L", "ML", "M", "H1"))
    g = lambda num, den: num / den if den > 0 else math.nan
    pop = max(st["n_pop"], 1)
    n = {"ari": A, "S": g(M - P, H1 - P), "scope": g(ML - P, M - P), "realised": g(L - P, ML - P),
         "merging": M - P, "unsharing": H1 - M, "scope_limit": M - ML, "gain_h1": H1 - P,
         "share_correct": st["n_correct"] / pop, "share_wrong_on_particle": st["n_wrong_on_particle"] / pop,
         "share_wrong_on_spurious": st["n_wrong_on_spurious"] / pop,
         "correct_on_duplicate": st["n_correct_on_duplicate"] / max(st["n_correct"], 1)}
    n["reading"] = "MERGE" if n["S"] >= 0.50 else "UNSHARE"
    n["scope_ok"] = n["scope"] >= 0.80
    return n


def verdict(nums):
    counts = {r: sum(1 for x in nums.values() if x["reading"] == r) for r in ("MERGE", "UNSHARE")}
    top = max(counts, key=counts.get)
    if counts[top] < 5:
        return {"counts": counts, "overall": "NOT ESTABLISHED",
                "next": "R3's registered branch: class-information probe"}
    if top == "UNSHARE":
        return {"counts": counts, "overall": "UNSHARE",
                "next": "R3's registered branch: class-information probe (un-sharing is hit-level; a merge head cannot deliver it)"}
    scope_ok = sum(1 for x in nums.values() if x["scope_ok"]) >= 5
    return {"counts": counts, "overall": "MERGE", "scope_ok": scope_ok,
            "next": "better duplicate consolidation before the class-information probe: "
                    + ("improve the merge scorer (the local scope suffices)" if scope_ok
                       else "improve the merge scorer AND widen the merge scope")}


def evaluate(merge_root, rank_root, mb_root):
    sums = {}
    for h in HEADS:
        p = Path(merge_root) / h / "merge_summary.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        sums[h] = json.loads(p.read_text())
    fails, flips = gates(merge_root, rank_root, mb_root, sums)
    nums = {h: head_numbers(s) for h, s in sums.items()}
    res = {"gates": {"pass": not fails, "failures": fails, "flips": flips}, "heads": nums,
           "n_events": {h: s["n_events"] for h, s in sums.items()}}
    res["verdict"] = verdict(nums) if not fails else {"overall": "INVALID", "next": "a gate failed; no reading"}
    return res


def render(res):
    f4, N, V = rd.f4, res["heads"], res["verdict"]
    hdr = "| | " + " | ".join(HEADS) + " |"
    sep = "|---" * (len(HEADS) + 1) + "|"
    row = lambda name, fn: f"| {name} | " + " | ".join(fn(N[h]) for h in HEADS) + " |"
    L = ["# Stage 1a-R4 — merging vs un-sharing in the oracle@1 gain (dev-A, pre-registered)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += [f"- per-event differences vs R3 / diagnose (tolerated ≤ {MAX_FLIPS}): {res['gates']['flips']}", "",
          f"**Verdict:** {V['overall']}" + (f" — readings {V['counts']}" if "counts" in V else "")
          + f"; next: {V['next']}", "",
          "## Decomposition", "", hdr, sep,
          row("reading", lambda x: x["reading"]),
          row("**S** = (M − P)/(H1 − P)", lambda x: f4(x["S"])),
          row("oracle@1 gain H1 − P", lambda x: f4(x["gain_h1"])),
          row("merging M − P", lambda x: f4(x["merging"])),
          row("un-sharing H1 − M", lambda x: f4(x["unsharing"])),
          row("local-scope reach (ML − P)/(M − P)", lambda x: f4(x["scope"])),
          row("learned merge realised (L − P)/(ML − P)", lambda x: f4(x["realised"])),
          "", "## Oracle ARI (event mean)", "", hdr, sep]
    for o in ORACLES:
        L.append(row(o, lambda x, o=o: f4(x["ari"][o])))
    L += ["", "## Hits (share of population hits)", "", hdr, sep,
          row("correct pick", lambda x: f4(x["share_correct"])),
          row("wrong pick on a particle's candidate (M moves these)", lambda x: f4(x["share_wrong_on_particle"])),
          row("wrong pick on a spurious candidate", lambda x: f4(x["share_wrong_on_spurious"])),
          row("correct picks landing on a duplicate (of correct)", lambda x: f4(x["correct_on_duplicate"])),
          "| events | " + " | ".join(str(res["n_events"][h]) for h in HEADS) + " |"]
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(a.merge_root, a.rank_root, a.mb_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "report.md").write_text(render(res))
    print(render(res))
    if not res["gates"]["pass"]:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--head", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batches", default="0:150")
    s.add_argument("--deva-end", type=int, default=150)
    s.add_argument("--device", default="0")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("--merge-root", required=True)
    s.add_argument("--rank-root", required=True, help="Stage 1a-R3 outputs (Q1)")
    s.add_argument("--mb-root", required=True, help="norm_ep29 dir (Q2)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
