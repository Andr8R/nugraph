#!/usr/bin/env python3
"""Stage 1a-R3: where is the owner in the head's ranking? Diagnostic only, dev-A only.

Pre-registration: experiments/particle_hierarchy/ranking_diag.md (written before this code ran).

  run     (EAF, torch) one trained M=20 head on a dev-A cache -> per-event CSV + summary JSON:
          head-vs-distance top-k oracle ARI, owner-rank distribution on misses, confidence/margin
          histograms, error types. Gates R1 (pick_only == evaluated decode path), R3, R4 in-run.
  report  (pure python) the six summaries -> gates (incl. R2 vs the earlier ep29 diagnose), the
          pre-registered per-head readings (AMBIG / INFO / MIXED) and the overall verdict.

The per-event analysis (`analyse_event`) is plain numpy so it is testable without torch.
"""
import argparse
import csv
import json
import math
from pathlib import Path

KS = (1, 2, 3, 5, 10, 20)
RANK_BINS = ((2, 2), (3, 3), (4, 5), (6, 10), (11, 20))
N_CONF = 200                                # histogram bins for p1 and margin on [0, 1]
LOW_MARGIN = 0.1
N_CLS = 7                                   # semantic classes (population hits have y_sem in 0..6)
HEADS = ("ep29_s1", "ep29_s2", "T_s1", "T_s2", "C_s1", "C_s2")
EP29_DIAG = {"ep29_s1": "retrieval/diag_m20_s1", "ep29_s2": "retrieval/diag_m20_s2"}
# R1/R2 compare two separate GPU forward passes; a near-tie argmax can flip between them (seen once in
# the M5 timing gate), so up to MAX_FLIPS events may differ -- they are listed, never silently absorbed.
MAX_FLIPS = 2


def bin_label(lo, hi):
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


# --------------------------------------------------------------------------- #
# per-event analysis (numpy)
# --------------------------------------------------------------------------- #
def first_true(flags):
    """1-based column of the first True per row, 0 if none. flags (n, m) bool."""
    import numpy as np
    any_ = flags.any(axis=1)
    return np.where(any_, flags.argmax(axis=1) + 1, 0)


def analyse_event(W, nb, own, pop, owner, cand_pid, cand_cls, hit_cls, ari_fn, ks=KS,
                  hit_pcls=None, cand_pcls=None):
    """One event's labels, ARIs and hit-level counts.

    W (n, m) membership probabilities over the hit's m candidates (columns in distance order);
    nb (n, m) candidate ids; own (n, m) candidate belongs to the hit's particle; pop (n,) population
    hit; owner (n,) canonical candidate of the hit's particle (-1 none); cand_pid / cand_cls (K,)
    particle id / true class of each candidate's seed hit; hit_cls (n,) true class of each hit;
    ari_fn(labels over these n hits) -> particle ARI of the event.
    hit_pcls / cand_pcls (optional, Stage 1a-R6): the trunk's PREDICTED class of each hit / of each
    candidate's seed hit; adds the addressability split of the different-class misses.
    """
    import numpy as np
    n, m = W.shape
    rows = np.arange(n)
    order_h = np.argsort(-W, axis=1, kind="stable")          # ties resolve like argmax
    own_h = np.take_along_axis(own, order_h, axis=1)
    rank_h, rank_d = first_true(own_h), first_true(own)
    avail = pop & own.any(axis=1)
    top1 = nb[rows, order_h[:, 0]]                            # pick_only
    labels = {"pick_only": top1}
    for k in ks:
        kk = min(k, m)
        for tag, r in (("head", rank_h), ("dist", rank_d)):
            res = avail & (r >= 1) & (r <= kk)
            labels[f"{tag}@{k}"] = np.where(res, owner, top1)
    ev = {name: ari_fn(lab) for name, lab in labels.items()}

    miss = avail & (rank_h != 1)
    p = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
    ps = -np.sort(-p, axis=1)
    p1 = ps[:, 0]
    margin = p1 - (ps[:, 1] if m > 1 else 0.0)
    ok = avail & (rank_h == 1)
    conf_bin = lambda x: np.clip((x * N_CONF).astype(int), 0, N_CONF - 1)
    picked = top1[miss]
    pid, ccls = cand_pid[picked], cand_cls[picked]
    st = {
        "n_pop": int(pop.sum()), "n_avail": int(avail.sum()), "n_correct": int(ok.sum()),
        "n_miss": int(miss.sum()), "owner_missing": int((avail & (owner < 0)).sum()),
        "demoted": int((miss & (rank_d == 1)).sum()),         # nearest was own, head chose otherwise
        "recall_head": [int((avail & (rank_h >= 1) & (rank_h <= k)).sum()) for k in ks],
        "recall_dist": [int((avail & (rank_d >= 1) & (rank_d <= k)).sum()) for k in ks],
        "miss_rank_head": [int((miss & (rank_h >= lo) & (rank_h <= hi)).sum()) for lo, hi in RANK_BINS],
        "miss_rank_dist": [int((miss & (rank_d == 1)).sum())]
                          + [int((miss & (rank_d >= lo) & (rank_d <= hi)).sum()) for lo, hi in RANK_BINS],
        "p1_ok": np.bincount(conf_bin(p1[ok]), minlength=N_CONF).tolist(),
        "p1_miss": np.bincount(conf_bin(p1[miss]), minlength=N_CONF).tolist(),
        "margin_ok_low": int((margin[ok] < LOW_MARGIN).sum()),
        "margin_miss_low": int((margin[miss] < LOW_MARGIN).sum()),
        "err_spurious": int((pid < 0).sum()),
        "err_same_class": int(((pid >= 0) & (ccls == hit_cls[miss])).sum()),
        "err_diff_class": int(((pid >= 0) & (ccls != hit_cls[miss])).sum()),
    }
    if hit_pcls is not None:
        diff = (pid >= 0) & (ccls != hit_cls[miss])
        same_pred = cand_pcls[picked] == hit_pcls[miss]
        st["err_diff_pred_same"] = int((diff & same_pred).sum())     # trunk-confused
        st["err_diff_pred_diff"] = int((diff & ~same_pred).sum())    # addressable by predicted class
        # predicted-class disagreement between a hit and the candidate it picked, on CORRECT picks too:
        # a link-rejection signal is only useful if it fires more on wrong links than on right ones
        dis = hit_pcls != cand_pcls[top1]
        hit_wrong = hit_pcls != hit_cls                              # the trunk misclassifies the hit
        st["ok_pred_diff"] = int((ok & dis).sum())
        st["miss_pred_diff"] = int((miss & dis).sum())
        st["err_same_pred_diff"] = int(((pid >= 0) & (ccls == hit_cls[miss]) & ~same_pred).sum())
        st["err_spurious_pred_diff"] = int(((pid < 0) & ~same_pred).sum())
        st["ok_hit_mispredicted"] = int((ok & hit_wrong).sum())
        st["miss_hit_mispredicted"] = int((miss & hit_wrong).sum())
        st["ok_dis_hit_mispredicted"] = int((ok & dis & hit_wrong).sum())
        for tag, sel in (("ok", ok), ("miss", miss)):                # by the hit's TRUE class
            st[f"{tag}_by_cls"] = [int((sel & (hit_cls == c)).sum()) for c in range(N_CLS)]
            st[f"{tag}_dis_by_cls"] = [int((sel & dis & (hit_cls == c)).sum()) for c in range(N_CLS)]
        # Stage 1a-R7 Step 1: who did the trunk get wrong on each different-class miss?
        # cand_cls = TRUE class of a candidate's seed hit, cand_pcls = its PREDICTED class.
        hc, hp = hit_cls[miss], hit_pcls[miss]
        cc, cpk = ccls, cand_pcls[picked]
        dm = (pid >= 0) & (cc != hc)                                 # different-class misses
        hw, cw = hp != hc, cpk != cc                                 # hit / picked seed mispredicted
        cells = {"cand_disguised": ~hw & cw & (cpk == hc),           # seed predicted AS the hit's class
                 "hit_disguised": hw & ~cw & (hp == cc),             # hit predicted AS the candidate's class
                 "both_correct": ~hw & ~cw,                          # trunk separates them; head linked anyway
                 "both_wrong": hw & cw,
                 "seed_wrong_third": ~hw & cw & (cpk != hc),
                 "hit_wrong_third": hw & ~cw & (hp != cc)}
        for k, m in cells.items():
            st[f"dm_{k}"] = int((dm & m).sum())
            st[f"dm_{k}_by_cls"] = [int((dm & m & (hc == c)).sum()) for c in range(N_CLS)]
        st["dm_by_cls"] = [int((dm & (hc == c)).sum()) for c in range(N_CLS)]
        # owner side: is the CORRECT candidate's seed mispredicted (so the owner looks foreign)?
        # on a miss: the first own candidate in the head's ranking; on a correct pick: the picked one
        owner_col = np.take_along_axis(order_h, np.clip(rank_h - 1, 0, None)[:, None], axis=1)[:, 0]
        owner = nb[rows, owner_col]
        seed_wrong = cand_pcls != cand_cls
        st["miss_owner_seed_wrong"] = int((miss & seed_wrong[owner]).sum())
        # Stage 1a-R8 geometry counters: inside each cell, is the owner's seed mispredicted, and is the picked
        # candidate nearer in the latent than the owner (columns of nb are in distance order)?
        osw = seed_wrong[owner[miss]]
        nearer = order_h[miss, 0] < owner_col[miss]
        for k, m in cells.items():
            st[f"dm_{k}_owner_seed_wrong"] = int((dm & m & osw).sum())
            st[f"dm_{k}_picked_nearer"] = int((dm & m & nearer).sum())
            st[f"dm_{k}_nearer_owner_seed_ok"] = int((dm & m & nearer & ~osw).sum())
        st["ok_owner_seed_wrong"] = int((ok & seed_wrong[top1]).sum())
        for tag, sel, cand in (("miss", miss, owner), ("ok", ok, top1)):
            st[f"{tag}_owner_seed_wrong_by_cls"] = [int((sel & seed_wrong[cand] & (hit_cls == c)).sum())
                                                    for c in range(N_CLS)]
    return ev, st, labels


def same_partition(a, b):
    """True if two label vectors define the same partition (ids may differ)."""
    fwd, bwd = {}, {}
    for x, y in zip(a, b):
        if fwd.setdefault(x, y) != y or bwd.setdefault(y, x) != x:
            return False
    return True


def merge_stats(total, st):
    for k, v in st.items():
        if isinstance(v, list):
            total[k] = [a + b for a, b in zip(total.get(k, [0] * len(v)), v)]
        else:
            total[k] = total.get(k, 0) + v
    return total


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie) from two histograms over the same bins."""
    n_pos, n_neg = sum(pos), sum(neg)
    if not n_pos or not n_neg:
        return math.nan
    below, acc = 0, 0.0
    for p_, q_ in zip(pos, neg):
        acc += p_ * (below + 0.5 * q_)
        below += q_
    return acc / (n_pos * n_neg)


# --------------------------------------------------------------------------- #
# run (torch)
# --------------------------------------------------------------------------- #
def cmd_run(a):
    import importlib.util
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("soft_particle", Path(__file__).with_name("soft_particle.py"))
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    lo, hi = sp.mb.parse_range(a.batches)
    if hi is None or hi > a.deva_end:
        raise SystemExit(f"ranking_diag is dev-A only (batches < {a.deva_end}); got {a.batches}")
    mat, dev = sp._setup(a)
    ck = torch.load(a.head, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    sp.apply_head_cfg(a, cfg)
    if cfg["m"] != 20:
        raise SystemExit(f"registered for M=20 heads; this head has m={cfg['m']}")
    model = sp.load_head(ck, dev)
    readout = sp.semantic_readout(a.cache) if a.pred_class else None
    rows, total, r1_fail, r3_fail = [], {}, [], []
    for d in sp.iter_events(a.cache, lo, hi, mat, sp.s0.make_rule(cfg["rule"]), a, dev):
        with torch.no_grad():
            lg, _, _, _ = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
            W = lg.softmax(dim=1)[:, :d["m"]]
            ref_fg, _ = sp.predict_labels(model, d, **sp._OFF)
        fg_idx = d["fg_idx"]
        ysem, yi = d["y_sem"], d["y_i"]
        keep = d["fg"] & (ysem >= 0)
        pop = d["pop_fg"]                      # prepare's population: fg & labelled & has a particle

        def ari_fn(lab_fg):
            lab = torch.full((d["n_hit"],), -1, dtype=torch.long)
            lab[fg_idx.cpu()] = torch.as_tensor(lab_fg, dtype=torch.long)
            return sp.mb.event_quality(lab, yi, ysem, keep).get("ari_particle", math.nan)

        cand_hit = fg_idx.cpu()[d["cp"].cpu()]
        pc = {}
        if a.pred_class:
            emb = d["hit_x"][:, :readout[0].shape[1]]                # original h.x, not probe channels
            hp = sp.predicted_logits(emb, readout).argmax(dim=1).cpu().numpy()
            pc = {"hit_pcls": hp, "cand_pcls": hp[d["cp"].cpu().numpy()]}
        ev, st, labels = analyse_event(
            W.cpu().numpy(), d["nb"].cpu().numpy(), d["own"].cpu().numpy(), pop.cpu().numpy(),
            np.where(d["own"].any(dim=1).cpu().numpy(), d["correct"].cpu().numpy(), -1),
            d["seed_true"].cpu().numpy(), ysem[cand_hit].cpu().numpy(), ysem[fg_idx.cpu()].cpu().numpy(),
            ari_fn, **pc)
        key = (int(d["batch"]), int(d["event"]))
        if not same_partition(labels["pick_only"].tolist(), ref_fg.cpu().tolist()):
            r1_fail.append(key)
        if not np.array_equal(labels["head@20"], labels["dist@20"]):
            r3_fail.append(key)
        rows.append({"batch": key[0], "event": key[1], "n_fg": int(fg_idx.numel()), "K": d["K"],
                     "n_avail": st["n_avail"], "n_miss": st["n_miss"], **ev})
        merge_stats(total, st)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp.mb.write_rows(out / "rank_events.csv", rows)
    aris = {name: _nanmean([r[name] for r in rows]) for name in rows[0]
            if name == "pick_only" or "@" in name}
    res = {"head": a.head, "cache": a.cache, "batches": a.batches, "head_cfg_m": cfg["m"],
           "n_events": len(rows), "ari": aris, "stats": total, "ks": list(KS),
           "rank_bins": [bin_label(*b) for b in RANK_BINS],
           "gates": {"R1_pick_only_fail": r1_fail, "R3_head_eq_dist_at_M_fail": r3_fail,
                     "R4_owner_missing": total.get("owner_missing", 0)}}
    (out / "rank_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"ari": aris, "gates": res["gates"]}, indent=2))
    if len(r1_fail) > MAX_FLIPS or r3_fail or res["gates"]["R4_owner_missing"]:
        raise SystemExit("FAIL: an in-run gate failed (see rank_summary.json)")


def _nanmean(v):
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return sum(v) / len(v) if v else math.nan


# --------------------------------------------------------------------------- #
# report (pure python)
# --------------------------------------------------------------------------- #
def head_numbers(s):
    A, st = s["ari"], s["stats"]
    H = {k: A[f"head@{k}"] for k in KS}
    D = {k: A[f"dist@{k}"] for k in KS}
    C, H1 = H[KS[-1]], H[1]
    span = C - H1
    miss = max(st["n_miss"], 1)
    avail = max(st["n_avail"], 1)
    rank_share = [c / miss for c in st["miss_rank_head"]]
    n = {"pick_only": A["pick_only"], "H": H, "D": D, "C": C,
         "closure2": (H[2] - H1) / span if span > 0 else math.nan,
         "closure3": (H[3] - H1) / span if span > 0 else math.nan,
         "rank2_share": rank_share[0], "miss_rank_share": rank_share,
         "miss_rank_dist_share": [c / miss for c in st["miss_rank_dist"]],
         "demoted_share": st["demoted"] / miss,
         "auroc": auroc(st["p1_ok"], st["p1_miss"]),
         "low_margin_miss": st["margin_miss_low"] / miss,
         "low_margin_ok": st["margin_ok_low"] / max(st["n_correct"], 1),
         "recall_head": [c / avail for c in st["recall_head"]],
         "recall_dist": [c / avail for c in st["recall_dist"]],
         "err": {k: st[f"err_{k}"] / miss for k in ("spurious", "same_class", "diff_class")},
         "avail_of_pop": st["n_avail"] / max(st["n_pop"], 1),
         "correct_of_avail": st["n_correct"] / avail}
    n["reading"] = reading(n)
    return n


def reading(n):
    ambig = n["closure2"] >= 0.50 and n["rank2_share"] >= 0.50 and n["auroc"] >= 0.75
    info = n["closure3"] < 0.35 and n["rank2_share"] < 0.35
    return "AMBIG" if ambig else "INFO" if info else "MIXED"


def verdict(nums):
    counts = {r: sum(1 for x in nums.values() if x["reading"] == r) for r in ("AMBIG", "INFO", "MIXED")}
    top = max(counts, key=counts.get)
    overall = top if counts[top] >= 5 else "MIXED"
    diff = sum(x["err"]["diff_class"] for x in nums.values()) / len(nums)
    spur = sum(x["err"]["spurious"] for x in nums.values()) / len(nums)
    if overall == "AMBIG":
        nxt = "output-side multiple membership: keep the top-2 memberships and resolve with particle-level context"
    else:
        probe = "class-information probe" if diff >= 0.50 else "hit-to-hit context probe"
        nxt = (f"richer head information first: {probe}"
               + ("; multiple membership after" if overall == "MIXED" else ""))
    flag = spur >= 0.30
    return {"counts": counts, "overall": overall, "next": nxt, "mean_diff_class": diff,
            "mean_spurious": spur, "spurious_flag": flag}


def check_r2(root, name, s):
    """ep29 heads: pick_only per event must reproduce the earlier diagnose pick_only."""
    p = Path(root) / EP29_DIAG[name] / "diagnose_events.csv"
    if not p.exists():
        return [f"R2 {name}: no earlier diagnose at {p}"]
    ref = {(int(r["batch"]), int(r["event"])): float(r["ari_particle"])
           for r in csv.DictReader(open(p)) if r["variant"] == "pick_only"}
    ev = Path(s["_dir"]) / "rank_events.csv"
    got = {(int(r["batch"]), int(r["event"])): float(r["pick_only"]) for r in csv.DictReader(open(ev))}
    if set(ref) != set(got):
        return [f"R2 {name}: event sets differ ({len(set(ref) ^ set(got))} events)"]
    bad = [k for k in ref if not (math.isnan(ref[k]) and math.isnan(got[k]))
           and not abs(ref[k] - got[k]) <= 1e-9]
    mean_diff = abs(_nanmean(list(ref.values())) - _nanmean(list(got.values())))
    if len(bad) > MAX_FLIPS or mean_diff > 1e-4:
        return [f"R2 {name}: pick_only differs on {len(bad)} events (mean diff {mean_diff:.2e})"]
    return []


def evaluate(rank_root, mb_root):
    sums, fails = {}, []
    for h in HEADS:
        p = Path(rank_root) / h / "rank_summary.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        s = json.loads(p.read_text())
        s["_dir"] = str(p.parent)
        g = s["gates"]
        if len(g["R1_pick_only_fail"]) > MAX_FLIPS:
            fails.append(f"R1 {h}: {len(g['R1_pick_only_fail'])} events")
        if g["R3_head_eq_dist_at_M_fail"]:
            fails.append(f"R3 {h}: {len(g['R3_head_eq_dist_at_M_fail'])} events")
        if g["R4_owner_missing"]:
            fails.append(f"R4 {h}: {g['R4_owner_missing']} available hits without an owner")
        if h in EP29_DIAG:
            fails += check_r2(mb_root, h, s)
        sums[h] = s
    nums = {h: head_numbers(s) for h, s in sums.items()}
    res = {"gates": {"pass": not fails, "failures": fails}, "heads": nums,
           "flips": {h: s["gates"]["R1_pick_only_fail"] for h, s in sums.items()},
           "n_events": {h: s["n_events"] for h, s in sums.items()}}
    res["verdict"] = (verdict(nums) if not fails
                      else {"overall": "INVALID", "next": "a gate failed; no reading"})
    return res


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def render(res):
    V, N = res["verdict"], res["heads"]
    L = ["# Stage 1a-R3 — owner rank in the head's ranking (dev-A, pre-registered)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += [f"- tolerated near-tie flips (R1, ≤ {MAX_FLIPS}/head): {res['flips']}"]
    L += ["", f"**Verdict:** {V['overall']}"
          + (f" — readings {V['counts']}; next: {V['next']}" if "counts" in V else f" — {V['next']}")]
    if V.get("spurious_flag"):
        L.append(f"**Flag:** spurious candidates are {V['mean_spurious']:.0%} of misses (candidate quality).")
    hdr = "| | " + " | ".join(HEADS) + " |"
    sep = "|---" * (len(HEADS) + 1) + "|"
    row = lambda name, f: f"| {name} | " + " | ".join(f(N[h]) for h in HEADS) + " |"
    L += ["", "## Readings", "", hdr, sep,
          row("reading", lambda x: x["reading"]),
          row("A1/I1 closure@2 = (H2−H1)/(C−H1)", lambda x: f4(x["closure2"])),
          row("I1 closure@3", lambda x: f4(x["closure3"])),
          row("A2/I2 misses with owner at head rank 2", lambda x: f4(x["rank2_share"])),
          row("A3 AUROC(p1)", lambda x: f4(x["auroc"])),
          "", "## Oracle ARI (event mean): head / distance ranking", "", hdr, sep,
          row("pick_only", lambda x: f4(x["pick_only"]))]
    for k in KS:
        L.append(row(f"@{k}  head / dist", lambda x, k=k: f"{f4(x['H'][k])} / {f4(x['D'][k])}"))
    L += ["", "## Owner recall@k among available hits: head / distance", "", hdr, sep]
    for i, k in enumerate(KS):
        L.append(row(f"@{k}", lambda x, i=i: f"{f4(x['recall_head'][i])} / {f4(x['recall_dist'][i])}"))
    L += ["", "## Misses: where the owner is (share of misses)", "", hdr, sep]
    for i, b in enumerate(RANK_BINS):
        L.append(row(f"head rank {bin_label(*b)}", lambda x, i=i: f4(x["miss_rank_share"][i])))
    L.append(row("nearest was own (head demoted it)", lambda x: f4(x["demoted_share"])))
    L += ["", "## Confidence", "", hdr, sep,
          row("margin < 0.1 among misses", lambda x: f4(x["low_margin_miss"])),
          row("margin < 0.1 among correct", lambda x: f4(x["low_margin_ok"])),
          "", "## Error type of misses", "", hdr, sep,
          row("different class", lambda x: f4(x["err"]["diff_class"])),
          row("same class", lambda x: f4(x["err"]["same_class"])),
          row("spurious candidate", lambda x: f4(x["err"]["spurious"])),
          "", "## Context", "", hdr, sep,
          row("available / population", lambda x: f4(x["avail_of_pop"])),
          row("correct / available", lambda x: f4(x["correct_of_avail"])),
          "| events | " + " | ".join(str(res["n_events"][h]) for h in HEADS) + " |"]
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(a.rank_root, a.mb_root)
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
    s.add_argument("--pred-class", action="store_true",
                   help="also split different-class misses by the trunk's predicted classes (R6)")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("report")
    s.add_argument("--rank-root", required=True, help="dir with one subdir per head tag")
    s.add_argument("--mb-root", required=True, help="norm_ep29 dir (for the R2 cross-check)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
