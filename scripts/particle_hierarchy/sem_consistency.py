#!/usr/bin/env python3
"""Stage 1a-R7: within-particle semantic consistency of the trunk.

Pre-registration: experiments/particle_hierarchy/sem_consistency.md (written before any run).

Library (used by finetune.py):
  sem_consistency_loss  L_cons = hit-weighted mean over labelled hits of particles with >= 2 labelled hits of
                        KL( sg(mean predicted distribution of the hit's particle) || predicted distribution of the hit )
  pair_counts_torch     within-particle predicted-class disagreement (pair-weighted) + accuracy, for monitoring

Commands:
  audit          (EAF, torch) one dev-A cache -> semantic accuracy, within-particle disagreement (pair-weighted and
                 particle-mean), seed-hit vs other-hit agreement with the particle consensus, by true class
  audit-report   (pure python) Step 0: ep29, T_s1/2, C_s1/2 audits -> GO / STOP (+ overlap flag)
  pilot-report   (pure python) the exploratory pilot: manipulation check, health, head-level readouts
"""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

N_CLS = 7
CLASSES = ("pion", "muon", "kaon", "hadron", "shower", "michel", "diffuse")
# Step 0 rule, fixed before any audit ran
D_MIN_GO = 0.10            # ep29 pair-weighted within-particle disagreement needed for room to improve
OVERLAP_REL = 0.25         # T reduces disagreement vs C by >= 25 % on both seeds -> Stage-1b already does this
# pilot bands, fixed before the run
MANIP_REL = 0.25           # SEM must cut pair disagreement by >= 25 % vs C
ACC_TOL = 0.01             # ... without losing more than 0.01 semantic accuracy
HEALTH_TOL = 0.01          # dev semantic / event recall at the last epoch vs C
GAIN_PROMISING, GAIN_FUTILE = 0.03, 0.01


# --------------------------------------------------------------------------- #
# torch: loss and monitoring
# --------------------------------------------------------------------------- #
def _particle_index(y_i, y_sem, batch_vec):
    import torch
    lab = (y_sem >= 0) & (y_i >= 0)
    if not bool(lab.any()):
        return lab, None
    key = batch_vec[lab].long() * (int(y_i[lab].max()) + 1) + y_i[lab].long()
    _, idx = torch.unique(key, return_inverse=True)
    return lab, idx


def sem_consistency_loss(x_sem, y_i, y_sem, batch_vec):
    """KL( sg(particle-mean p) || p_i ), hit-weighted over labelled hits of particles with >= 2 labelled hits."""
    import torch
    lab, idx = _particle_index(y_i, y_sem, batch_vec)
    if idx is None:
        return x_sem.sum() * 0.0
    logp = torch.log_softmax(x_sem[lab].float(), dim=1)
    p = logp.exp()
    n_p = int(idx.max()) + 1
    cnt = torch.zeros(n_p, device=p.device).index_add_(0, idx, torch.ones_like(idx, dtype=p.dtype))
    pbar = torch.zeros(n_p, p.size(1), device=p.device).index_add_(0, idx, p) / cnt[:, None]
    pbar = pbar.detach()[idx]
    kl = (pbar * (pbar.clamp_min(1e-12).log() - logp)).sum(dim=1)
    keep = cnt[idx] >= 2
    return kl[keep].mean() if bool(keep.any()) else x_sem.sum() * 0.0


def pair_counts_torch(pred, y_i, y_sem, batch_vec, n_cls=N_CLS):
    """(disagreeing pairs, total pairs, labelled hits, correctly predicted labelled hits) — pair-weighted."""
    import torch
    lab, idx = _particle_index(y_i, y_sem, batch_vec)
    if idx is None:
        return 0, 0, 0, 0
    pc = pred[lab].long().clamp(0, n_cls - 1)
    n_p = int(idx.max()) + 1
    counts = torch.zeros(n_p * n_cls, device=pred.device, dtype=torch.float64)
    counts.index_add_(0, idx * n_cls + pc, torch.ones_like(pc, dtype=torch.float64))
    counts = counts.view(n_p, n_cls)
    n = counts.sum(1)
    tot = float((n * (n - 1)).sum())
    same = float((counts * (counts - 1)).sum())
    return tot - same, tot, int(lab.sum()), int((pc == y_sem[lab].long()).sum())


# --------------------------------------------------------------------------- #
# numpy: audit maths (testable without torch)
# --------------------------------------------------------------------------- #
def audit_event(pid, pcls, tcls, seed_mask, n_cls=N_CLS):
    """Per-event counts. pid: particle id per labelled hit; pcls / tcls: predicted / true class; seed_mask: the hit
    is a candidate seed. Returns pair counts, accuracy counts, consensus agreement for seed vs other hits, by class."""
    import numpy as np
    out = {"pairs_dis": 0.0, "pairs_tot": 0.0, "n": int(pid.size), "n_ok": int((pcls == tcls).sum()),
           "part_dis_sum": 0.0, "n_part2": 0, "seed_n": 0, "seed_cons": 0, "other_n": 0, "other_cons": 0,
           "seed_ok": 0, "cls_n": [0] * n_cls, "cls_ok": [0] * n_cls,
           "cls_pairs_dis": [0.0] * n_cls, "cls_pairs_tot": [0.0] * n_cls}
    if pid.size == 0:
        return out
    uniq, inv = np.unique(pid, return_inverse=True)
    counts = np.zeros((uniq.size, n_cls))
    np.add.at(counts, (inv, np.clip(pcls, 0, n_cls - 1)), 1)
    n = counts.sum(1)
    tot = n * (n - 1)
    same = (counts * (counts - 1)).sum(1)
    out["pairs_dis"], out["pairs_tot"] = float((tot - same).sum()), float(tot.sum())
    two = n >= 2
    out["part_dis_sum"] = float(((tot - same)[two] / tot[two]).sum())
    out["n_part2"] = int(two.sum())
    consensus = counts.argmax(1)[inv]                                # the particle's majority predicted class
    agree = pcls == consensus
    in2 = two[inv]
    s, o = seed_mask & in2, ~seed_mask & in2
    out.update(seed_n=int(s.sum()), seed_cons=int((agree & s).sum()), other_n=int(o.sum()),
               other_cons=int((agree & o).sum()), seed_ok=int(((pcls == tcls) & s).sum()))
    tc = np.zeros((uniq.size, n_cls))                               # particle's true class = majority of its hits
    np.add.at(tc, (inv, np.clip(tcls, 0, n_cls - 1)), 1)
    pcl_true = tc.argmax(1)
    for c in range(n_cls):
        m = tcls == c
        out["cls_n"][c] = int(m.sum())
        out["cls_ok"][c] = int((pcls[m] == c).sum())
        pm = pcl_true == c
        out["cls_pairs_dis"][c] = float((tot - same)[pm].sum())
        out["cls_pairs_tot"][c] = float(tot[pm].sum())
    return out


def merge(total, ev):
    for k, v in ev.items():
        total[k] = [a + b for a, b in zip(total.get(k, [0] * len(v)), v)] if isinstance(v, list) else total.get(k, 0) + v
    return total


def summarise(t):
    r = lambda a, b: a / b if b else math.nan
    return {"acc": r(t["n_ok"], t["n"]), "pair_disagree": r(t["pairs_dis"], t["pairs_tot"]),
            "particle_mean_disagree": r(t["part_dis_sum"], t["n_part2"]),
            "seed_consensus_agree": r(t["seed_cons"], t["seed_n"]),
            "other_consensus_agree": r(t["other_cons"], t["other_n"]), "seed_acc": r(t["seed_ok"], t["seed_n"]),
            "n_hits": t["n"], "n_seed": t["seed_n"],
            "by_class": {CLASSES[c]: {"n": t["cls_n"][c], "acc": r(t["cls_ok"][c], t["cls_n"][c]),
                                      "pair_disagree": r(t["cls_pairs_dis"][c], t["cls_pairs_tot"][c])}
                         for c in range(N_CLS - 1)}}


# --------------------------------------------------------------------------- #
# audit run (torch)
# --------------------------------------------------------------------------- #
def cmd_audit(a):
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("soft_particle", Path(__file__).with_name("soft_particle.py"))
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    mat = sp._import_materialize()
    dev = torch.device("cpu")
    readout = sp.semantic_readout(a.cache)
    rule_fn = sp.s0.make_rule(sp.DEFAULT_RULE)
    lo, hi = sp.mb.parse_range(a.batches)
    total, n_fg, n_diffuse, n_ev = {}, 0, 0, 0
    for ev in sp.mb.iter_cache(a.cache, lo, hi):
        fg = ev["mask"].bool()
        if not bool(fg.any()):
            continue
        pred = sp.predicted_logits(ev["hit_x_fg"], readout).argmax(dim=1)
        n_fg += int(fg.sum()); n_diffuse += int((pred == sp.DIFFUSE).sum())
        x = sp.infer_inputs(ev, mat, rule_fn, 20, 0.3, float("inf"), dev, need_hit_x=False)
        seed = torch.zeros(int(fg.sum()), dtype=torch.bool)
        if x is not None:
            seed[x["cp"].cpu()] = True
        ys, yi = ev["y_sem"].long()[fg], ev["y_i"].long()[fg]
        lab = (ys >= 0) & (yi >= 0)
        merge(total, audit_event(yi[lab].numpy(), pred[lab].numpy(), ys[lab].numpy(), seed[lab].numpy()))
        n_ev += 1
    res = {"cache": a.cache, "batches": a.batches, "checkpoint_sha256": readout[2], "n_events": n_ev,
           "fg_reclassified_diffuse": n_diffuse / max(n_fg, 1), "totals": total, **summarise(total)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("totals", "by_class")}, indent=2))


# --------------------------------------------------------------------------- #
# Step 0 report (pure python)
# --------------------------------------------------------------------------- #
TRUNKS = ("ep29", "T_s1", "T_s2", "C_s1", "C_s2")


def audit_decision(aud):
    d = aud["ep29"]["pair_disagree"]
    rel = {t: 1 - aud[f"T_s{t}"]["pair_disagree"] / aud[f"C_s{t}"]["pair_disagree"] for t in (1, 2)}
    overlap = all(v >= OVERLAP_REL for v in rel.values())
    go = d >= D_MIN_GO
    return {"go": go, "reading": "GO" if go else "STOP", "ep29_pair_disagree": d,
            "T_vs_C_relative_reduction": rel, "overlap_with_stage1b": overlap,
            "next": ("run the SEM pilot" if go else "within-particle inconsistency too small to be the lever: "
                     "do not train SEM") + ("; FLAG: Stage-1b T already reduces inconsistency by >= 25 %"
                                             if overlap else "")}


def cmd_audit_report(a):
    aud = {t: json.loads((Path(a.root) / f"{t}.json").read_text()) for t in TRUNKS}
    dec = audit_decision(aud)
    f = lambda x: f"{x:.4f}" if isinstance(x, float) and math.isfinite(x) else str(x)
    L = ["# Stage 1a-R7 Step 0 — semantic audit of existing trunks (dev-A)", "",
         f"**Decision:** {dec['reading']} — {dec['next']}", "",
         "| | " + " | ".join(TRUNKS) + " |", "|---" * (len(TRUNKS) + 1) + "|"]
    for k, lab in (("acc", "semantic accuracy (labelled fg hits)"), ("pair_disagree", "within-particle disagreement (pair-weighted)"),
                   ("particle_mean_disagree", "within-particle disagreement (particle mean)"),
                   ("seed_acc", "seed-hit accuracy"), ("seed_consensus_agree", "seed hits agreeing with particle consensus"),
                   ("other_consensus_agree", "other hits agreeing with particle consensus"),
                   ("fg_reclassified_diffuse", "read-out fidelity (fg re-classified diffuse)")):
        L.append(f"| {lab} | " + " | ".join(f(aud[t][k]) for t in TRUNKS) + " |")
    for c in CLASSES[:-1]:
        L.append(f"| {c}: acc / pair disagreement | " + " | ".join(
            f"{f(aud[t]['by_class'][c]['acc'])} / {f(aud[t]['by_class'][c]['pair_disagree'])}" for t in TRUNKS) + " |")
    L += ["", f"T vs C relative reduction of pair disagreement: seed 1 {f(dec['T_vs_C_relative_reduction'][1])}, "
              f"seed 2 {f(dec['T_vs_C_relative_reduction'][2])}"]
    md = "\n".join(L) + "\n"
    (Path(a.root) / "step0_report.md").write_text(md)
    (Path(a.root) / "step0_decision.json").write_text(json.dumps(dec, indent=2))
    print(md)
    if not dec["go"]:
        raise SystemExit(3)                                          # the pipeline stops here


# --------------------------------------------------------------------------- #
# pilot report (pure python)
# --------------------------------------------------------------------------- #
def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def last_row(hist):
    rows = [r for r in csv.DictReader(open(hist)) if r.get("epoch") not in (None, "", "-1")]
    return max(rows, key=lambda r: int(r["epoch"]))


def paired_csv(a_path, b_path, col, variant=None):
    """Mean ± SE of b − a per event (keys batch, event), optional variant filter."""
    def load(p):
        out = {}
        for r in csv.DictReader(open(p)):
            if variant is not None and r.get("variant") != variant:
                continue
            v = fnum(r.get(col))
            if math.isfinite(v):
                out[(r["batch"], r["event"])] = v
        return out
    A, B = load(a_path), load(b_path)
    d = [B[k] - A[k] for k in set(A) & set(B)]
    if len(d) < 2:
        return {"mean": math.nan, "se": math.nan, "n": len(d)}
    return {"mean": st.mean(d), "se": st.stdev(d) / math.sqrt(len(d)), "n": len(d)}


def pilot(root, step0):
    R, S0 = Path(root), Path(step0)
    audC, audT = json.loads((S0 / "C_s1.json").read_text()), json.loads((S0 / "T_s1.json").read_text())
    audS = json.loads((R / "audit_SEM_s1.json").read_text())
    manip = {"pair_disagree": {"C": audC["pair_disagree"], "T": audT["pair_disagree"], "SEM": audS["pair_disagree"]},
             "acc": {"C": audC["acc"], "T": audT["acc"], "SEM": audS["acc"]}}
    manip["relative_reduction_vs_C"] = 1 - audS["pair_disagree"] / audC["pair_disagree"]
    manip["pass"] = (manip["relative_reduction_vs_C"] >= MANIP_REL and audS["acc"] >= audC["acc"] - ACC_TOL)
    hS, hC = last_row(R / "trunks/SEM_s1/history.csv"), last_row(R / "ref/C_s1_history.csv")
    health = {k: {"SEM": fnum(hS.get(k)), "C": fnum(hC.get(k))}
              for k in ("dev_ng/semantic/recall-val", "dev_ng/event/recall-val", "dev_ng/instance/adjusted-rand-val")}
    health["pass"] = all(health[k]["SEM"] >= health[k]["C"] - HEALTH_TOL
                         for k in ("dev_ng/semantic/recall-val", "dev_ng/event/recall-val"))
    ev = lambda d: R / d
    joint = {m: paired_csv(ev("ref/C_s1_dev_events_ep4.csv"), ev("trunks/SEM_s1/dev_events_ep4.csv"), m)
             for m in ("ari_soft", "ari_hard")}
    joint_T = {m: paired_csv(ev("ref/C_s1_dev_events_ep4.csv"), ev("ref/T_s1_dev_events_ep4.csv"), m)
               for m in ("ari_soft", "ari_hard")}
    heads = {}
    for arm in ("SEM", "T"):
        heads[arm] = paired_csv(ev("diag/C_s1_m20/diagnose_events.csv"), ev(f"diag/{arm}_s1_m20/diagnose_events.csv"),
                                "ari_particle", variant="v2_nonull")
    mech = {}
    for arm in ("C", "T", "SEM"):
        p = ev(f"rank/{arm}_s1_m20/rank_summary.json")
        if p.exists():
            stt = json.loads(p.read_text())["stats"]
            mech[arm] = {"diff_class": stt["err_diff_class"] / max(stt["n_miss"], 1),
                         "own_pick": stt["n_correct"] / max(stt["n_pop"], 1),
                         "p_dis_correct": stt.get("ok_pred_diff", 0) / max(stt["n_correct"], 1)}
    g = heads["SEM"]["mean"]
    if not manip["pass"] or not health["pass"]:
        reading = "FALLBACK" if not health["pass"] else "MANIPULATION-FAILED"
    elif g >= GAIN_PROMISING:
        reading = "PROMISING"
    elif g < GAIN_FUTILE:
        reading = "FUTILE"
    else:
        reading = "WEAK-POSITIVE"
    nxt = {"PROMISING": "launch the formal matched C_s1 at the same commit (then seed 2)",
           "WEAK-POSITIVE": "launch the formal matched C_s1 (the formal test decides)",
           "FUTILE": "stop / deprioritise SEM: consistency changed, the head did not gain",
           "MANIPULATION-FAILED": "one fallback: lambda_cons = 4, then stop",
           "FALLBACK": "health gate failed: one fallback lambda_cons = 0.25, then stop"}[reading]
    return {"manipulation": manip, "health": health, "joint_soft_layer_M5": joint, "joint_T_reference": joint_T,
            "head_M20_vs_C": heads, "mechanism": mech, "reading": reading, "next": nxt,
            "exploratory": True,
            "note": "controls are the Stage-1b C_s1/T_s1 trunks (older commit): exploratory only, not the formal test"}


def cmd_pilot_report(a):
    res = pilot(a.root, a.step0)
    f = lambda x: f"{x:+.4f}" if isinstance(x, float) and math.isfinite(x) else "n/a"
    pm = lambda d: f"{f(d['mean'])} ± {abs(d['se']):.4f}" if math.isfinite(d["se"]) else "n/a"
    m, h = res["manipulation"], res["health"]
    L = ["# Stage 1a-R7 SEM pilot (EXPLORATORY; controls = Stage-1b C_s1 / T_s1 at an older commit)", "",
         f"**Reading:** {res['reading']} — next: {res['next']}", "",
         "| | C_s1 | T_s1 | SEM_s1 |", "|---|---|---|---|",
         f"| within-particle pair disagreement | {m['pair_disagree']['C']:.4f} | {m['pair_disagree']['T']:.4f} | {m['pair_disagree']['SEM']:.4f} |",
         f"| semantic accuracy | {m['acc']['C']:.4f} | {m['acc']['T']:.4f} | {m['acc']['SEM']:.4f} |",
         "", f"Manipulation: SEM cuts disagreement by {m['relative_reduction_vs_C']:.1%} vs C (needs ≥ {MANIP_REL:.0%}) → "
             f"{'PASS' if m['pass'] else 'FAIL'}; health → {'PASS' if h['pass'] else 'FAIL'}", "",
         "| health, last epoch | C_s1 | SEM_s1 |", "|---|---|---|"]
    for k in ("dev_ng/semantic/recall-val", "dev_ng/event/recall-val", "dev_ng/instance/adjusted-rand-val"):
        L.append(f"| {k} | {h[k]['C']:.4f} | {h[k]['SEM']:.4f} |")
    L += ["", "| paired vs C_s1 (dev-A) | SEM | T (Stage-1b reference) |", "|---|---|---|",
          f"| joint M=5 soft layer, epoch 4 | {pm(res['joint_soft_layer_M5']['ari_soft'])} | {pm(res['joint_T_reference']['ari_soft'])} |",
          f"| hard seed decoder, epoch 4 | {pm(res['joint_soft_layer_M5']['ari_hard'])} | {pm(res['joint_T_reference']['ari_hard'])} |",
          f"| fresh M=20 head (last epoch) | {pm(res['head_M20_vs_C']['SEM'])} | {pm(res['head_M20_vs_C']['T'])} |"]
    if res["mechanism"]:
        L += ["", "| M=20 head mechanism | " + " | ".join(res["mechanism"]) + " |", "|---" * (len(res["mechanism"]) + 1) + "|"]
        for k, lab in (("diff_class", "different-class share of misses"), ("own_pick", "absolute own-pick"),
                       ("p_dis_correct", "P(pred-class disagreement | correct pick)")):
            L.append(f"| {lab} | " + " | ".join(f"{res['mechanism'][x][k]:.4f}" for x in res["mechanism"]) + " |")
    md = "\n".join(L) + "\n"
    out = Path(a.root) / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_report.md").write_text(md)
    (out / "pilot_report.json").write_text(json.dumps(res, indent=2, default=str))
    print(md)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit")
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:150")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("audit-report")
    s.add_argument("--root", required=True, help="dir with ep29.json, T_s1.json, T_s2.json, C_s1.json, C_s2.json")
    s.set_defaults(fn=cmd_audit_report)
    s = sub.add_parser("pilot-report")
    s.add_argument("--root", required=True)
    s.add_argument("--step0", required=True)
    s.set_defaults(fn=cmd_pilot_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
