#!/usr/bin/env python3
"""Stage 1a-R6 follow-up (diagnostic only): does the trunk's predicted-class disagreement between a hit and the
candidate it picked separate WRONG links from RIGHT ones?

Reads `ranking_diag run --pred-class` summaries (one dir per head) and reports, per head:
  P(dis | correct pick), P(dis | miss), their ratio LR, the rate by miss type and by the hit's true class,
  and what a veto "reject links whose predicted classes disagree" would cost and catch.
Interpretation bands are fixed here, before any number was seen (experiments/particle_hierarchy/pred_class.md):
  LR < 1.5                                  weak signal - consistent with R6 (the head already has it)
  LR >= 3 and P(dis | correct) <= 0.10      strong signal the head is NOT using - a genuine contradiction of R6
  otherwise                                 partial
Pure python.
"""
import argparse
import json
import math
from pathlib import Path

CLASSES = ("pion", "muon", "kaon", "hadron", "shower", "michel", "diffuse")
LR_WEAK, LR_STRONG, CORRECT_MAX = 1.5, 3.0, 0.10


def r(a, b):
    return a / b if b else math.nan


def head(st):
    n_ok, n_miss = st["n_correct"], st["n_miss"]
    p_ok, p_miss = r(st["ok_pred_diff"], n_ok), r(st["miss_pred_diff"], n_miss)
    lr = r(p_miss, p_ok)
    flagged = st["ok_pred_diff"] + st["miss_pred_diff"]
    out = {"p_dis_correct": p_ok, "p_dis_miss": p_miss, "LR": lr,
           "p_dis_diff_class": r(st["err_diff_pred_diff"], st["err_diff_class"]),
           "p_dis_same_class": r(st["err_same_pred_diff"], st["err_same_class"]),
           "p_dis_spurious": r(st["err_spurious_pred_diff"], st["err_spurious"]),
           "veto_precision": r(st["miss_pred_diff"], flagged),
           "veto_correct_lost": p_ok, "veto_misses_caught": p_miss,
           "hit_mispredicted_correct": r(st["ok_hit_mispredicted"], n_ok),
           "hit_mispredicted_miss": r(st["miss_hit_mispredicted"], n_miss),
           "dis_correct_from_hit_error": r(st["ok_dis_hit_mispredicted"], st["ok_pred_diff"]),
           "by_class": {c: {"n_correct": st["ok_by_cls"][i], "n_miss": st["miss_by_cls"][i],
                            "p_dis_correct": r(st["ok_dis_by_cls"][i], st["ok_by_cls"][i]),
                            "p_dis_miss": r(st["miss_dis_by_cls"][i], st["miss_by_cls"][i])}
                        for i, c in enumerate(CLASSES)}}
    if lr < LR_WEAK:
        out["reading"] = "WEAK"
    elif lr >= LR_STRONG and p_ok <= CORRECT_MAX:
        out["reading"] = "STRONG-UNUSED"
    else:
        out["reading"] = "PARTIAL"
    return out


def evaluate(root):
    res = {}
    for d in sorted(p for p in Path(root).iterdir() if (p / "rank_summary.json").exists()):
        st = json.loads((d / "rank_summary.json").read_text())["stats"]
        if "ok_pred_diff" not in st:
            raise SystemExit(f"{d}: summary lacks the disagreement counters (run ranking_diag --pred-class)")
        res[d.name] = head(st)
    if not res:
        raise SystemExit(f"no rank summaries under {root}")
    contra = [h for h, v in res.items() if v["reading"] == "STRONG-UNUSED" and not h.startswith("ORACLE")]
    return {"heads": res, "contradiction_with_R6": bool(contra), "contradicting_heads": contra}


def f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def render(res):
    H = list(res["heads"])
    hdr, sep = "| | " + " | ".join(H) + " |", "|---" * (len(H) + 1) + "|"
    row = lambda lab, fn: f"| {lab} | " + " | ".join(fn(res["heads"][h]) for h in H) + " |"
    L = ["# Predicted-class disagreement on correct vs wrong picks (diagnostic, dev-A)", "",
         f"**Contradiction with R6:** {'YES — ' + ', '.join(res['contradicting_heads']) if res['contradiction_with_R6'] else 'no'}"
         " (bands fixed before the run: LR < 1.5 weak; LR ≥ 3 with P(dis | correct) ≤ 0.10 = strong unused signal)", "",
         hdr, sep,
         row("reading", lambda v: v["reading"]),
         row("P(disagree | correct pick)", lambda v: f(v["p_dis_correct"])),
         row("P(disagree | miss)", lambda v: f(v["p_dis_miss"])),
         row("LR = ratio", lambda v: f(v["LR"])),
         row("  … different-class misses", lambda v: f(v["p_dis_diff_class"])),
         row("  … same-class misses", lambda v: f(v["p_dis_same_class"])),
         row("  … spurious misses", lambda v: f(v["p_dis_spurious"])),
         row("veto precision (share of flagged links that are wrong)", lambda v: f(v["veto_precision"])),
         row("hit mispredicted: correct picks", lambda v: f(v["hit_mispredicted_correct"])),
         row("hit mispredicted: misses", lambda v: f(v["hit_mispredicted_miss"])),
         row("disagreements on correct picks caused by a mispredicted hit",
             lambda v: f(v["dis_correct_from_hit_error"])),
         "", "## By the hit's true class: P(disagree | correct) / P(disagree | miss)  [n correct / n miss]", "",
         hdr, sep]
    for c in CLASSES[:-1]:
        L.append(row(c, lambda v, c=c: f"{f(v['by_class'][c]['p_dis_correct'])} / {f(v['by_class'][c]['p_dis_miss'])} "
                                      f"[{v['by_class'][c]['n_correct']} / {v['by_class'][c]['n_miss']}]"))
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, indent=2))
    (out / "report.md").write_text(render(res))
    print(render(res))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("report")
    s.add_argument("--root", required=True, help="dir with one ranking_diag --pred-class output per head")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
