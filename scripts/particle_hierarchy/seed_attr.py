#!/usr/bin/env python3
"""Stage 1a-R7 Step 1: are the head's class errors seed-side, hit-side, or neither? (diagnostic only)

Reads `ranking_diag run --pred-class` summaries (one dir per head) and splits each head's DIFFERENT-CLASS misses by
which predictions the trunk got wrong:
  cand_disguised   hit predicted correctly; the picked candidate's seed hit predicted AS the hit's class
  hit_disguised    the hit predicted AS the candidate's class; the seed predicted correctly
  both_correct     both predicted correctly (the trunk separates them), the head linked them anyway
  (both_wrong, seed_wrong_third, hit_wrong_third reported only)
and compares how often the CORRECT candidate's seed is mispredicted on misses vs on correct picks (LR_seed).

Readings (fixed here before any number, experiments/particle_hierarchy/seed_attr.md); decided on the CTRL heads, which
must agree on both seeds, ORACLE / PRED heads reported for reference:
  SEED            the largest of the three shares is cand_disguised, >= 0.40, and LR_seed >= 1.5
  HIT             the largest is hit_disguised, >= 0.40
  HEAD/GEOMETRY   the largest is both_correct, >= 0.40
  MIXED           otherwise, or the two CTRL seeds disagree
Pure python.
"""
import argparse
import json
import math
from pathlib import Path

CLASSES = ("pion", "muon", "kaon", "hadron", "shower", "michel", "diffuse")
MAIN = ("cand_disguised", "hit_disguised", "both_correct")
OTHER = ("both_wrong", "seed_wrong_third", "hit_wrong_third")
SHARE_MIN, LR_MIN = 0.40, 1.5
NEXT = {"SEED": "Step 2: ORACLE-SEED head test (true class on seed hits only) before any seed-specific trunk work",
        "HIT": "per-hit semantic quality of the hit (hadronic classes) is the lever: design a class-targeted "
               "objective; no seed-specific work",
        "HEAD/GEOMETRY": "class is available but not the operative factor in these links: deprioritise trunk-semantic work",
        "MIXED": "no single failure mode: no trunk campaign; decide jointly"}


def r(a, b):
    return a / b if b else math.nan


def head(st):
    n = sum(st[f"dm_{k}"] for k in MAIN + OTHER)
    shares = {k: r(st[f"dm_{k}"], n) for k in MAIN + OTHER}
    p_miss = r(st["miss_owner_seed_wrong"], st["n_miss"])
    p_ok = r(st["ok_owner_seed_wrong"], st["n_correct"])
    lr = r(p_miss, p_ok)
    top = max(MAIN, key=lambda k: shares[k])
    if shares[top] < SHARE_MIN:
        reading = "MIXED"
    elif top == "cand_disguised":
        reading = "SEED" if lr >= LR_MIN else "MIXED"
    elif top == "hit_disguised":
        reading = "HIT"
    else:
        reading = "HEAD/GEOMETRY"
    by_cls = {}
    for i, c in enumerate(CLASSES[:-1]):
        m = st["dm_by_cls"][i]
        by_cls[c] = {"n_diff_misses": m, **{k: r(st[f"dm_{k}_by_cls"][i], m) for k in MAIN},
                     "p_owner_seed_wrong_miss": r(st["miss_owner_seed_wrong_by_cls"][i], st["miss_by_cls"][i]),
                     "p_owner_seed_wrong_ok": r(st["ok_owner_seed_wrong_by_cls"][i], st["ok_by_cls"][i])}
    return {"n_diff_misses": n, "shares": shares, "p_owner_seed_wrong_miss": p_miss,
            "p_owner_seed_wrong_ok": p_ok, "LR_seed": lr, "largest": top, "reading": reading, "by_class": by_cls}


def evaluate(root):
    res = {}
    for d in sorted(p for p in Path(root).iterdir() if (p / "rank_summary.json").exists()):
        st = json.loads((d / "rank_summary.json").read_text())["stats"]
        if "dm_cand_disguised" not in st:
            raise SystemExit(f"{d}: summary lacks the Step-1 counters (run ranking_diag --pred-class at the new commit)")
        res[d.name] = head(st)
    ctrl = [h for h in res if h.startswith("CTRL")]
    if len(ctrl) < 2:
        raise SystemExit(f"need both CTRL heads, found {ctrl}")
    readings = {res[h]["reading"] for h in ctrl}
    overall = readings.pop() if len(readings) == 1 else "MIXED"
    return {"heads": res, "decided_on": ctrl, "reading": overall, "next": NEXT[overall]}


def f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def render(res):
    H = list(res["heads"])
    hdr, sep = "| | " + " | ".join(H) + " |", "|---" * (len(H) + 1) + "|"
    row = lambda lab, fn: f"| {lab} | " + " | ".join(fn(res["heads"][h]) for h in H) + " |"
    L = ["# Stage 1a-R7 Step 1 — seed-side vs hit-side vs head/geometry (diagnostic, dev-A)", "",
         f"**Reading (decided on {', '.join(res['decided_on'])}):** {res['reading']} — next: {res['next']}", "",
         "Bands fixed before the run: largest share ≥ 0.40 decides; SEED also needs LR_seed ≥ 1.5; the CTRL seeds "
         "must agree.", "", hdr, sep,
         row("reading", lambda v: v["reading"]),
         row("different-class misses", lambda v: str(v["n_diff_misses"]))]
    for k, lab in (("cand_disguised", "candidate disguised (seed predicted as the hit's class)"),
                   ("hit_disguised", "hit disguised (hit predicted as the candidate's class)"),
                   ("both_correct", "both predicted correctly"), ("both_wrong", "both wrong"),
                   ("seed_wrong_third", "seed wrong, as a third class"), ("hit_wrong_third", "hit wrong, as a third class")):
        L.append(row(f"share: {lab}", lambda v, k=k: f(v["shares"][k])))
    L += [row("P(owner seed mispredicted | miss)", lambda v: f(v["p_owner_seed_wrong_miss"])),
          row("P(owner seed mispredicted | correct pick)", lambda v: f(v["p_owner_seed_wrong_ok"])),
          row("LR_seed", lambda v: f(v["LR_seed"])),
          "", "## By the hit's true class: cand / hit / both-correct shares of different-class misses  [n]", "", hdr, sep]
    for c in CLASSES[:-1]:
        L.append(row(c, lambda v, c=c: " / ".join(f(v["by_class"][c][k]) for k in MAIN)
                     + f" [{v['by_class'][c]['n_diff_misses']}]"))
    L += ["", "## By class: P(owner seed mispredicted) on misses / on correct picks", "", hdr, sep]
    for c in CLASSES[:-1]:
        L.append(row(c, lambda v, c=c: f"{f(v['by_class'][c]['p_owner_seed_wrong_miss'])} / "
                                      f"{f(v['by_class'][c]['p_owner_seed_wrong_ok'])}"))
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
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
