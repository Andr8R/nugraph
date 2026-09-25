#!/usr/bin/env python3
"""Stage 1a-R5 report: class-information probe, ORACLE UPPER BOUND (not a deployable model).

Pre-registration: experiments/particle_hierarchy/class_probe.md (written before this code ran).

Pure python. Reads, under --root:
  heads/{ARM}_s{t}/train_meta.json          (soft_particle train --class-channels zeros|oracle)
  diag_last/{ARM}_s{t}/diagnose_events.csv  (diagnose on head.pt = last epoch; PRIMARY)
  diag_best/{ARM}_s{t}/diagnose_events.csv  (diagnose on head_best.pt; secondary)
  rank/{ARM}_s{t}/rank_summary.json         (ranking_diag run on head.pt; mechanism)
and, for the calibration row, the existing unpaired heads' retrieval/diag_m20_s{t} under --mb-root.
"""
import argparse
import csv
import importlib.util
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ranking_diag", HERE / "ranking_diag.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)

ARMS = ("CTRL", "ORACLE")
MODE = {"CTRL": "zeros", "ORACLE": "oracle"}
SEEDS = (1, 2)
PRIMARY = "v2_nonull"
OUTCOME_KEYS = {"class_channels", "oracle_upper_bound", "best_epoch", "best_dev_ari"}
CHECKED_ELSEWHERE = {"init_sha256", "perm_sha256"}          # C1 / C2
# argv is the only field allowed to differ besides outcomes: commit, dirty flag and materializer hash
# must match (controlled-comparison rule), and are compared by C3 like any recipe field.
PROVENANCE_KEYS = {"argv"}
GAIN_CLASS, GAIN_NULL = 0.03, 0.01


def name(arm, t):
    return f"{arm}_s{t}"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def diag(path, variant):
    """{(batch, event): (ari_particle, ari_hardseed)} for one decode variant."""
    return {(int(r["batch"]), int(r["event"])): (fnum(r["ari_particle"]), fnum(r["ari_hardseed"]))
            for r in csv.DictReader(open(path)) if r["variant"] == variant}


def paired(a, b):
    """Mean ± SE of b − a over events where both are finite."""
    d = [b[k][0] - a[k][0] for k in sorted(set(a) & set(b)) if math.isfinite(a[k][0]) and math.isfinite(b[k][0])]
    if len(d) < 2:
        return {"mean": math.nan, "se": math.nan, "n": len(d)}
    return {"mean": st.mean(d), "se": st.stdev(d) / math.sqrt(len(d)), "n": len(d)}


def gates(root):
    fails, metas = [], {}
    for arm in ARMS:
        for t in SEEDS:
            p = Path(root) / "heads" / name(arm, t) / "train_meta.json"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            m = json.loads(p.read_text())
            metas[(arm, t)] = m
            if m.get("class_channels") != MODE[arm]:
                fails.append(f"C3 {name(arm, t)}: class_channels {m.get('class_channels')} != {MODE[arm]}")
            if m.get("m") != 20 or m.get("seed") != t:
                fails.append(f"C3 {name(arm, t)}: m {m.get('m')} seed {m.get('seed')}")
            if m.get("git_dirty") is not False:
                fails.append(f"C3 {name(arm, t)}: trained from a dirty or unrecorded tree")
    for t in SEEDS:
        c, o = metas[("CTRL", t)], metas[("ORACLE", t)]
        if c.get("init_sha256") is None or c.get("init_sha256") != o.get("init_sha256"):
            fails.append(f"C1 seed {t}: initialisation differs")
        if not c.get("perm_sha256") or c.get("perm_sha256") != o.get("perm_sha256"):
            fails.append(f"C2 seed {t}: training order differs")
        diff = ({k for k in set(c) | set(o) if c.get(k) != o.get(k)}
                - OUTCOME_KEYS - PROVENANCE_KEYS - CHECKED_ELSEWHERE)
        if diff:
            fails.append(f"C3 seed {t}: recipe differs in {sorted(diff)}")
        a = diag(Path(root) / "diag_last" / name("CTRL", t) / "diagnose_events.csv", PRIMARY)
        b = diag(Path(root) / "diag_last" / name("ORACLE", t) / "diagnose_events.csv", PRIMARY)
        if set(a) != set(b):
            fails.append(f"C4 seed {t}: evaluated event sets differ")
        else:
            bad = [k for k in a if not rd_same(a[k][1], b[k][1])]
            if bad:
                fails.append(f"C4 seed {t}: hard-seed ARI differs on {len(bad)} events")
    return fails, metas


def rd_same(x, y, tol=1e-9):
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(x - y) <= tol


def mechanism(root, arm, t):
    s = json.loads((Path(root) / "rank" / name(arm, t) / "rank_summary.json").read_text())
    n = rd.head_numbers(s)
    return {"diff_class": n["err"]["diff_class"], "same_class": n["err"]["same_class"],
            "spurious": n["err"]["spurious"], "own_pick": n["correct_of_avail"] * n["avail_of_pop"],
            "recall1": n["recall_head"][0], "closure2": n["closure2"], "closure3": n["closure3"],
            "pick_only": n["pick_only"]}


def evaluate(root, mb_root):
    fails, metas = gates(root)
    res = {"gates": {"pass": not fails, "failures": fails}, "seeds": {}}
    for t in SEEDS:
        row = {}
        for which, sub in (("last", "diag_last"), ("best", "diag_best")):
            for var in (PRIMARY, "pick_only"):
                a = diag(Path(root) / sub / name("CTRL", t) / "diagnose_events.csv", var)
                b = diag(Path(root) / sub / name("ORACLE", t) / "diagnose_events.csv", var)
                row[f"{which}:{var}"] = paired(a, b)
                row[f"{which}:{var}:ctrl"] = rd._nanmean([v[0] for v in a.values()])
                row[f"{which}:{var}:oracle"] = rd._nanmean([v[0] for v in b.values()])
        row["mech"] = {arm: mechanism(root, arm, t) for arm in ARMS}
        old = Path(mb_root) / "retrieval" / f"diag_m20_s{t}" / "diagnose_events.csv"
        if old.exists():
            row["calibration_ctrl_vs_unpaired"] = paired(
                diag(old, PRIMARY), diag(Path(root) / "diag_best" / name("CTRL", t) / "diagnose_events.csv", PRIMARY))
        row["best_epoch"] = {arm: metas[(arm, t)].get("best_epoch") for arm in ARMS}
        res["seeds"][t] = row
    res["verdict"] = verdict(res) if not fails else {"reading": "INVALID", "next": "a gate failed; no reading"}
    return res


def verdict(res):
    gains = [res["seeds"][t][f"last:{PRIMARY}"]["mean"] for t in SEEDS]
    halved = [res["seeds"][t]["mech"]["ORACLE"]["diff_class"] <= 0.5 * res["seeds"][t]["mech"]["CTRL"]["diff_class"]
              for t in SEEDS]
    flag = None
    if all(g >= GAIN_CLASS for g in gains) and all(halved):
        reading, nxt = "CLASS", ("class information is a real lever: repeat with the trunk's PREDICTED class "
                                 "probabilities (cache extension) to see how much survives without truth")
    elif all(g < GAIN_NULL for g in gains):
        reading, nxt = "NOT-CLASS", "class information is not the missing piece: hit-to-hit context probe"
    else:
        reading, nxt = "PARTIAL", "hit-to-hit context probe next; predicted-class version deferred"
    if all(g >= GAIN_CLASS for g in gains) and not all(halved):
        flag = "ARI gain >= 0.03 while the different-class share did not halve: explain before acting"
    return {"reading": reading, "next": nxt, "gains": gains, "diff_class_halved": halved, "flag": flag}


def f4(x):
    return rd.f4(x)


def render(res):
    V = res["verdict"]
    L = ["# Stage 1a-R5 — class-information probe (ORACLE UPPER BOUND, not deployable; dev-A)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += ["", f"**Reading:** {V['reading']} — next: {V['next']}"]
    if V.get("flag"):
        L.append(f"**Flag:** {V['flag']}")
    L += ["", "## Primary: ORACLE − CTRL, per-event paired (last epoch, deployed decode)", "",
          "| | seed 1 | seed 2 |", "|---|---|---|"]
    for key, lab in ((f"last:{PRIMARY}", "Δ ARI v2_nonull (primary)"), ("last:pick_only", "Δ ARI pick_only"),
                     (f"best:{PRIMARY}", "Δ ARI v2_nonull, best epoch (secondary)")):
        L.append(f"| {lab} | " + " | ".join(
            f"{f4(res['seeds'][t][key]['mean'])} ± {f4(res['seeds'][t][key]['se'])}" for t in SEEDS) + " |")
    L.append("| CTRL / ORACLE ARI (last, v2_nonull) | " + " | ".join(
        f"{f4(res['seeds'][t][f'last:{PRIMARY}:ctrl'])} / {f4(res['seeds'][t][f'last:{PRIMARY}:oracle'])}"
        for t in SEEDS) + " |")
    L += ["", "## Mechanism (ranking_diag on head.pt): CTRL → ORACLE", "", "| | seed 1 | seed 2 |", "|---|---|---|"]
    for k, lab in (("diff_class", "different-class share of misses"), ("same_class", "same-class share"),
                   ("spurious", "spurious share"), ("own_pick", "absolute own-pick"),
                   ("recall1", "owner recall@1"), ("closure2", "closure@2"), ("closure3", "closure@3")):
        L.append(f"| {lab} | " + " | ".join(
            f"{f4(res['seeds'][t]['mech']['CTRL'][k])} → {f4(res['seeds'][t]['mech']['ORACLE'][k])}"
            for t in SEEDS) + " |")
    L += ["", "## Calibration (reported only): CTRL vs the existing unpaired m20 head, best epochs", "",
          "| | seed 1 | seed 2 |", "|---|---|---|",
          "| Δ ARI v2_nonull | " + " | ".join(
              f"{f4(res['seeds'][t]['calibration_ctrl_vs_unpaired']['mean'])} ± "
              f"{f4(res['seeds'][t]['calibration_ctrl_vs_unpaired']['se'])}"
              if "calibration_ctrl_vs_unpaired" in res["seeds"][t] else "n/a" for t in SEEDS) + " |"]
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(a.root, a.mb_root)
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
    s = sub.add_parser("report")
    s.add_argument("--root", required=True)
    s.add_argument("--mb-root", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
