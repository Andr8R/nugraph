#!/usr/bin/env python3
"""Stage 1a-R10 report: co-membership objective -- objective gap or information gap for unreachable hits?

Pre-registration: experiments/particle_hierarchy/comem.md (written before this code ran). Deployable arms (truth only
in the training loss). Arms CTRL (lambda 0), CM03 (0.3), CM1 (1.0) x head seeds {1, 2, 3}, identical init and order.
Pure python; reads, under --root:
  heads/{ARM}_s{t}/{train_meta.json,train_history.csv}   diag_last|diag_best/{ARM}_s{t}/diagnose_events.csv
  rank/{ARM}_s{t}/rank_summary.json   (ranking_diag with the R10 co_* pair statistics)
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("class_probe", HERE / "class_probe.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)
rd = cp.rd

ARMS = ("CTRL", "CM03", "CM1")
LAMBDA = {"CTRL": 0.0, "CM03": 0.3, "CM1": 1.0}
SEEDS = (1, 2, 3)
PRIMARY = "v2_nonull"
GAIN, NULL = 0.02, 0.01
MANIP = 0.9
CTRL_BAND = (0.22, 0.30)
U_SHARE = 0.50
CATS = ("RR", "RU", "UU")
NEXT = {"OBJECTIVE-GAP": "confirm the co-membership head on a decision split; Stage 1c = Stage-1b T + co-membership "
                         "gradient vs T without it, both retrained at one commit",
        "INFORMATION-GAP": "the trunk must supply the grouping information: trunk pilot Stage-1b T + particle-weighted "
                           "semantic consistency vs T",
        "MANIPULATION-FAILED": "one fallback at lambda = 3, then decide",
        "PARTIAL": "report; no campaign on this evidence"}


def name(arm, t):
    return f"{arm}_s{t}"


def diag(root, sub, arm, t, variant=PRIMARY):
    return cp.diag(Path(root) / sub / name(arm, t) / "diagnose_events.csv", variant)


def last_comem(root, arm, t):
    rows = list(csv.DictReader(open(Path(root) / "heads" / name(arm, t) / "train_history.csv")))
    return float(rows[-1]["comem"]) if rows and rows[-1].get("comem") not in (None, "") else math.nan


def r(a, b):
    return a / b if b else math.nan


def categories(st, lvl):
    """{RR, RU, UU}: {pairs, same_part, same_lab, same_both} from a rank summary (RU = pop - RR - UU)."""
    key = lambda tag, k: f"co_{lvl}_{tag}_{k}"
    if key("pop", "pairs") not in st:
        raise SystemExit("rank summary lacks the R10 co_* statistics (run ranking_diag at the R10 commit)")
    out = {"RR": {}, "UU": {}, "RU": {}}
    for k in ("pairs", "same_part", "same_lab", "same_both"):
        out["RR"][k], out["UU"][k] = st[key("avail", k)], st[key("unavail", k)]
        out["RU"][k] = st[key("pop", k)] - out["RR"][k] - out["UU"][k]
    return out


def rates(c):
    """co-assignment = same-particle pairs in one cluster / same-particle pairs; false = different-particle pairs in
    one cluster / different-particle pairs."""
    return {"coassign": r(c["same_both"], c["same_part"]),
            "false": r(c["same_lab"] - c["same_both"], c["pairs"] - c["same_part"])}


def gates(root):
    fails, metas = [], {}
    for arm in ARMS:
        for t in SEEDS:
            p = Path(root) / "heads" / name(arm, t) / "train_meta.json"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            m = json.loads(p.read_text())
            metas[(arm, t)] = m
            if (m.get("comem") != LAMBDA[arm] or m.get("m") != 20 or m.get("seed") != t
                    or m.get("class_channels", "none") != "none"):
                fails.append(f"E3 {name(arm, t)}: comem {m.get('comem')} m {m.get('m')} seed {m.get('seed')} "
                             f"class_channels {m.get('class_channels')}")
            if m.get("git_dirty") is not False:
                fails.append(f"E3 {name(arm, t)}: trained from a dirty or unrecorded tree")
    allowed = cp.OUTCOME_KEYS | cp.PROVENANCE_KEYS | cp.CHECKED_ELSEWHERE | {"comem"}
    for t in SEEDS:
        c = metas[("CTRL", t)]
        for arm in ARMS[1:]:
            o = metas[(arm, t)]
            if not c.get("init_sha256") or c.get("init_sha256") != o.get("init_sha256"):
                fails.append(f"E1 seed {t}: {arm} initialisation differs from CTRL")
            if not c.get("perm_sha256") or c.get("perm_sha256") != o.get("perm_sha256"):
                fails.append(f"E2 seed {t}: {arm} training order differs from CTRL")
            diff = {k for k in set(c) | set(o) if c.get(k) != o.get(k)} - allowed
            if diff:
                fails.append(f"E3 seed {t}: {arm} recipe differs from CTRL in {sorted(diff)}")
        runs = {arm: diag(root, "diag_last", arm, t) for arm in ARMS}
        for arm in ARMS[1:]:
            if set(runs[arm]) != set(runs["CTRL"]):
                fails.append(f"E4 seed {t}: {arm} evaluated a different event set")
                continue
            bad = [k for k in runs["CTRL"] if not cp.rd_same(runs["CTRL"][k][1], runs[arm][k][1])]
            if bad:
                fails.append(f"E4 seed {t}: {arm} hard-seed ARI differs on {len(bad)} events")
        ari = rd._nanmean([v[0] for v in runs["CTRL"].values()])
        if not CTRL_BAND[0] <= ari <= CTRL_BAND[1]:
            fails.append(f"E5 seed {t}: CTRL ARI {ari:.4f} outside {CTRL_BAND}")
    return fails


def arm_reading(g):
    if not g["manipulated"]:
        return "MANIPULATION-FAILED"
    if all(d >= GAIN for d in g["delta"]):
        return "OBJECTIVE-GAP" if g["ignored_population"] else "OBJECTIVE-GAP (other route)"
    if g["delta_mean"] < NULL:
        return "NULL"
    return "PARTIAL"


def evaluate(root):
    fails = gates(root)
    res = {"gates": {"pass": not fails, "failures": fails}, "seeds": {}, "arms": {}}
    mech = {}
    for t in SEEDS:
        runs = {a: diag(root, "diag_last", a, t) for a in ARMS}
        best = {a: diag(root, "diag_best", a, t) for a in ARMS}
        stats = {a: json.loads((Path(root) / "rank" / name(a, t) / "rank_summary.json").read_text())["stats"]
                 for a in ARMS}
        mech[t] = {a: {lvl: categories(stats[a], lvl) for lvl in ("pick", "dec")} for a in ARMS}
        res["seeds"][t] = {
            "ari": {a: rd._nanmean([v[0] for v in runs[a].values()]) for a in ARMS},
            "delta": {a: cp.paired(runs["CTRL"], runs[a]) for a in ARMS[1:]},
            "delta_best": {a: cp.paired(best["CTRL"], best[a])["mean"] for a in ARMS[1:]},
            "comem_loss": {a: last_comem(root, a, t) for a in ARMS},
            "rates": {a: {lvl: {c: rates(mech[t][a][lvl][c]) for c in CATS} for lvl in ("pick", "dec")}
                      for a in ARMS},
            "mech": {a: cp.mechanism(root, a, t) for a in ARMS}}
    for a in ARMS[1:]:
        S = res["seeds"]
        delta = [S[t]["delta"][a]["mean"] for t in SEEDS]
        gain = {c: sum(mech[t][a]["dec"][c]["same_both"] - mech[t]["CTRL"]["dec"][c]["same_both"] for t in SEEDS)
                for c in CATS}
        total = sum(gain.values())
        u_share = r(gain["RU"] + gain["UU"], total) if total > 0 else math.nan
        uu_up = all(S[t]["rates"][a]["dec"]["UU"]["coassign"] > S[t]["rates"]["CTRL"]["dec"]["UU"]["coassign"]
                    for t in SEEDS)
        manip = all(S[t]["comem_loss"][a] <= MANIP * S[t]["comem_loss"]["CTRL"] for t in SEEDS)
        g = {"lambda": LAMBDA[a], "delta": delta, "delta_mean": sum(delta) / len(delta),
             "coassign_gain_pairs": gain, "unreachable_share": u_share, "uu_rate_up_all_seeds": uu_up,
             "ignored_population": bool(total > 0 and u_share >= U_SHARE and uu_up), "manipulated": manip}
        g["reading"] = arm_reading(g)
        res["arms"][a] = g
    if fails:
        res["verdict"] = {"reading": "INVALID", "next": "a gate failed; no reading"}
        return res
    rs = {a: res["arms"][a]["reading"] for a in ARMS[1:]}
    obj = [a for a in ARMS[1:] if rs[a] == "OBJECTIVE-GAP"]
    if obj:
        pick = max(obj, key=lambda a: res["arms"][a]["delta_mean"])
        v = {"reading": "OBJECTIVE-GAP", "carried_forward": pick}
    elif all(x == "NULL" for x in rs.values()):
        v = {"reading": "INFORMATION-GAP"}
    elif all(x == "MANIPULATION-FAILED" for x in rs.values()):
        v = {"reading": "MANIPULATION-FAILED"}
    else:
        v = {"reading": "PARTIAL"}
    v["next"] = NEXT[v["reading"]]
    v["per_lambda"] = rs
    res["verdict"] = v
    return res


def render(res):
    f4, V, S = rd.f4, res["verdict"], res["seeds"]
    L = ["# Stage 1a-R10 — co-membership objective (deployable head arms; dev-A, last epoch)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += ["", f"**Reading:** {V['reading']}"
          + (f" (carried forward: {V['carried_forward']})" if V.get("carried_forward") else "") + f" — {V['next']}"]
    if "per_lambda" in V:
        L.append("**Per λ:** " + ", ".join(f"{a} (λ {LAMBDA[a]}): {x}" for a, x in V["per_lambda"].items()))
    L += ["", "## Primary: paired Δ ARI vs CTRL (deployed decode v2_nonull, last epoch; ± event SE)", "",
          "| arm | " + " | ".join(f"seed {t}" for t in SEEDS) + " | mean | best-epoch Δ (mean) |",
          "|---" * (len(SEEDS) + 3) + "|"]
    for a in ARMS[1:]:
        cells = [f"{f4(S[t]['delta'][a]['mean'])} ± {f4(S[t]['delta'][a]['se'])}" for t in SEEDS]
        bm = sum(S[t]["delta_best"][a] for t in SEEDS) / len(SEEDS)
        L.append(f"| {a} (λ {LAMBDA[a]}) | " + " | ".join(cells) + f" | {f4(res['arms'][a]['delta_mean'])} | {f4(bm)} |")
    L += ["", "| ARI (last epoch) | " + " | ".join(ARMS) + " |", "|---" * (len(ARMS) + 1) + "|"]
    for t in SEEDS:
        L.append(f"| seed {t} | " + " | ".join(f4(S[t]["ari"][a]) for a in ARMS) + " |")
    L += ["", "## Manipulation: last-epoch training co-membership loss (arm must be ≤ 0.9 × CTRL on every seed)", "",
          "| seed | " + " | ".join(ARMS) + " |", "|---" * (len(ARMS) + 1) + "|"]
    for t in SEEDS:
        L.append(f"| {t} | " + " | ".join(f4(S[t]["comem_loss"][a]) for a in ARMS) + " |")
    L += ["", "## Mechanism: same-particle co-assignment / false co-assignment by pair class "
          "(R = reachable, U = unreachable)", "",
          "| seed | level | class | " + " | ".join(ARMS) + " |", "|---|---|---" + "|---" * len(ARMS) + "|"]
    for t in SEEDS:
        for lvl in ("dec", "pick"):
            for c in CATS:
                L.append(f"| {t} | {lvl} | {c} | " + " | ".join(
                    f"{f4(S[t]['rates'][a][lvl][c]['coassign'])} / {f4(S[t]['rates'][a][lvl][c]['false'])}"
                    for a in ARMS) + " |")
    L += ["", "## Attribution of the co-assignment gain (decode level, pooled over seeds)", "",
          "| arm | Δ co-assigned pairs RR | RU | UU | unreachable share (RU + UU) | UU rate up on every seed | "
          "IGNORED-POPULATION |", "|---|---|---|---|---|---|---|"]
    for a in ARMS[1:]:
        g = res["arms"][a]
        L.append(f"| {a} | " + " | ".join(f"{g['coassign_gain_pairs'][c]:.3e}" for c in CATS)
                 + f" | {f4(g['unreachable_share'])} | {g['uu_rate_up_all_seeds']} | {g['ignored_population']} |")
    L += ["", "## Head mechanism (ranking_diag)", "", "| seed | metric | " + " | ".join(ARMS) + " |",
          "|---|---" + "|---" * len(ARMS) + "|"]
    for t in SEEDS:
        for k, lab in (("own_pick", "absolute own-pick"), ("diff_class", "different-class share of misses"),
                       ("pick_only", "pick-only ARI")):
            L.append(f"| {t} | {lab} | " + " | ".join(f4(S[t]["mech"][a][k]) for a in ARMS) + " |")
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(a.root)
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
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
