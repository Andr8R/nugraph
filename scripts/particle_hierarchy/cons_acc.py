#!/usr/bin/env python3
"""Stage 1a-R9 report: consistency vs accuracy -- which per-hit semantic property carries the class-truth gain?

Pre-registration: experiments/particle_hierarchy/cons_acc.md (written before this code ran). Head-level diagnostic,
dev-A; every truth-derived arm is an ORACLE UPPER BOUND (CONS uses TRUE particle membership to make the trunk's own
class constant within particles).

Arms (8 one-hot channels, identical architecture / init / order): CTRL, ORACLE (R5/R6 anchor), TRUECLS (matched
denominator), CONS, FIX50, FIXTRACK. Shares C = d_CONS / G_T, A = d_FIX50 / G_T, K = d_FIXTRACK / G_T with
G_T = TRUECLS - CTRL, joint paired bootstrap. Pure python; reads, under --root:
  readout/{train,devA}.json   heads/{ARM}_s{t}/train_meta.json
  diag_last|diag_best/{ARM}_s{t}/diagnose_events.csv   rank/{ARM}_s{t}/rank_summary.json
"""
import argparse
import importlib.util
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("class_probe", HERE / "class_probe.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)
rd = cp.rd

ARMS = ("CTRL", "ORACLE", "TRUECLS", "CONS", "FIX50", "FIXTRACK")
MODE = {"CTRL": "zeros", "ORACLE": "oracle", "TRUECLS": "trueclass", "CONS": "cons", "FIX50": "fix50",
        "FIXTRACK": "fixtrack"}
LABEL_ARMS = ARMS[2:]
SEEDS = (1, 2)
PRIMARY = "v2_nonull"
REPL_TOL = 0.03
N_BOOT, BOOT_SEED = 2000, 20260927
CONS_MIN, ACC_MAX, BOTH_MIN, NULL_MAX = 0.50, 0.30, 0.30, 0.15
FIX50_RANGE = (0.48, 0.50)
NEXT = {"CONSISTENCY": "formal trunk campaign: Stage-1b T + SEM vs T without SEM, both retrained at one commit "
                       "(T is the control), seeds 1/2, M=20 CTRL heads, decision on dev-B; minimum effect from the "
                       "CONS ceiling",
        "ACCURACY": "class-targeted semantic objective, aimed by FIXTRACK / e; no SEM",
        "BOTH": "consistency first (T + SEM vs T), accuracy objective after",
        "NEITHER": "no realistic trunk semantic lever: close the semantic thread; Stage 1c = Stage-1b geometry alone",
        "PARTIAL": "report; no trunk campaign on this evidence",
        "CONFOUNDED": "CONS is more accurate than FIX50: consistency and accuracy are not separated; add an "
                      "accuracy-matched arm before any trunk campaign"}


def name(arm, t):
    return f"{arm}_s{t}"


def diag(root, sub, arm, t, variant=PRIMARY):
    return cp.diag(Path(root) / sub / name(arm, t) / "diagnose_events.csv", variant)


def rank_stats(root, arm, t):
    return json.loads((Path(root) / "rank" / name(arm, t) / "rank_summary.json").read_text())["stats"]


def r(a, b):
    return a / b if b else math.nan


CH_KEYS = ("ch_n", "ch_correct", "trunk_correct", "trunk_err", "ch_fixed", "trunk_err_track", "ch_fixed_track",
           "ch_broken", "ch_pairs", "ch_pairs_same", "trunk_pairs_same")


def construction(st, arm):
    """D6 failures for one label arm's rank summary (empty list = pass)."""
    if any(k not in st for k in CH_KEYS):
        return ["channel counters missing (run ranking_diag --pred-class at the R9 commit)"]
    f = []
    if arm == "TRUECLS" and st["ch_correct"] != st["ch_n"]:
        f.append(f"accuracy {st['ch_correct']}/{st['ch_n']} != 1")
    if arm == "CONS" and st["ch_pairs_same"] != st["ch_pairs"]:
        f.append(f"within-particle disagreement {1 - r(st['ch_pairs_same'], st['ch_pairs']):.2e} != 0")
    if arm == "FIX50" and not FIX50_RANGE[0] <= r(st["ch_fixed"], st["trunk_err"]) <= FIX50_RANGE[1]:
        f.append(f"corrected fraction {r(st['ch_fixed'], st['trunk_err']):.4f} outside {FIX50_RANGE}")
    if arm == "FIXTRACK" and (st["ch_fixed_track"] != st["trunk_err_track"] or st["ch_fixed"] != st["ch_fixed_track"]):
        f.append(f"fixed track {st['ch_fixed_track']}/{st['trunk_err_track']}, fixed total {st['ch_fixed']}")
    if arm in ("TRUECLS", "FIX50", "FIXTRACK") and st["ch_broken"]:
        f.append(f"{st['ch_broken']} correct argmax hits made wrong")
    return f


def gates(root, r6_root):
    fails, metas, ro = [], {}, {}
    for split in ("train", "devA"):
        p = Path(root) / "readout" / f"{split}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        ro[split] = json.loads(p.read_text())
        if not ro[split].get("pass"):
            fails.append(f"D0 {split}: {ro[split].get('fg_reclassified_diffuse')} of foreground re-classified diffuse")
    if ro["train"].get("checkpoint_sha256") != ro["devA"].get("checkpoint_sha256"):
        fails.append("D0: train and dev-A read-outs come from different checkpoints")
    for arm in ARMS:
        for t in SEEDS:
            p = Path(root) / "heads" / name(arm, t) / "train_meta.json"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            m = json.loads(p.read_text())
            metas[(arm, t)] = m
            if m.get("class_channels") != MODE[arm] or m.get("m") != 20 or m.get("seed") != t:
                fails.append(f"D3 {name(arm, t)}: class_channels {m.get('class_channels')} m {m.get('m')} "
                             f"seed {m.get('seed')}")
            if m.get("git_dirty") is not False:
                fails.append(f"D3 {name(arm, t)}: trained from a dirty or unrecorded tree")
            if m.get("latent_checkpoint_sha256") != ro["train"].get("checkpoint_sha256"):
                fails.append(f"D0 {name(arm, t)}: latent checkpoint differs from the read-out checkpoint")
            if arm in LABEL_ARMS:
                fails += [f"D6 {name(arm, t)}: {x}" for x in construction(rank_stats(root, arm, t), arm)]
    for t in SEEDS:
        c = metas[("CTRL", t)]
        for arm in ARMS[1:]:
            o = metas[(arm, t)]
            if not c.get("init_sha256") or c.get("init_sha256") != o.get("init_sha256"):
                fails.append(f"D1 seed {t}: {arm} initialisation differs from CTRL")
            if not c.get("perm_sha256") or c.get("perm_sha256") != o.get("perm_sha256"):
                fails.append(f"D2 seed {t}: {arm} training order differs from CTRL")
            diff = ({k for k in set(c) | set(o) if c.get(k) != o.get(k)}
                    - cp.OUTCOME_KEYS - cp.PROVENANCE_KEYS - cp.CHECKED_ELSEWHERE)
            if diff:
                fails.append(f"D3 seed {t}: {arm} recipe differs from CTRL in {sorted(diff)}")
        runs = {arm: diag(root, "diag_last", arm, t) for arm in ARMS}
        for arm in ARMS[1:]:
            if set(runs[arm]) != set(runs["CTRL"]):
                fails.append(f"D4 seed {t}: {arm} evaluated a different event set")
                continue
            bad = [k for k in runs["CTRL"] if not cp.rd_same(runs["CTRL"][k][1], runs[arm][k][1])]
            if bad:
                fails.append(f"D4 seed {t}: {arm} hard-seed ARI differs on {len(bad)} events")
        new = cp.paired(runs["CTRL"], runs["ORACLE"])["mean"]
        old = cp.paired(cp.diag(Path(r6_root) / "diag_last" / f"CTRL_s{t}" / "diagnose_events.csv", PRIMARY),
                        cp.diag(Path(r6_root) / "diag_last" / f"ORACLE_s{t}" / "diagnose_events.csv", PRIMARY))["mean"]
        if not abs(new - old) <= REPL_TOL:
            fails.append(f"D5 seed {t}: ORACLE - CTRL {new:.4f} does not reproduce R6 {old:.4f}")
    return fails


def decompose(runs, boot=N_BOOT, seed=BOOT_SEED):
    """Arm means minus CTRL, the shares of G_T and the bucket-7 share of G, with a joint paired event bootstrap."""
    keys = [k for k in sorted(set.intersection(*(set(x) for x in runs.values())))
            if all(math.isfinite(runs[a][k][0]) for a in ARMS)]
    d = {a: [runs[a][k][0] - runs["CTRL"][k][0] for k in keys] for a in ARMS[1:]}

    def stats(idx):
        m = {a: sum(d[a][i] for i in idx) / len(idx) for a in d}
        GT, G = m["TRUECLS"], m["ORACLE"]
        return {"G": G, "G_T": GT, "d_cons": m["CONS"], "d_fix50": m["FIX50"], "d_fixtrack": m["FIXTRACK"],
                "C": r(m["CONS"], GT), "A": r(m["FIX50"], GT), "K": r(m["FIXTRACK"], GT),
                "bucket7_share": r(G - GT, G)}

    point = stats(list(range(len(keys))))
    rng = random.Random(seed)
    samples = [stats([rng.randrange(len(keys)) for _ in keys]) for _ in range(boot)]
    ci = {}
    for k in point:
        v = sorted(s[k] for s in samples if math.isfinite(s[k]))
        ci[k] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else [math.nan, math.nan]
    return {"point": point, "ci95": ci, "n": len(keys)}


def channel(st):
    """Descriptive channel properties on the dev-A population (label arms only)."""
    if any(k not in st for k in CH_KEYS):
        return None
    return {"accuracy": r(st["ch_correct"], st["ch_n"]), "trunk_accuracy": r(st["trunk_correct"], st["ch_n"]),
            "fixed_of_errors": r(st["ch_fixed"], st["trunk_err"]), "n_fixed": st["ch_fixed"],
            "n_fixed_track": st["ch_fixed_track"], "broken": st["ch_broken"],
            "pair_disagreement": 1 - r(st["ch_pairs_same"], st["ch_pairs"]),
            "trunk_pair_disagreement": 1 - r(st["trunk_pairs_same"], st["ch_pairs"])}


def coassign(st):
    return {tag: r(st.get(f"coassign_{tag}_same", 0), st.get(f"coassign_{tag}_pairs", 0))
            for tag in ("avail", "unavail")}


def reading_for(p):
    C, A = p["C"], p["A"]
    if C >= CONS_MIN and A < ACC_MAX:
        return "CONSISTENCY"
    if A >= ACC_MAX and C < ACC_MAX:
        return "ACCURACY"
    if C >= BOTH_MIN and A >= BOTH_MIN:
        return "BOTH"
    if C < NULL_MAX and A < NULL_MAX:
        return "NEITHER"
    return "PARTIAL"


def evaluate(root, r6_root):
    fails = gates(root, r6_root)
    res = {"gates": {"pass": not fails, "failures": fails}, "seeds": {}}
    for t in SEEDS:
        runs = {a: diag(root, "diag_last", a, t) for a in ARMS}
        best = {a: diag(root, "diag_best", a, t) for a in ARMS}
        stats = {a: rank_stats(root, a, t) for a in ARMS}
        dec = decompose(runs)
        p = dec["point"]
        ch = {a: channel(stats[a]) for a in ARMS}
        co = {a: coassign(stats[a]) for a in ARMS}
        e = math.nan
        if ch["FIXTRACK"] and ch["FIX50"]:
            e = r(r(p["d_fixtrack"], ch["FIXTRACK"]["n_fixed_track"]), r(p["d_fix50"], ch["FIX50"]["n_fixed"]))
        lift = {tag: co["TRUECLS"][tag] - co["CTRL"][tag] for tag in ("avail", "unavail")}
        res["seeds"][t] = {
            "decomp": dec, "decomp_best_epoch": decompose(best, boot=200)["point"],
            "ari": {a: rd._nanmean([v[0] for v in runs[a].values()]) for a in ARMS},
            "mech": {a: cp.mechanism(root, a, t) for a in ARMS}, "channel": ch, "coassign": co,
            "coassign_lift_truecls": lift, "coassign_supported": lift["unavail"] >= lift["avail"],
            "efficiency_track": e, "reading": reading_for(p),
            "cons_more_accurate": bool(ch["CONS"] and ch["FIX50"] and ch["CONS"]["accuracy"] > ch["FIX50"]["accuracy"])}
    if fails:
        res["verdict"] = {"reading": "INVALID", "next": "a gate failed; no reading"}
        return res
    rs = {res["seeds"][t]["reading"] for t in SEEDS}
    reading = rs.pop() if len(rs) == 1 else "PARTIAL"
    if reading in ("CONSISTENCY", "BOTH") and any(res["seeds"][t]["cons_more_accurate"] for t in SEEDS):
        reading = "CONFOUNDED"
    res["verdict"] = {"reading": reading, "next": NEXT[reading],
                      "coassign_hypothesis": ("SUPPORTED" if all(res["seeds"][t]["coassign_supported"] for t in SEEDS)
                                              else "NOT SUPPORTED")}
    return res


def render(res):
    f4, V, S = rd.f4, res["verdict"], res["seeds"]
    ci = lambda t, k: (f"{f4(S[t]['decomp']['point'][k])} [{f4(S[t]['decomp']['ci95'][k][0])}, "
                       f"{f4(S[t]['decomp']['ci95'][k][1])}]")
    L = ["# Stage 1a-R9 — consistency vs accuracy (head-level diagnostic, dev-A; every truth-derived arm, incl. "
         "CONS, is an ORACLE UPPER BOUND)", "", f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += ["", f"**Reading:** {V['reading']} — {V['next']}"]
    if "coassign_hypothesis" in V:
        L.append(f"**Co-assignment hypothesis (descriptive):** {V['coassign_hypothesis']}")
    L += ["", "## Decomposition (last epoch, paired vs CTRL, 95 % joint bootstrap)", "", "| | seed 1 | seed 2 |",
          "|---|---|---|"]
    for k, lab in (("G", "G = ORACLE − CTRL"), ("G_T", "G_T = TRUECLS − CTRL (denominator)"),
                   ("bucket7_share", "bucket-7 share (ORACLE − TRUECLS) / G"),
                   ("d_cons", "Δ CONS (consistency oracle)"), ("d_fix50", "Δ FIX50 (half the errors fixed)"),
                   ("d_fixtrack", "Δ FIXTRACK (track-class errors fixed)"), ("C", "C = Δ_CONS / G_T"),
                   ("A", "A = Δ_FIX50 / G_T"), ("K", "K = Δ_FIXTRACK / G_T")):
        L.append(f"| {lab} | {ci(1, k)} | {ci(2, k)} |")
    L.append("| efficiency per corrected hit, FIXTRACK vs FIX50 | "
             + " | ".join(f4(S[t]["efficiency_track"]) for t in SEEDS) + " |")
    L.append("| per-seed reading | " + " | ".join(S[t]["reading"] for t in SEEDS) + " |")
    L += ["", "| ARI (last epoch) | " + " | ".join(ARMS) + " |", "|---" * (len(ARMS) + 1) + "|"]
    for t in SEEDS:
        L.append(f"| seed {t} | " + " | ".join(f4(S[t]["ari"][a]) for a in ARMS) + " |")
    L += ["", "## Channels on the dev-A population (what each arm's input actually is)", "",
          "| seed | property | " + " | ".join(LABEL_ARMS) + " |", "|---|---" + "|---" * len(LABEL_ARMS) + "|"]
    for t in SEEDS:
        for k, lab in (("accuracy", "channel accuracy"), ("trunk_accuracy", "trunk argmax accuracy"),
                       ("fixed_of_errors", "trunk errors corrected"),
                       ("pair_disagreement", "within-particle pair disagreement"),
                       ("trunk_pair_disagreement", "trunk pair disagreement"), ("broken", "correct hits made wrong")):
            L.append(f"| {t} | {lab} | " + " | ".join(
                (str(S[t]["channel"][a][k]) if k == "broken" else f4(S[t]["channel"][a][k]))
                if S[t]["channel"][a] else "n/a" for a in LABEL_ARMS) + " |")
    L += ["", "## Mechanism and co-assignment (ranking_diag, CPU)", "",
          "| seed | metric | " + " | ".join(ARMS) + " |", "|---|---" + "|---" * len(ARMS) + "|"]
    for t in SEEDS:
        for k, lab in (("diff_class", "different-class share of misses"), ("own_pick", "absolute own-pick")):
            L.append(f"| {t} | {lab} | " + " | ".join(f4(S[t]["mech"][a][k]) for a in ARMS) + " |")
        for tag, lab in (("avail", "same-particle pairs co-assigned: available hits"),
                         ("unavail", "same-particle pairs co-assigned: unavailable hits")):
            L.append(f"| {t} | {lab} | " + " | ".join(f4(S[t]["coassign"][a][tag]) for a in ARMS) + " |")
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(a.root, a.r6_root)
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
    s.add_argument("--r6-root", required=True, help="Stage 1a-R6 pred_class root (replication gate D5)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
