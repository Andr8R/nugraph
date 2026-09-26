#!/usr/bin/env python3
"""Stage 1a-R8 report: where does the +0.17 class-truth gain live? (head-level diagnostic, dev-A)

Pre-registration: experiments/particle_hierarchy/class_decomp.md (written before this code ran).

Arms (16 split channels, identical architecture / init / order): CTRL, ORACLE, SEED (truth to the candidate encoder
only), HIT (truth to the hit encoder only), HARD (one-hot of the trunk's argmax: an inductive-bias probe, no new
information). Shares S_seed, S_hit, F_hard of G = ORACLE - CTRL, and the interaction I = G - (d_seed + d_hit) - no
additivity is assumed. Pure python; reads, under --root:
  readout/{train,devA}.json          heads/{ARM}_s{t}/train_meta.json   diag_last|diag_best/{ARM}_s{t}/diagnose_events.csv   rank/{ARM}_s{t}/rank_summary.json
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

ARMS = ("CTRL", "ORACLE", "SEED", "HIT", "HARD")
MODE = {"CTRL": "s_zeros", "ORACLE": "s_oracle", "SEED": "s_oracle_seed", "HIT": "s_oracle_hit", "HARD": "s_pred_hard"}
SEEDS = (1, 2)
PRIMARY = "v2_nonull"
REPL_TOL = 0.03
N_BOOT, BOOT_SEED = 2000, 20260926
SEED_MIN, HIT_MIN, HIT_SEED_MAX, JOINT_MIN = 0.50, 0.50, 0.30, 0.30
HARD_GAIN, HARD_SHARE, HARD_NULL = 0.02, 0.20, 0.01
GEO_SEED, GEO_NEAR = 0.50, 0.70


def name(arm, t):
    return f"{arm}_s{t}"


def diag(root, sub, arm, t, variant=PRIMARY):
    return cp.diag(Path(root) / sub / name(arm, t) / "diagnose_events.csv", variant)


def gates(root, r6_root):
    fails, metas, ro = [], {}, {}
    for split in ("train", "devA"):                     # D0: PRED-HARD's input is the exact trunk read-out
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
            if (m.get("class_channels") != MODE[arm] or m.get("split_channels") != 8 or m.get("m") != 20
                    or m.get("seed") != t):
                fails.append(f"D3 {name(arm, t)}: class_channels {m.get('class_channels')} split "
                             f"{m.get('split_channels')} m {m.get('m')} seed {m.get('seed')}")
            if m.get("git_dirty") is not False:
                fails.append(f"D3 {name(arm, t)}: trained from a dirty or unrecorded tree")
            if m.get("latent_checkpoint_sha256") != ro["train"].get("checkpoint_sha256"):
                fails.append(f"D0 {name(arm, t)}: latent checkpoint differs from the read-out checkpoint")
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
    """Means of each arm minus CTRL, the shares and I/G, with a joint paired event bootstrap."""
    keys = [k for k in sorted(set.intersection(*(set(r) for r in runs.values())))
            if all(math.isfinite(runs[a][k][0]) for a in ARMS)]
    d = {a: [runs[a][k][0] - runs["CTRL"][k][0] for k in keys] for a in ARMS[1:]}

    def stats(idx):
        m = {a: sum(d[a][i] for i in idx) / len(idx) for a in d}
        G = m["ORACLE"]
        r = lambda x: x / G if G else math.nan
        return {"G": G, "d_seed": m["SEED"], "d_hit": m["HIT"], "d_hard": m["HARD"], "S_seed": r(m["SEED"]),
                "S_hit": r(m["HIT"]), "F_hard": r(m["HARD"]), "I_over_G": r(G - m["SEED"] - m["HIT"])}

    point = stats(list(range(len(keys))))
    rng = random.Random(seed)
    samples = [stats([rng.randrange(len(keys)) for _ in keys]) for _ in range(boot)]
    ci = {}
    for k in point:
        v = sorted(s[k] for s in samples if math.isfinite(s[k]))
        ci[k] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else [math.nan, math.nan]
    return {"point": point, "ci95": ci, "n": len(keys)}


CELLS = ("cand_disguised", "hit_disguised", "both_correct", "both_wrong", "seed_wrong_third", "hit_wrong_third")


def cells(root, arm, t):
    """R7 Step-1 cells of the different-class misses and the owner-seed likelihood ratio, per head."""
    st = json.loads((Path(root) / "rank" / name(arm, t) / "rank_summary.json").read_text())["stats"]
    n = sum(st.get(f"dm_{k}", 0) for k in CELLS)
    r = lambda a, b: a / b if b else math.nan
    out = {k: r(st.get(f"dm_{k}", 0), n) for k in CELLS[:3]}
    out["LR_seed"] = r(r(st.get("miss_owner_seed_wrong", 0), st.get("n_miss", 0)),
                       r(st.get("ok_owner_seed_wrong", 0), st.get("n_correct", 0)))
    return out


def geometry(root, t):
    st = json.loads((Path(root) / "rank" / name("CTRL", t) / "rank_summary.json").read_text())["stats"]
    n = st.get("dm_both_correct", 0)
    r = lambda k: st.get(k, 0) / n if n else math.nan
    out = {"n_both_correct": n, "owner_seed_wrong": r("dm_both_correct_owner_seed_wrong"),
           "picked_nearer": r("dm_both_correct_picked_nearer"),
           "nearer_and_owner_seed_ok": r("dm_both_correct_nearer_owner_seed_ok")}
    out["reading"] = ("SEED-EFFECT" if out["owner_seed_wrong"] >= GEO_SEED else
                      "GEOMETRY" if out["nearer_and_owner_seed_ok"] >= GEO_NEAR else "UNRESOLVED")
    return out


def reading_for(p):
    s, h, i = p["S_seed"], p["S_hit"], p["I_over_G"]
    if s >= SEED_MIN and h >= HIT_MIN:
        return "EITHER"
    if s >= SEED_MIN:
        return "SEED"
    if h >= HIT_MIN and s < HIT_SEED_MAX:
        return "HIT"
    if i >= JOINT_MIN:
        return "JOINT"
    return "PARTIAL"


def hard_for(p):
    if p["d_hard"] >= HARD_GAIN and p["F_hard"] >= HARD_SHARE:
        return "DECISIVE"
    if p["d_hard"] < HARD_NULL:
        return "NO-EFFECT"
    return "INCONCLUSIVE"


NEXT = {"SEED": "seed-specific trunk work is justified (ceiling d_seed)",
        "HIT": "per-hit class quality is the lever; no seed work",
        "EITHER": "redundant sides: improve seeds first (the smaller population)",
        "JOINT": "truth is needed on both sides together: no single-sided campaign",
        "PARTIAL": "no trunk campaign; report"}


def evaluate(root, r6_root):
    fails = gates(root, r6_root)
    res = {"gates": {"pass": not fails, "failures": fails}, "seeds": {}}
    for t in SEEDS:
        runs = {a: diag(root, "diag_last", a, t) for a in ARMS}
        best = {a: diag(root, "diag_best", a, t) for a in ARMS}
        dec = decompose(runs)
        res["seeds"][t] = {"decomp": dec, "decomp_best_epoch": decompose(best, boot=200)["point"],
                           "ari": {a: rd._nanmean([v[0] for v in runs[a].values()]) for a in ARMS},
                           "mech": {a: {**cp.mechanism(root, a, t), **cells(root, a, t)} for a in ARMS},
                           "geometry": geometry(root, t),
                           "reading": reading_for(dec["point"]), "hard": hard_for(dec["point"])}
    if fails:
        res["verdict"] = {"reading": "INVALID", "hard": "INVALID", "next": "a gate failed; no reading"}
        return res
    rs = {res["seeds"][t]["reading"] for t in SEEDS}
    hs = {res["seeds"][t]["hard"] for t in SEEDS}
    reading = rs.pop() if len(rs) == 1 else "PARTIAL"
    hard = hs.pop() if len(hs) == 1 else "INCONCLUSIVE"
    geo = {res["seeds"][t]["geometry"]["reading"] for t in SEEDS}
    res["verdict"] = {"reading": reading, "next": NEXT[reading], "hard": hard,
                      "geometry": geo.pop() if len(geo) == 1 else "UNRESOLVED"}
    return res


def render(res):
    f4, V, S = rd.f4, res["verdict"], res["seeds"]
    ci = lambda t, k: (f"{f4(S[t]['decomp']['point'][k])} [{f4(S[t]['decomp']['ci95'][k][0])}, "
                       f"{f4(S[t]['decomp']['ci95'][k][1])}]")
    L = ["# Stage 1a-R8 — class-truth decomposition (head-level diagnostic, dev-A; ORACLE arms are upper bounds)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += ["", f"**Reading:** {V['reading']} — {V['next']}",
          f"**PRED-HARD (inductive-bias probe):** {V['hard']}   **Geometry of both-correct misses:** {V.get('geometry')}",
          "", "## Decomposition (last epoch, paired vs CTRL, 95 % joint bootstrap)", "",
          "| | seed 1 | seed 2 |", "|---|---|---|"]
    for k, lab in (("G", "G = ORACLE − CTRL"), ("d_seed", "Δ SEED (candidate encoder only)"),
                   ("d_hit", "Δ HIT (hit encoder only)"), ("S_seed", "S_seed = Δ_seed / G"),
                   ("S_hit", "S_hit = Δ_hit / G"), ("I_over_G", "interaction I / G = (G − Δ_seed − Δ_hit) / G"),
                   ("d_hard", "Δ HARD (inductive-bias probe)"), ("F_hard", "F_hard = Δ_hard / G")):
        L.append(f"| {lab} | {ci(1, k)} | {ci(2, k)} |")
    L.append("| per-seed reading / hard | " + " | ".join(f"{S[t]['reading']} / {S[t]['hard']}" for t in (1, 2)) + " |")
    L += ["", "| ARI (last epoch) | " + " | ".join(ARMS) + " |", "|---" * (len(ARMS) + 1) + "|"]
    for t in (1, 2):
        L.append(f"| seed {t} | " + " | ".join(f4(S[t]["ari"][a]) for a in ARMS) + " |")
    L += ["", "## Mechanism (ranking_diag, CPU)", "", "| seed | metric | " + " | ".join(ARMS) + " |",
          "|---|---" + "|---" * len(ARMS) + "|"]
    for t in (1, 2):
        for k, lab in (("diff_class", "different-class share of misses"), ("own_pick", "absolute own-pick"),
                       ("cand_disguised", "diff-class misses: candidate disguised"),
                       ("hit_disguised", "diff-class misses: hit disguised"),
                       ("both_correct", "diff-class misses: both predicted correctly"),
                       ("LR_seed", "LR_seed (owner seed mispredicted, miss vs correct)")):
            L.append(f"| {t} | {lab} | " + " | ".join(f4(S[t]["mech"][a][k]) for a in ARMS) + " |")
    L += ["", "## Both-correct misses of the CTRL heads (geometry vs owner-seed)", "", "| | seed 1 | seed 2 |",
          "|---|---|---|"]
    for k, lab in (("n_both_correct", "n"), ("owner_seed_wrong", "owner seed also mispredicted"),
                   ("picked_nearer", "picked candidate nearer than the owner"),
                   ("nearer_and_owner_seed_ok", "picked nearer AND owner seed correct"), ("reading", "reading")):
        L.append(f"| {lab} | " + " | ".join(str(S[t]["geometry"][k]) if k in ("n_both_correct", "reading")
                                            else f4(S[t]["geometry"][k]) for t in (1, 2)) + " |")
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
