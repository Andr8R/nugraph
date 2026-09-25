#!/usr/bin/env python3
"""Stage 1a-R6 report: how much of the ORACLE class gain does the trunk's PREDICTED semantic signal recover?

Pre-registration: experiments/particle_hierarchy/pred_class.md (written before this code ran).

Pure python. Reads, under --root (same layout as class_probe, three arms CTRL / ORACLE / PRED):
  readout/{train,devA}.json                  (soft_particle readout-check; gate P0)
  heads/{ARM}_s{t}/train_meta.json            diag_last|diag_best/{ARM}_s{t}/diagnose_events.csv
  rank/{ARM}_s{t}/rank_summary.json           (ranking_diag run --pred-class)
and the R5 outputs under --r5-root for the replication gate P5.
"""
import argparse
import json
import math
import random
from pathlib import Path

import importlib.util

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("class_probe", HERE / "class_probe.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)
rd = cp.rd

ARMS = ("CTRL", "ORACLE", "PRED")
MODE = {"CTRL": "zeros", "ORACLE": "oracle", "PRED": "pred"}
SEEDS = cp.SEEDS
PRIMARY = cp.PRIMARY
REPLICATION_TOL = 0.03
N_BOOT, BOOT_SEED = 2000, 20260925
F_DEPLOY, F_TRUNK, GAIN_DEPLOY, GAIN_ADOPT = 0.50, 0.20, 0.03, 0.01


def name(arm, t):
    return f"{arm}_s{t}"


def diag(root, sub, arm, t, variant=PRIMARY):
    return cp.diag(Path(root) / sub / name(arm, t) / "diagnose_events.csv", variant)


def gates(root, r5_root):
    fails, metas = [], {}
    ro = {}
    for split in ("train", "devA"):
        p = Path(root) / "readout" / f"{split}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        ro[split] = json.loads(p.read_text())
        if not ro[split].get("pass"):
            fails.append(f"P0 {split}: {ro[split].get('fg_reclassified_diffuse')} of foreground re-classified diffuse")
    if ro["train"].get("checkpoint_sha256") != ro["devA"].get("checkpoint_sha256"):
        fails.append("P0: train and dev-A read-outs come from different checkpoints")
    for arm in ARMS:
        for t in SEEDS:
            p = Path(root) / "heads" / name(arm, t) / "train_meta.json"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            m = json.loads(p.read_text())
            metas[(arm, t)] = m
            if m.get("class_channels") != MODE[arm] or m.get("m") != 20 or m.get("seed") != t:
                fails.append(f"P3 {name(arm, t)}: class_channels {m.get('class_channels')} m {m.get('m')} "
                             f"seed {m.get('seed')}")
            if m.get("git_dirty") is not False:
                fails.append(f"P3 {name(arm, t)}: trained from a dirty or unrecorded tree")
            if m.get("latent_checkpoint_sha256") != ro["train"].get("checkpoint_sha256"):
                fails.append(f"P0 {name(arm, t)}: latent checkpoint differs from the read-out checkpoint")
    for t in SEEDS:
        c = metas[("CTRL", t)]
        for arm in ARMS[1:]:
            o = metas[(arm, t)]
            if c.get("init_sha256") is None or c.get("init_sha256") != o.get("init_sha256"):
                fails.append(f"P1 seed {t}: {arm} initialisation differs from CTRL")
            if not c.get("perm_sha256") or c.get("perm_sha256") != o.get("perm_sha256"):
                fails.append(f"P2 seed {t}: {arm} training order differs from CTRL")
            diff = ({k for k in set(c) | set(o) if c.get(k) != o.get(k)}
                    - cp.OUTCOME_KEYS - cp.PROVENANCE_KEYS - cp.CHECKED_ELSEWHERE)
            if diff:
                fails.append(f"P3 seed {t}: {arm} recipe differs from CTRL in {sorted(diff)}")
        runs = {arm: diag(root, "diag_last", arm, t) for arm in ARMS}
        keys = set(runs["CTRL"])
        for arm in ARMS[1:]:
            if set(runs[arm]) != keys:
                fails.append(f"P4 seed {t}: {arm} evaluated a different event set")
                continue
            bad = [k for k in keys if not cp.rd_same(runs["CTRL"][k][1], runs[arm][k][1])]
            if bad:
                fails.append(f"P4 seed {t}: {arm} hard-seed ARI differs on {len(bad)} events")
        new = cp.paired(runs["CTRL"], runs["ORACLE"])["mean"]
        old = cp.paired(cp.diag(Path(r5_root) / "diag_last" / name("CTRL", t) / "diagnose_events.csv", PRIMARY),
                        cp.diag(Path(r5_root) / "diag_last" / name("ORACLE", t) / "diagnose_events.csv", PRIMARY))["mean"]
        if not abs(new - old) <= REPLICATION_TOL:
            fails.append(f"P5 seed {t}: ORACLE − CTRL {new:.4f} does not reproduce R5 {old:.4f}")
    return fails, metas, ro


def recovered(runs, boot=N_BOOT, seed=BOOT_SEED):
    """F = mean(PRED − CTRL) / mean(ORACLE − CTRL) with a paired event bootstrap (arms resampled together)."""
    keys = [k for k in sorted(set.intersection(*(set(r) for r in runs.values())))
            if all(math.isfinite(runs[a][k][0]) for a in ARMS)]
    dO = [runs["ORACLE"][k][0] - runs["CTRL"][k][0] for k in keys]
    dP = [runs["PRED"][k][0] - runs["CTRL"][k][0] for k in keys]
    f = lambda o, p: (sum(p) / len(p)) / (sum(o) / len(o)) if sum(o) != 0 else math.nan
    F = f(dO, dP)
    rng = random.Random(seed)
    fs = []
    for _ in range(boot):
        idx = [rng.randrange(len(keys)) for _ in keys]
        v = f([dO[i] for i in idx], [dP[i] for i in idx])
        if math.isfinite(v):
            fs.append(v)
    fs.sort()
    ci = [fs[int(0.025 * len(fs))], fs[int(0.975 * len(fs)) - 1]] if fs else [math.nan, math.nan]
    return {"F": F, "ci95": ci, "n": len(keys)}


def evaluate(root, r5_root):
    fails, metas, ro = gates(root, r5_root)
    res = {"gates": {"pass": not fails, "failures": fails}, "readout": ro, "seeds": {}}
    for t in SEEDS:
        row = {}
        for which, sub in (("last", "diag_last"), ("best", "diag_best")):
            for var in (PRIMARY, "pick_only"):
                runs = {arm: diag(root, sub, arm, t, var) for arm in ARMS}
                for arm in ARMS[1:]:
                    row[f"{which}:{var}:{arm}"] = cp.paired(runs["CTRL"], runs[arm])
                for arm in ARMS:
                    row[f"{which}:{var}:{arm}:ari"] = rd._nanmean([v[0] for v in runs[arm].values()])
                if which == "last" and var == PRIMARY:
                    row["F"] = recovered(runs)
        row["mech"] = {arm: cp.mechanism(root, arm, t) for arm in ARMS}
        st = json.loads((Path(root) / "rank" / name("CTRL", t) / "rank_summary.json").read_text())["stats"]
        same, diff = st.get("err_diff_pred_same"), st.get("err_diff_pred_diff")
        row["addressability"] = ({"trunk_confused": same / max(same + diff, 1), "addressable": diff / max(same + diff, 1),
                                  "n": same + diff} if same is not None else None)
        res["seeds"][t] = row
    res["verdict"] = verdict(res) if not fails else {"reading": "INVALID", "next": "a gate failed; no reading"}
    return res


def verdict(res):
    F = [res["seeds"][t]["F"]["F"] for t in SEEDS]
    gP = [res["seeds"][t][f"last:{PRIMARY}:PRED"]["mean"] for t in SEEDS]
    if all(f >= F_DEPLOY for f in F) and all(g >= GAIN_DEPLOY for g in gP):
        return {"reading": "DEPLOYABLE", "F": F, "gain_pred": gP,
                "next": "adopt predicted-class channels in the deployable head; confirm on a decision split"}
    if all(f < F_TRUNK for f in F):
        return {"reading": "TRUNK-LIMITED", "F": F, "gain_pred": gP,
                "next": "the gain needs semantics the trunk does not have: trunk-level semantic quality is the "
                        "lever, sized by the addressability split"}
    adopt = all(g >= GAIN_ADOPT for g in gP)
    return {"reading": "PARTIAL", "F": F, "gain_pred": gP, "adopt_pred_channels": adopt,
            "next": ("adopt predicted channels (cheap deployable gain); " if adopt else "")
                    + "treat the remainder as trunk-limited"}


def render(res):
    f4, V = rd.f4, res["verdict"]
    L = ["# Stage 1a-R6 — predicted-class probe vs the ORACLE upper bound (dev-A, pre-registered)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += ["", f"**Reading:** {V['reading']} — next: {V['next']}", "",
          "## Recovered fraction F = (PRED − CTRL) / (ORACLE − CTRL), last epoch, deployed decode", "",
          "| | seed 1 | seed 2 |", "|---|---|---|"]
    S = res["seeds"]
    cell = lambda t, k: f"{f4(S[t][k]['mean'])} ± {f4(S[t][k]['se'])}"
    L += [f"| Δ ORACLE (upper bound) | {cell(1, f'last:{PRIMARY}:ORACLE')} | {cell(2, f'last:{PRIMARY}:ORACLE')} |",
          f"| Δ PRED (deployable) | {cell(1, f'last:{PRIMARY}:PRED')} | {cell(2, f'last:{PRIMARY}:PRED')} |",
          "| **F** [95 % bootstrap] | " + " | ".join(
              f"**{f4(S[t]['F']['F'])}** [{f4(S[t]['F']['ci95'][0])}, {f4(S[t]['F']['ci95'][1])}]" for t in SEEDS) + " |",
          "| ARI CTRL / PRED / ORACLE | " + " | ".join(
              " / ".join(f4(S[t][f"last:{PRIMARY}:{a}:ari"]) for a in ARMS) for t in SEEDS) + " |",
          f"| Δ PRED pick_only | {cell(1, 'last:pick_only:PRED')} | {cell(2, 'last:pick_only:PRED')} |",
          f"| Δ PRED best epoch (secondary) | {cell(1, f'best:{PRIMARY}:PRED')} | {cell(2, f'best:{PRIMARY}:PRED')} |",
          "", "## Mechanism (ranking_diag, CPU): CTRL / PRED / ORACLE", "", "| | seed 1 | seed 2 |", "|---|---|---|"]
    for k, lab in (("diff_class", "different-class share of misses"), ("same_class", "same-class share"),
                   ("spurious", "spurious share"), ("own_pick", "absolute own-pick"), ("recall1", "owner recall@1"),
                   ("closure2", "closure@2"), ("closure3", "closure@3")):
        L.append(f"| {lab} | " + " | ".join(" / ".join(f4(S[t]["mech"][a][k]) for a in ARMS) for t in SEEDS) + " |")
    L += ["", "## Addressability of CTRL's different-class misses by the trunk's predicted classes", "",
          "| | seed 1 | seed 2 |", "|---|---|---|"]
    for k, lab in (("trunk_confused", "trunk predicts the SAME class for hit and candidate"),
                   ("addressable", "trunk predicts DIFFERENT classes")):
        L.append(f"| {lab} | " + " | ".join(f4(S[t]["addressability"][k]) if S[t]["addressability"] else "n/a"
                                            for t in SEEDS) + " |")
    L += ["", "## Read-out fidelity (P0) and trunk semantic accuracy on foreground population hits", "",
          "| | train | dev-A |", "|---|---|---|",
          "| foreground re-classified diffuse | " + " | ".join(f4(res["readout"][s]["fg_reclassified_diffuse"])
                                                        for s in ("train", "devA")) + " |",
          "| predicted == true class | " + " | ".join(f4(res["readout"][s]["pop_accuracy"])
                                                  for s in ("train", "devA")) + " |"]
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(a.root, a.r5_root)
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
    s.add_argument("--r5-root", required=True, help="Stage 1a-R5 class_probe root (replication gate P5)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
