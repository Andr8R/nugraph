#!/usr/bin/env python3
"""Stage 1a-R11 report: confirm co-membership (lambda 0.3) with the no-merge decode on dev-B (a REUSED split).

Pre-registration: experiments/particle_hierarchy/comem_confirm.md (written before this code ran). Nine frozen R10
heads, no training. Pure python; reads, under --root:
  {devA,devB}/{NM,DEP}/{ARM}_s{t}/{eval_events.csv,eval_eval_meta.json}   (soft_particle eval --tag eval)
and, for gate F5, R10's diag_last/{ARM}_s{t}/diagnose_events.csv under --r10-root.
"""
import argparse
import csv
import json
import math
import statistics as stat
from pathlib import Path

ARMS = ("CTRL", "CM03", "CM1")
PRIMARY_ARM = "CM03"
SEEDS = (1, 2, 3)
DECODES = ("NM", "DEP")
R10_VARIANT = {"NM": "v2_nonull_nomerge", "DEP": "v2_nonull"}
DECODE_KW = {"NM": {"merge_thr": None, "exist_thr": 0.5, "allow_null": False, "merge_scope": "local",
                    "reassign": True, "keep_one": True},
             "DEP": {"merge_thr": 0.5, "exist_thr": 0.5, "allow_null": False, "merge_scope": "local",
                     "reassign": True, "keep_one": True}}
HEAD_SHA = {"CTRL": ("f19c0e3b0069f054", "07e31455b389f980", "21199b887b398588"),
            "CM03": ("88f239c1a46b6d56", "112923713f4c377a", "dd1c66ca6a3bd164"),
            "CM1": ("9f97164341f64d76", "08b9171348b4e4a6", "d637bd5982d62084")}
GAIN, NULL = 0.02, 0.01
RATIO_MIN, SINGLE_SLACK = 0.8, 0.05
MAX_FLIPS = 2
TOL = 1e-9
NEXT = {"CONFIRMED": "register Stage 1c: Stage-1b T + co-membership (lambda 0.3) vs T without it, both retrained at "
                     "one commit, NM decode, seeds 1/2",
        "COLLAPSE": "not confirmed: the gain is over-merging; no campaign",
        "NOT CONFIRMED": "the dev-A decomposition was selection or noise; the information-gap route (trunk semantic "
                         "consistency on T) returns",
        "PARTIAL": "report; no campaign"}


def name(arm, t):
    return f"{arm}_s{t}"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load(root, split, dec, arm, t):
    d = Path(root) / split / dec / name(arm, t)
    rows = {(int(r["batch"]), int(r["event"])): r for r in csv.DictReader(open(d / "eval_events.csv"))}
    meta = json.loads((d / "eval_eval_meta.json").read_text())
    return rows, meta


def r10(r10_root, arm, t, variant):
    p = Path(r10_root) / "diag_last" / name(arm, t) / "diagnose_events.csv"
    return {(int(r["batch"]), int(r["event"])): fnum(r["ari_particle"])
            for r in csv.DictReader(open(p)) if r["variant"] == variant}


def same(x, y):
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(x - y) <= TOL


def paired(a, b):
    """mean +- SE of b - a over events where both are finite."""
    d = [fnum(b[k]["ari_particle"]) - fnum(a[k]["ari_particle"]) for k in sorted(set(a) & set(b))]
    d = [x for x in d if math.isfinite(x)]
    if len(d) < 2:
        return {"mean": math.nan, "se": math.nan, "n": len(d)}
    return {"mean": stat.mean(d), "se": stat.stdev(d) / math.sqrt(len(d)), "n": len(d)}


def shape(rows):
    """median n_pred / n_true and the share of single-cluster events (events with a finite ARI)."""
    ev = [r for r in rows.values() if math.isfinite(fnum(r["ari_particle"]))]
    ratio = [fnum(r["n_pred"]) / max(fnum(r["n_true_all"]), 1) for r in ev if math.isfinite(fnum(r["n_pred"]))]
    single = sum(1 for r in ev if fnum(r["n_pred"]) <= 1) / max(len(ev), 1)
    return {"ratio_median": stat.median(ratio) if ratio else math.nan, "single_share": single}


def meta_fails(meta, split, dec, arm, t):
    tag = f"{split}/{dec}/{name(arm, t)}"
    f = []
    if not str(meta.get("head_sha256", "")).startswith(HEAD_SHA[arm][t - 1]):
        f.append(f"F1 {tag}: head sha256 {str(meta.get('head_sha256'))[:16]} != frozen")
    if not meta.get("cache_checkpoint_sha256") or \
            meta.get("cache_checkpoint_sha256") != meta.get("head_latent_checkpoint_sha256"):
        f.append(f"F4 {tag}: cache checkpoint differs from the head's latent checkpoint")
    if meta.get("decode_kwargs") != DECODE_KW[dec]:
        f.append(f"F4 {tag}: decode {meta.get('decode_kwargs')} is not {dec}")
    if meta.get("git_dirty") is not False:
        f.append(f"F4 {tag}: evaluated from a dirty or unrecorded tree")
    return f


def read_split(root, split):
    fails, data = [], {}
    for dec in DECODES:
        for arm in ARMS:
            for t in SEEDS:
                p = Path(root) / split / dec / name(arm, t) / "eval_events.csv"
                if not p.exists():
                    raise SystemExit(f"missing {p}")
                rows, meta = load(root, split, dec, arm, t)
                data[(split, dec, arm, t)] = rows
                fails += meta_fails(meta, split, dec, arm, t)
    return fails, data


def f5_fails(data, r10_root):
    fails = []
    for dec in DECODES:
        for arm in ARMS:
            for t in SEEDS:
                old = r10(r10_root, arm, t, R10_VARIANT[dec])
                new = data[("devA", dec, arm, t)]
                if set(old) != set(new):
                    fails.append(f"F5 {dec} {name(arm, t)}: dev-A event set differs from R10")
                    continue
                flips = [k for k in old if not same(old[k], fnum(new[k]["ari_particle"]))]
                if len(flips) > MAX_FLIPS:
                    fails.append(f"F5 {dec} {name(arm, t)}: {len(flips)} dev-A events differ from R10 "
                                 f"(limit {MAX_FLIPS})")
    return fails


def devA_gates(root, r10_root):
    """F1 / F4 on the dev-A evaluations and F5; checked BEFORE dev-B is read."""
    fails, data = read_split(root, "devA")
    return fails + f5_fails(data, r10_root), data


def gates(root, r10_root):
    fails, data = devA_gates(root, r10_root)
    fb, db = read_split(root, "devB")
    fails += fb
    data.update(db)
    for split in ("devA", "devB"):
        for dec in DECODES:
            for t in SEEDS:
                ref = data[(split, dec, "CTRL", t)]
                for arm in ARMS[1:]:
                    x = data[(split, dec, arm, t)]
                    if set(x) != set(ref):
                        fails.append(f"F2 {split}/{dec} seed {t}: {arm} evaluated a different event set")
                        continue
                    bad = [k for k in ref if not same(fnum(ref[k]["ari_particle_hardseed"]),
                                                      fnum(x[k]["ari_particle_hardseed"]))]
                    if bad:
                        fails.append(f"F3 {split}/{dec} seed {t}: {arm} hard-seed ARI differs on {len(bad)} events")
    return fails, data


def evaluate(root, r10_root):
    fails, data = gates(root, r10_root)
    res = {"gates": {"pass": not fails, "failures": fails}, "devB": {}, "devA": {}}
    for split in ("devB", "devA"):
        for dec in DECODES:
            for t in SEEDS:
                c = data[(split, dec, "CTRL", t)]
                row = {"ari": {a: stat.mean(v for v in (fnum(r["ari_particle"]) for r in data[(split, dec, a, t)].values())
                                            if math.isfinite(v)) for a in ARMS},
                       "delta": {a: paired(c, data[(split, dec, a, t)]) for a in ARMS[1:]},
                       "shape": {a: shape(data[(split, dec, a, t)]) for a in ARMS}}
                if split == "devB" and dec == "NM":
                    nt = sorted(fnum(r["n_true_all"]) for r in c.values() if math.isfinite(fnum(r["ari_particle"])))
                    edges = [nt[len(nt) * i // 4] for i in (1, 2, 3)]
                    x = data[(split, dec, PRIMARY_ARM, t)]
                    bins = {}
                    for k, r in c.items():
                        dv = fnum(x[k]["ari_particle"]) - fnum(r["ari_particle"])
                        if math.isfinite(dv):
                            bins.setdefault(sum(fnum(r["n_true_all"]) >= e for e in edges), []).append(dv)
                    row["delta_by_ntrue_quartile"] = [stat.mean(bins[b]) if b in bins else math.nan for b in range(4)]
                res[split].setdefault(dec, {})[t] = row
    nm = res["devB"]["NM"]
    delta = [nm[t]["delta"][PRIMARY_ARM]["mean"] for t in SEEDS]
    guard = [nm[t]["shape"][PRIMARY_ARM]["ratio_median"] >= RATIO_MIN
             and nm[t]["shape"][PRIMARY_ARM]["single_share"] <= nm[t]["shape"]["CTRL"]["single_share"] + SINGLE_SLACK
             for t in SEEDS]
    if fails:
        reading = "INVALID"
    elif all(d >= GAIN for d in delta):
        reading = "CONFIRMED" if all(guard) else "COLLAPSE"
    elif stat.mean(delta) < NULL:
        reading = "NOT CONFIRMED"
    else:
        reading = "PARTIAL"
    res["verdict"] = {"reading": reading, "delta": delta, "delta_mean": stat.mean(delta), "collapse_guard": guard,
                      "next": NEXT.get(reading, "a gate failed; no reading")}
    return res


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and math.isfinite(x) else "n/a"


def render(res):
    V = res["verdict"]
    L = ["# Stage 1a-R11 — co-membership (λ 0.3) + no-merge decode, confirmation on dev-B "
         "(a REUSED split, 4th read; frozen R10 heads)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += ["", f"**Reading:** {V['reading']} — {V['next']}", "",
          "## Primary: dev-B, decode NM (no merge), paired Δ ARI vs CTRL ± event SE", "",
          "| arm | " + " | ".join(f"seed {t}" for t in SEEDS) + " | mean |", "|---" * (len(SEEDS) + 2) + "|"]
    for split, dec in (("devB", "NM"), ("devB", "DEP"), ("devA", "NM"), ("devA", "DEP")):
        if (split, dec) != ("devB", "NM"):
            L += ["", f"### {split}, decode {dec}", "", "| arm | " + " | ".join(f"seed {t}" for t in SEEDS)
                  + " | mean |", "|---" * (len(SEEDS) + 2) + "|"]
        for a in ARMS[1:]:
            ds = [res[split][dec][t]["delta"][a] for t in SEEDS]
            L.append(f"| {a}{' (primary)' if a == PRIMARY_ARM and (split, dec) == ('devB', 'NM') else ''} | "
                     + " | ".join(f"{f4(d['mean'])} ± {f4(d['se'])}" for d in ds)
                     + f" | {f4(stat.mean(d['mean'] for d in ds))} |")
    L += ["", "## ARI levels (dev-B)", "", "| decode | seed | " + " | ".join(ARMS) + " |",
          "|---|---" + "|---" * len(ARMS) + "|"]
    for dec in DECODES:
        for t in SEEDS:
            L.append(f"| {dec} | {t} | " + " | ".join(f4(res['devB'][dec][t]['ari'][a]) for a in ARMS) + " |")
    L += ["", "## Collapse guard and cluster shape (dev-B): median n_pred / n_true, single-cluster share", "",
          "| decode | seed | " + " | ".join(ARMS) + " | guard (CM03) |", "|---|---" + "|---" * (len(ARMS) + 1) + "|"]
    for dec in DECODES:
        for t in SEEDS:
            sh = res["devB"][dec][t]["shape"]
            g = V["collapse_guard"][t - 1] if dec == "NM" else ""
            L.append(f"| {dec} | {t} | " + " | ".join(f"{f4(sh[a]['ratio_median'])} / {f4(sh[a]['single_share'])}"
                                                      for a in ARMS) + f" | {g} |")
    L += ["", "## CM03 − CTRL on dev-B (NM) by quartile of n_true", "", "| seed | Q1 | Q2 | Q3 | Q4 |",
          "|---|---|---|---|---|"]
    for t in SEEDS:
        L.append(f"| {t} | " + " | ".join(f4(x) for x in res["devB"]["NM"][t]["delta_by_ntrue_quartile"]) + " |")
    L += ["", "## Merge cost on dev-B (NM − DEP, same head)", "", "| seed | " + " | ".join(ARMS) + " |",
          "|---" * (len(ARMS) + 1) + "|"]
    for t in SEEDS:
        L.append(f"| {t} | " + " | ".join(f4(res['devB']['NM'][t]['ari'][a] - res['devB']['DEP'][t]['ari'][a])
                                           for a in ARMS) + " |")
    return "\n".join(L) + "\n"


def cmd_f5(a):
    fails, _ = devA_gates(a.root, a.r10_root)
    for f in fails:
        print(f)
    if fails:
        raise SystemExit(1)
    print("F1 / F4 / F5 on dev-A: PASS")


def cmd_report(a):
    res = evaluate(a.root, a.r10_root)
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
    s.add_argument("--r10-root", required=True, help="the R10 comem root (gate F5)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    s = sub.add_parser("f5", help="gates F1 / F4 / F5 on the dev-A evaluations only (run before dev-B is read)")
    s.add_argument("--root", required=True)
    s.add_argument("--r10-root", required=True)
    s.set_defaults(fn=cmd_f5)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
