#!/usr/bin/env python3
"""Stage 1a-R2: does the Stage-1b representation make the extra M=20 candidates more usable?

Pre-registration: experiments/particle_hierarchy/retrieval_tc.md (written before any result).

Design: fresh Stage-1a heads (frozen recipe) on the four Stage-1b trunks {T, C} x trunk seeds {1, 2},
at M in {5, 20}, head seed 1, all trained on one seeded train-split event set; epoch selected on dev-A;
evaluated once on dev-B with the dev-B caches the Stage-1b decision used.

  identity  (EAF, needs torch) cached events are the same events, in the same order, in every cache:
            per event (batch, event, n_hit, sha1(y_i), sha1(y_sem)) -- truth only, so trunk-independent
  report    (pure python) gates + the pre-registered contrasts from the eight devB_events.csv files

Layout under --root:  heads/{A}_s{t}_m{M}/train_meta.json   eval/{A}_s{t}_m{M}/devB_events.csv
"""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

ARMS, TRUNK_SEEDS, MS, HEAD_SEED = ("T", "C"), (1, 2), (5, 20), 1
TAG = "devB"
METRICS = ("ari", "own", "comp", "pur")        # outcome + the three mechanism metrics
LABEL = {"ari": "particle ARI (deployed decode)", "own": "absolute own-pick (event mean)",
         "comp": "completeness", "pur": "purity (homogeneity)"}
N_BOOT, BOOT_SEED = 2000, 20260924


def run_name(arm, t, m):
    return f"{arm}_s{t}_m{m}"


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def fingerprints(cache, lo=0, hi=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dune_materialization"))
    import mat_bench as mb
    out = []
    for ev in mb.iter_cache(cache, lo, hi):
        # int64 before hashing: the fingerprint must not depend on the dtype a cache version used
        yi, ys = ev["y_i"].long().contiguous().numpy(), ev["y_sem"].long().contiguous().numpy()
        out.append((int(ev["batch"]), int(ev["event"]), int(yi.shape[0]),
                    hashlib.sha1(yi.tobytes()).hexdigest(), hashlib.sha1(ys.tobytes()).hexdigest()))
    return out


def compare_fingerprints(named):
    """named: {name: [fingerprint, ...]} -> (ok, message). The first cache is the reference."""
    (ref_name, ref), *rest = named.items()
    if not ref:
        return False, f"{ref_name}: no events"
    for name, fp in rest:
        if len(fp) != len(ref):
            return False, f"{name}: {len(fp)} events vs {len(ref)} in {ref_name}"
        bad = [i for i, (a, b) in enumerate(zip(ref, fp)) if a != b]
        if bad:
            return False, f"{name}: {len(bad)} events differ from {ref_name}, first at index {bad[0]}"
    return True, f"{len(named)} caches, {len(ref)} identical events"


def cmd_identity(a):
    lo, hi = (int(x) if x else None for x in a.batches.split(":"))
    named = {c: fingerprints(c, lo or 0, hi) for c in a.caches}
    ok, msg = compare_fingerprints(named)
    print(("PASS: " if ok else "FAIL: ") + msg)
    if not ok:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def load_events(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        pop = fnum(r["pop"])
        rows[(int(r["batch"]), int(r["event"]))] = {
            "ari": fnum(r["ari_particle"]), "comp": fnum(r["completeness"]),
            "pur": fnum(r["purity"]), "hard": fnum(r["ari_particle_hardseed"]),
            "own": fnum(r["pick_own"]) / pop if pop > 0 else math.nan,
            "pick_own": fnum(r["pick_own"]), "own_in_nb": fnum(r["own_in_nb"]), "pop": pop}
    return rows


def mean_se(d):
    d = [x for x in d if math.isfinite(x)]
    if len(d) < 2:
        return math.nan, math.nan, len(d)
    return st.mean(d), st.stdev(d) / math.sqrt(len(d)), len(d)


def did(ev, t, metric):
    """Per-event difference-in-differences for trunk seed t: (T20 - T5) - (C20 - C5)."""
    runs = {(a, m): ev[run_name(a, t, m)] for a in ARMS for m in MS}
    keys = set.intersection(*(set(r) for r in runs.values()))
    out = []
    for k in sorted(keys):
        v = {am: runs[am][k][metric] for am in runs}
        out.append((v[("T", 20)] - v[("T", 5)]) - (v[("C", 20)] - v[("C", 5)]))
    return out


def paired(ev, a, b, metric):
    ka, kb = ev[a], ev[b]
    return [kb[k][metric] - ka[k][metric] for k in sorted(set(ka) & set(kb))]


def kappa(rows5, rows20, keys):
    """Conversion of newly available owners: d(sum pick_own) / d(sum own_in_nb), M5 -> M20."""
    dp = sum(rows20[k]["pick_own"] - rows5[k]["pick_own"] for k in keys)
    da = sum(rows20[k]["own_in_nb"] - rows5[k]["own_in_nb"] for k in keys)
    return dp / da if da > 0 else math.nan


def kappa_diff(ev, t, boot=N_BOOT, seed=BOOT_SEED):
    """kappa_T - kappa_C for trunk seed t, with a paired event bootstrap 95 % interval."""
    r = {(a, m): ev[run_name(a, t, m)] for a in ARMS for m in MS}
    keys = sorted(set.intersection(*(set(x) for x in r.values())))

    def stat(ks):
        return (kappa(r[("T", 5)], r[("T", 20)], ks), kappa(r[("C", 5)], r[("C", 20)], ks))
    kt, kc = stat(keys)
    rng = random.Random(seed)
    diffs = []
    for _ in range(boot):
        ks = [keys[rng.randrange(len(keys))] for _ in keys]
        a, b = stat(ks)
        if math.isfinite(a) and math.isfinite(b):
            diffs.append(a - b)
    diffs.sort()
    lo_, hi_ = (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]) if diffs else (math.nan,) * 2
    return {"kappa_T": kt, "kappa_C": kc, "diff": kt - kc, "ci95": [lo_, hi_], "n_events": len(keys)}


def same_value(x, y, tol=1e-9):
    """Equal within tol, where NaN equals only NaN (a NaN-vs-number mismatch must not pass)."""
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(x - y) <= tol


def check_gates(root, ev, s1b_devb):
    """G1 heads as registered; G2 every run on the same events; G3 hard seed == Stage-1b dev-B."""
    fails = []
    for a in ARMS:
        for t in TRUNK_SEEDS:
            for m in MS:
                meta_p = root / "heads" / run_name(a, t, m) / "train_meta.json"
                if not meta_p.exists():
                    fails.append(f"G1 {run_name(a, t, m)}: no train_meta.json")
                    continue
                meta = json.loads(meta_p.read_text())
                want = {"m": m, "seed": HEAD_SEED, "rule": "quant:0.995:0.05:0.5", "r": 2,
                        "width": 64, "epochs": 8, "target": "own", "exist_target": "real",
                        "merge_scope": "local", "dist_only": False}
                bad = {k: meta.get(k) for k, v in want.items() if meta.get(k) != v}
                if bad:
                    fails.append(f"G1 {run_name(a, t, m)}: {bad}")
                if f"cache_train_{a}_s{t}" not in str(meta.get("train_cache", "")):
                    fails.append(f"G1 {run_name(a, t, m)}: train cache {meta.get('train_cache')}")
    sets = {n: set(r) for n, r in ev.items()}
    ref = next(iter(sets.values()))
    for n, s in sets.items():
        if s != ref:
            fails.append(f"G2 {n}: {len(s ^ ref)} events differ from the other runs")
    for a in ARMS:
        for t in TRUNK_SEEDS:
            p = Path(s1b_devb) / f"{a}_s{t}" / "soft" / f"{TAG}_events.csv"
            if not p.exists():
                fails.append(f"G3 {a}_s{t}: no Stage-1b reference {p}")
                continue
            refr = load_events(p)
            for m in MS:
                r = ev[run_name(a, t, m)]
                if set(r) != set(refr):
                    fails.append(f"G3 {run_name(a, t, m)}: event set differs from Stage-1b dev-B")
                    continue
                n_bad = sum(1 for k in r if not same_value(r[k]["hard"], refr[k]["hard"]))
                if n_bad:
                    fails.append(f"G3 {run_name(a, t, m)}: hard seed differs on {n_bad} events")
    return fails


def combine(per_seed):
    """Mean over trunk seeds of the per-seed means; SE from the per-seed SEs."""
    ms, ses = [p[0] for p in per_seed], [p[1] for p in per_seed]
    return st.mean(ms), math.sqrt(sum(s * s for s in ses)) / len(ses)


def passes(per_seed, comb):
    """> 2 SE combined AND the same sign on both trunk seeds (Stage-1b convention)."""
    m, se = comb
    return m > 2 * se and all(p[0] > 0 for p in per_seed)


def evaluate(root, s1b_devb):
    ev = {}
    for a in ARMS:
        for t in TRUNK_SEEDS:
            for m in MS:
                p = root / "eval" / run_name(a, t, m) / f"{TAG}_events.csv"
                if not p.exists():
                    raise SystemExit(f"missing {p}")
                ev[run_name(a, t, m)] = load_events(p)
    fails = check_gates(root, ev, s1b_devb)
    res = {"gates": {"pass": not fails, "failures": fails}, "runs": {}, "interaction": {},
           "widening": {}, "t_minus_c": {}, "kappa": {}, "hit_weighted_own": {}}
    for n, r in ev.items():
        res["runs"][n] = {k: mean_se([x[k] for x in r.values()])[0] for k in METRICS}
        tot = lambda c: sum(x[c] for x in r.values() if math.isfinite(x[c]))
        res["runs"][n]["own_hitw"] = tot("pick_own") / max(tot("pop"), 1)
        res["runs"][n]["avail_hitw"] = tot("own_in_nb") / max(tot("pop"), 1)
        res["runs"][n]["n_events"] = len(r)
    for k in METRICS:
        per = [mean_se(did(ev, t, k)) for t in TRUNK_SEEDS]
        comb = combine(per)
        res["interaction"][k] = {"per_seed": [p[:2] for p in per], "n": [p[2] for p in per],
                                 "combined": comb, "pass": passes(per, comb),
                                 "fail_negative": comb[0] < -2 * comb[1]}
        for a in ARMS:
            per_w = [mean_se(paired(ev, run_name(a, t, 5), run_name(a, t, 20), k)) for t in TRUNK_SEEDS]
            res["widening"].setdefault(k, {})[a] = {"per_seed": [p[:2] for p in per_w], "combined": combine(per_w)}
        for m in MS:
            per_tc = [mean_se(paired(ev, run_name("C", t, m), run_name("T", t, m), k)) for t in TRUNK_SEEDS]
            res["t_minus_c"].setdefault(k, {})[f"M{m}"] = {"per_seed": [p[:2] for p in per_tc], "combined": combine(per_tc)}
    for t in TRUNK_SEEDS:
        res["kappa"][f"s{t}"] = kappa_diff(ev, t)
        # hit-weighted own-pick interaction: reported next to the event-mean test (signs must agree)
        hw = {am: res["runs"][run_name(am[0], t, am[1])]["own_hitw"] for am in [(a, m) for a in ARMS for m in MS]}
        res["hit_weighted_own"][f"s{t}"] = (hw[("T", 20)] - hw[("T", 5)]) - (hw[("C", 20)] - hw[("C", 5)])
    res["verdict"] = verdict(res)
    return res


def verdict(res):
    I = res["interaction"]
    if not res["gates"]["pass"]:
        return "INVALID: a gate failed; no scientific reading"
    kap = [res["kappa"][f"s{t}"] for t in TRUNK_SEEDS]
    kappa_up = all(k["ci95"][0] > 0 for k in kap)
    own_sign = [I["own"]["combined"][0]] + list(res["hit_weighted_own"].values())
    own_consistent = all(x > 0 for x in own_sign) or all(x < 0 for x in own_sign)
    lines = []
    if I["ari"]["pass"]:
        lines.append("PRIMARY SUPPORTED: widening M5->M20 helps more on the T latent than on C "
                     "(the Stage-1b representation makes the extra candidates more usable).")
        if I["own"]["pass"] and kappa_up and own_consistent:
            lines.append("MECHANISM = owner identification: T converts more newly available owners "
                         "(own-pick interaction and kappa both up) -> discrimination is improvable "
                         "through the representation; supports a representation-level Stage 1c.")
        elif I["comp"]["pass"]:
            lines.append("MECHANISM = partition consistency, not owner identification: completeness "
                         "rises without a clear own-pick gain -> the representation helps grouping; "
                         "owner discrimination still needs head-level information.")
        else:
            lines.append("MECHANISM UNRESOLVED: ARI interaction without a significant mechanism metric.")
    elif I["ari"]["fail_negative"]:
        lines.append("PRIMARY REVERSED: widening helps LESS on T; read against availability "
                     "(T may already hold more owners in its top-5).")
    else:
        lines.append("PRIMARY NOT SUPPORTED: representation and width act additively; the T latent "
                     "does not unlock the extra candidates -> head-level discrimination features "
                     "before any representation-level Stage 1c.")
    if I["pur"]["fail_negative"]:
        lines.append("FLAG: purity interaction < -2 SE (completeness bought with purity).")
    if not own_consistent:
        lines.append("FLAG: event-mean and hit-weighted own-pick interactions disagree in sign.")
    return " ".join(lines)


def fmt(ms):
    m, se = ms
    return f"{m:+.4f} ± {se:.4f}" if math.isfinite(m) else "n/a"


def render(res):
    L = ["# Stage 1a-R2 — T/C latent × M=20 (dev-B, pre-registered)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'}"]
    L += [f"- {f}" for f in res["gates"]["failures"]]
    L += ["", f"**Verdict:** {res['verdict']}", "",
          "## Interaction (T20 − T5) − (C20 − C5), per-event paired", "",
          "| metric | seed 1 | seed 2 | combined | pass (>2 SE, same sign) |", "|---|---|---|---|---|"]
    for k in METRICS:
        i = res["interaction"][k]
        L.append(f"| {LABEL[k]} | {fmt(i['per_seed'][0])} | {fmt(i['per_seed'][1])} | "
                 f"{fmt(i['combined'])} | {'yes' if i['pass'] else 'no'} |")
    L += ["", "## Conversion of newly available owners κ = Δpick_own / Δown_in_nb (hit-weighted)", "",
          "| trunk seed | κ_T | κ_C | κ_T − κ_C | 95 % bootstrap |", "|---|---|---|---|---|"]
    for t in TRUNK_SEEDS:
        k = res["kappa"][f"s{t}"]
        L.append(f"| {t} | {k['kappa_T']:.4f} | {k['kappa_C']:.4f} | {k['diff']:+.4f} | "
                 f"[{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}] |")
    L += ["", "Hit-weighted own-pick interaction: "
          + ", ".join(f"seed {t} {res['hit_weighted_own'][f's{t}']:+.4f}" for t in TRUNK_SEEDS), "",
          "## Widening M5 → M20 within each trunk (combined over trunk seeds)", "",
          "| metric | T | C |", "|---|---|---|"]
    for k in METRICS:
        w = res["widening"][k]
        L.append(f"| {LABEL[k]} | {fmt(w['T']['combined'])} | {fmt(w['C']['combined'])} |")
    L += ["", "## T − C at fixed M (combined)", "", "| metric | M5 | M20 |", "|---|---|---|"]
    for k in METRICS:
        tc = res["t_minus_c"][k]
        L.append(f"| {LABEL[k]} | {fmt(tc['M5']['combined'])} | {fmt(tc['M20']['combined'])} |")
    L += ["", "## Runs (event means; own/avail also hit-weighted)", "",
          "| run | ARI | own (ev) | own (hit) | avail (hit) | completeness | purity | n |",
          "|---|---|---|---|---|---|---|---|"]
    for n, r in res["runs"].items():
        L.append(f"| {n} | {r['ari']:.4f} | {r['own']:.4f} | {r['own_hitw']:.4f} | {r['avail_hitw']:.4f} | "
                 f"{r['comp']:.4f} | {r['pur']:.4f} | {r['n_events']} |")
    return "\n".join(L) + "\n"


def cmd_report(a):
    res = evaluate(Path(a.root), a.stage1b_devb)
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
    s = sub.add_parser("identity")
    s.add_argument("--caches", nargs="+", required=True, help="the first one is the reference")
    s.add_argument("--batches", default="0:")
    s.set_defaults(fn=cmd_identity)
    s = sub.add_parser("report")
    s.add_argument("--root", required=True)
    s.add_argument("--stage1b-devb", required=True, help="stage1b/eval_devB (hard-seed reference)")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
