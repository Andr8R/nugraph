#!/usr/bin/env python3
"""Stage 1c: trunk co-membership supervision, Stage-1b T vs T + co-membership (lambda 0.3). Gates and report.

Pre-registration: experiments/particle_hierarchy/stage1c.md (written before this code ran). Pure python.

  smoke-check --root R/smoke                 resume identity + matched order + logged/trained term (dev-A smoke)
  train-gates --root R                       K1 (matched training) and K2 (manipulation), before any read-out
  head-gates  --root R                       K4 (matched heads), before the test lock opens
  report      --root R --out R/report        all gates, the primary on `test`, readings, secondary tables

Layout under R: trunks/{T,TCM}_s{1,2}/ (finetune outputs), gates/K3_{train,devA,test}.txt (identity gate output),
cache_test_{ARM}_s{t}/batch_metrics.jsonl, heads/{ARM}_s{t}_{CMH,CTH}_h{h}/train_meta.json,
eval/{ARM}_s{t}/{CMH_h1,CMH_h2,CTH_h1,soft}/{NM,DEP}/test_events.csv, rank/{ARM}_s{t}_CMH_h1/rank_summary.json.
"""
import argparse
import csv
import importlib.util
import json
import math
import statistics as stat
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ARMS = ("T", "TCM")
SEEDS = (1, 2)
LAMBDA = {"T": 0.0, "TCM": 0.3}
HEADS = ("CMH_h1", "CMH_h2", "CTH_h1")
HEAD_LAMBDA = {"CMH": 0.3, "CTH": 0.0}
PRIMARY_HEADS = ("CMH_h1", "CMH_h2")
DECODES = ("NM", "DEP")
EPOCHS = 5
MANIP = 0.9
HEALTH = {"semantic": "semantic/recall-test", "event": "event/recall-test",
          "potential": "instance/potential-loss-test"}
HEALTH_MIN = -0.01
S1B_T_TEST_HARD = 0.1920                    # Stage-1b final (test): T hard-seed decoder, mean of seeds (0.1835 / 0.2006)
REPRO_TOL = 0.025
ADOPT_SEED, ADOPT_SE = 0.015, 3.0
HEAD_SUFF, HARM = 0.005, -0.01
RATIO_MIN, SINGLE_SLACK = 0.8, 0.05
REFS = {"Stage-1b T soft (own M=5 head, DEP), test": (0.2998, 0.3073),
        "ep29 + frozen Stage-1a head, test": (0.2486,)}
STEP_KEYS = ("global_step", "epoch", "batch", "ids_hash")
NAN = float("nan")


def trunk(a, t):
    return f"{a}_s{t}"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


# --------------------------------------------------------------------------- #
# smoke (dev-A only)
# --------------------------------------------------------------------------- #
def smoke_check(root):
    root = Path(root)
    fails, info = [], {}
    R, I, C = (read_csv(root / n / "steps.csv") for n in ("R", "I", "C"))
    key = lambda rows: [tuple(r[k] for k in STEP_KEYS) for r in rows]
    if len(R) != 60 or key(R) != key(I):
        fails.append(f"resume: {len(R)} / {len(I)} rows, identical keys {key(R) == key(I)} (want 60, True)")
    man = json.loads((root / "I" / "manifest.json").read_text())
    if not man.get("resumes") or not man.get("finished"):
        fails.append("resume: the interrupted run was not resumed and finished")
    d = [abs(fnum(a["loss"]) - fnum(b["loss"])) for a, b in zip(R, I)]
    pre, post = d[30:40], d[45:60]
    info["loss_gap_pre_median"], info["loss_gap_post_median"] = stat.median(pre), stat.median(post)
    if not stat.median(post) <= 5 * max(stat.median(pre), 1e-3):
        fails.append(f"resume: loss gap jumps after the boundary ({stat.median(pre):.4f} -> {stat.median(post):.4f})")
    if key(C) != key(R)[:len(C)] or len(C) != 20:
        fails.append("order: the control's (step, epoch, batch, ids_hash) rows differ from TCM's")
    for name, rows, lam in (("TCM", R, LAMBDA["TCM"]), ("T", C, LAMBDA["T"])):
        cm = [fnum(r.get("comem")) for r in rows]
        if not all(fin(x) and x > 0 for x in cm):
            fails.append(f"{name}: the co-membership term is not logged on every step")
            continue
        gap = [fnum(r["soft"]) - (fnum(r["mem"]) + fnum(r["exist"]) + fnum(r["merge"])) for r in rows]
        want = [lam * x for x in cm]
        if not all(abs(g - w) <= 1e-4 * max(1.0, abs(w)) for g, w in zip(gap, want)):
            fails.append(f"{name}: soft - (mem + exist + merge) != {lam} x comem")
    return fails, info


# --------------------------------------------------------------------------- #
# gates before the test lock
# --------------------------------------------------------------------------- #
def train_gates(root):
    root = Path(root)
    fails, info = [], {"comem_last": {}}
    configs = {}
    for a in ARMS:
        for t in SEEDS:
            d = root / "trunks" / trunk(a, t)
            man = json.loads((d / "manifest.json").read_text())
            configs[(a, t)] = man["config"]
            if not man.get("finished"):
                fails.append(f"K1 {trunk(a, t)}: not finished")
            if man["config"].get("git_dirty") is not False:
                fails.append(f"K1 {trunk(a, t)}: dirty or unrecorded tree")
            if man["config"].get("comem") != LAMBDA[a] or man["config"].get("comem_log") is not True:
                fails.append(f"K1 {trunk(a, t)}: comem {man['config'].get('comem')} log "
                             f"{man['config'].get('comem_log')}")
            if man["config"].get("epochs") != EPOCHS or man["config"].get("arm") != "T":
                fails.append(f"K1 {trunk(a, t)}: epochs {man['config'].get('epochs')} arm {man['config'].get('arm')}")
            if not (d / f"trunk_ep{EPOCHS - 1}.ckpt").exists() or not (d / f"soft_ep{EPOCHS - 1}.pt").exists():
                fails.append(f"K1 {trunk(a, t)}: last-epoch checkpoints missing")
            hist = [r for r in read_csv(d / "history.csv") if r.get("epoch") == str(EPOCHS - 1)]
            info["comem_last"][trunk(a, t)] = fnum(hist[-1].get("train/comem")) if hist else NAN
    commits = {c.get("git_commit") for c in configs.values()}
    if len(commits) != 1:
        fails.append(f"K1: trunks trained at different commits {sorted(map(str, commits))}")
    for t in SEEDS:
        c, o = configs[("T", t)], configs[("TCM", t)]
        diff = {k for k in set(c) | set(o) if c.get(k) != o.get(k)} - {"comem"}
        if diff:
            fails.append(f"K1 seed {t}: resume configs differ beyond comem: {sorted(diff)}")
        rows = {a: [tuple(r[k] for k in STEP_KEYS) for r in read_csv(root / "trunks" / trunk(a, t) / "steps.csv")]
                for a in ARMS}
        if rows["T"] != rows["TCM"] or not rows["T"]:
            fails.append(f"K1 seed {t}: per-step (step, epoch, batch, ids_hash) rows differ ({len(rows['T'])} vs "
                         f"{len(rows['TCM'])})")
        cT, cM = info["comem_last"][trunk("T", t)], info["comem_last"][trunk("TCM", t)]
        if not (fin(cT) and fin(cM) and cM <= MANIP * cT):
            fails.append(f"K2 seed {t}: TCM last-epoch co-membership term {cM:.4f} > {MANIP} x T's {cT:.4f}")
    info["commit"] = commits.pop() if len(commits) == 1 else None
    return fails, info


def head_name(a, t, h):
    return f"{trunk(a, t)}_{h}"


def head_gates(root, commit=None):
    root = Path(root)
    fails, metas = [], {}
    for a in ARMS:
        for t in SEEDS:
            for h in HEADS:
                p = root / "heads" / head_name(a, t, h) / "train_meta.json"
                if not p.exists():
                    fails.append(f"K4 {head_name(a, t, h)}: missing")
                    continue
                m = json.loads(p.read_text())
                metas[(a, t, h)] = m
                recipe, seed = h.split("_h")
                if (m.get("comem") != HEAD_LAMBDA[recipe] or m.get("seed") != int(seed) or m.get("m") != 20
                        or m.get("class_channels", "none") != "none" or m.get("git_dirty") is not False):
                    fails.append(f"K4 {head_name(a, t, h)}: comem {m.get('comem')} seed {m.get('seed')} m {m.get('m')} "
                                 f"dirty {m.get('git_dirty')}")
                if commit and m.get("git_commit") != commit:
                    fails.append(f"K4 {head_name(a, t, h)}: commit {str(m.get('git_commit'))[:7]} != trunks' "
                                 f"{str(commit)[:7]}")
    for h in HEADS:
        got = [metas[k] for k in metas if k[2] == h]
        if len({m.get("init_sha256") for m in got}) != 1:
            fails.append(f"K4 {h}: init_sha256 differs across trunks")
        by_n = {}                        # usable train events can differ by trunk (empty foreground): same n => same order
        for m in got:
            by_n.setdefault(m.get("n_train_events"), set()).add(json.dumps(m.get("perm_sha256")))
        for n_ev, perms in by_n.items():
            if len(perms) != 1:
                fails.append(f"K4 {h}: training order differs across trunks with {n_ev} usable events")
    return fails


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def eval_rows(root, a, t, head, dec):
    p = Path(root) / "eval" / trunk(a, t) / head / dec / "test_events.csv"
    return {(int(r["batch"]), int(r["event"])): r for r in read_csv(p)}


def health(root, a, t):
    sums, w = {}, 0
    with open(Path(root) / f"cache_test_{trunk(a, t)}" / "batch_metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            n = rec.get("n_graphs", 1)
            w += n
            for k, key in HEALTH.items():
                sums[k] = sums.get(k, 0.0) + fnum(rec.get(key)) * n
    return {k: sums.get(k, NAN) / w if w else NAN for k in HEALTH}


def mean_col(rows, col):
    v = [fnum(r[col]) for r in rows.values() if fin(fnum(r[col]))]
    return stat.mean(v) if v else NAN


def paired(x, y, col="ari_particle"):
    d = [fnum(y[k][col]) - fnum(x[k][col]) for k in sorted(set(x) & set(y))]
    d = [v for v in d if fin(v)]
    if len(d) < 2:
        return {"mean": NAN, "se": NAN, "n": len(d)}
    return {"mean": stat.mean(d), "se": stat.stdev(d) / math.sqrt(len(d)), "n": len(d)}


def shape(rows):
    ev = [r for r in rows.values() if fin(fnum(r["ari_particle"]))]
    ratio = [fnum(r["n_pred"]) / max(fnum(r["n_true_all"]), 1) for r in ev if fin(fnum(r["n_pred"]))]
    return {"ratio_median": stat.median(ratio) if ratio else NAN,
            "single_share": sum(1 for r in ev if fnum(r["n_pred"]) <= 1) / max(len(ev), 1)}


def evaluate(root):
    root = Path(root)
    fails, tinfo = train_gates(root)
    fails += head_gates(root, tinfo.get("commit"))
    for split in ("train", "devA", "test"):
        p = root / "gates" / f"K3_{split}.txt"
        txt = p.read_text().strip() if p.exists() else "missing"
        if not txt.startswith("PASS"):
            fails.append(f"K3 {split}: {txt[:120]}")
    data = {}
    for a in ARMS:
        for t in SEEDS:
            for h in HEADS + ("soft",):
                for dec in DECODES:
                    data[(a, t, h, dec)] = eval_rows(root, a, t, h, dec)
    for a in ARMS:                                                    # K5: same candidates within a trunk
        for t in SEEDS:
            ref = data[(a, t, "CMH_h1", "NM")]
            for h in HEADS + ("soft",):
                for dec in DECODES:
                    x = data[(a, t, h, dec)]
                    if set(x) != set(ref):
                        fails.append(f"K5 {trunk(a, t)} {h}/{dec}: different event set")
                        continue
                    bad = sum(1 for k in ref if not (fnum(ref[k]["ari_particle_hardseed"]) == fnum(x[k]["ari_particle_hardseed"])
                                                     or (math.isnan(fnum(ref[k]["ari_particle_hardseed"]))
                                                         and math.isnan(fnum(x[k]["ari_particle_hardseed"])))))
                    if bad:
                        fails.append(f"K5 {trunk(a, t)} {h}/{dec}: hard-seed ARI differs on {bad} events")
    hl = {(a, t): health(root, a, t) for a in ARMS for t in SEEDS}
    hdelta = {k: stat.mean(hl[("TCM", t)][k] - hl[("T", t)][k] for t in SEEDS) for k in HEALTH}
    if not (hdelta["semantic"] >= HEALTH_MIN and hdelta["event"] >= HEALTH_MIN):
        fails.append(f"K6: health TCM - T semantic {hdelta['semantic']:+.4f}, event {hdelta['event']:+.4f} "
                     f"(each must be >= {HEALTH_MIN})")
    hard_T = stat.mean(mean_col(data[("T", t, "CMH_h1", "NM")], "ari_particle_hardseed") for t in SEEDS)
    if not abs(hard_T - S1B_T_TEST_HARD) <= REPRO_TOL:
        fails.append(f"K7: T arms' test hard-seed ARI {hard_T:.4f} not within {REPRO_TOL} of Stage-1b T {S1B_T_TEST_HARD}")

    res = {"gates": {"pass": not fails, "failures": fails, "comem_last": tinfo["comem_last"],
                     "commit": tinfo.get("commit")},
           "health": {trunk(a, t): hl[(a, t)] for a in ARMS for t in SEEDS}, "health_delta": hdelta,
           "hard_T_mean": hard_T}
    # primary: per event, mean over (s, h) of TCM_s + CMH_h minus T_s + CMH_h, decode NM
    pairs = [(t, h) for t in SEEDS for h in PRIMARY_HEADS]
    keys = set.intersection(*(set(data[(a, t, h, "NM")]) for a in ARMS for t, h in pairs))
    per_event, per_seed, per_pair, ev_keys = [], {t: [] for t in SEEDS}, {}, []
    for k in sorted(keys):
        ds = {(t, h): fnum(data[("TCM", t, h, "NM")][k]["ari_particle"]) - fnum(data[("T", t, h, "NM")][k]["ari_particle"])
              for t, h in pairs}
        if all(fin(v) for v in ds.values()):
            ev_keys.append(k)
            per_event.append(stat.mean(ds.values()))
            for t in SEEDS:
                per_seed[t].append(stat.mean(ds[(t, h)] for h in PRIMARY_HEADS))
            for p, v in ds.items():
                per_pair.setdefault(p, []).append(v)
    n = len(per_event)
    delta = stat.mean(per_event) if n else NAN
    se = stat.stdev(per_event) / math.sqrt(n) if n > 1 else NAN
    d_seed = {t: stat.mean(per_seed[t]) for t in SEEDS}
    d_pair = {f"s{t}_{h}": stat.mean(v) for (t, h), v in per_pair.items()}
    guard = {}
    for t, h in pairs:
        sm, st_ = shape(data[("TCM", t, h, "NM")]), shape(data[("T", t, h, "NM")])
        guard[f"s{t}_{h}"] = {"TCM": sm, "T": st_, "ok": sm["ratio_median"] >= RATIO_MIN
                              and sm["single_share"] <= st_["single_share"] + SINGLE_SLACK}
    res["primary"] = {"delta": delta, "se": se, "n": n, "delta_seed": d_seed, "delta_pair": d_pair, "guard": guard}
    if fails:
        reading = "INVALID"
    elif (all(d_seed[t] >= ADOPT_SEED for t in SEEDS) and delta > ADOPT_SE * se
          and all(v > 0 for v in d_pair.values()) and all(g["ok"] for g in guard.values())):
        reading = "ADOPT"
    elif HARM < delta < HEAD_SUFF:
        reading = "HEAD-SUFFICIENT"
    elif delta <= HARM:
        reading = "HARMFUL"
    else:
        reading = "PARTIAL"
    res["reading"] = reading
    # secondary (reported, never decided on)
    sec = {}
    for t in SEEDS:
        T1, M1 = data[("T", t, "CMH_h1", "NM")], data[("TCM", t, "CMH_h1", "NM")]
        sec[t] = {"hard_seed": paired(T1, M1, "ari_particle_hardseed"),
                  "soft_NM": paired(data[("T", t, "soft", "NM")], data[("TCM", t, "soft", "NM")]),
                  "soft_DEP": paired(data[("T", t, "soft", "DEP")], data[("TCM", t, "soft", "DEP")]),
                  "CTH_NM": paired(data[("T", t, "CTH_h1", "NM")], data[("TCM", t, "CTH_h1", "NM")]),
                  "CMH_DEP": {h: paired(data[("T", t, h, "DEP")], data[("TCM", t, h, "DEP")]) for h in PRIMARY_HEADS}}
    res["secondary"] = sec
    res["levels"] = {f"{trunk(a, t)} {h} {dec}": mean_col(data[(a, t, h, dec)], "ari_particle")
                     for a in ARMS for t in SEEDS for h in HEADS + ("soft",) for dec in DECODES}
    nt = sorted(fnum(data[("T", 1, "CMH_h1", "NM")][k]["n_true_all"]) for k in keys)
    edges = [nt[len(nt) * i // 4] for i in (1, 2, 3)] if nt else []
    bins = {}
    for k, v in zip(ev_keys, per_event):
        bins.setdefault(sum(fnum(data[("T", 1, "CMH_h1", "NM")][k]["n_true_all"]) >= e for e in edges), []).append(v)
    res["delta_by_ntrue_quartile"] = [stat.mean(bins[b]) if b in bins else NAN for b in range(4)]
    res["mechanism"] = mechanism(root)
    return res


def mechanism(root):
    cm = _load("comem_report", HERE / "comem.py")
    out = {}
    for a in ARMS:
        for t in SEEDS:
            p = Path(root) / "rank" / f"{trunk(a, t)}_CMH_h1" / "rank_summary.json"
            if not p.exists():
                continue
            st = json.loads(p.read_text())["stats"]
            try:
                cats = cm.categories(st, "dec")
            except SystemExit:
                continue
            out[trunk(a, t)] = {c: cm.rates(cats[c]) for c in cm.CATS}
    return out


def f4(x):
    return f"{x:+.4f}" if fin(x) else "n/a"


def f4u(x):
    return f"{x:.4f}" if fin(x) else "n/a"


NEXT = {"ADOPT": "Stage-1c system = TCM trunk + CMH head + NM decode; a further deployable claim needs a new "
                 "evaluation source (ledger)",
        "HEAD-SUFFICIENT": "co-membership belongs in the head: system = Stage-1b T trunk + CMH head + NM decode",
        "HARMFUL": "reject trunk co-membership",
        "PARTIAL": "report; no adoption",
        "INVALID": "a gate failed; no reading"}


def render(res):
    P = res["primary"]
    L = ["# Stage 1c — trunk co-membership (Stage-1b T vs T + co-membership λ 0.3), decision on `test` (2nd read)", "",
         f"**Gates:** {'PASS' if res['gates']['pass'] else 'FAIL'} (commit {str(res['gates'].get('commit'))[:7]})"]
    L += [f"- {x}" for x in res["gates"]["failures"]]
    L += ["", f"**Reading:** {res['reading']} — {NEXT[res['reading']]}", "",
          "## Primary: test, decode NM, TCM ⊕ CMH − T ⊕ CMH (head seed matched), paired per event", "",
          f"Δ = {f4(P['delta'])} ± {f4u(P['se'])} (n = {P['n']}); per trunk seed: "
          + ", ".join(f"s{t} {f4(P['delta_seed'][t])}" for t in SEEDS) + "; per (seed, head): "
          + ", ".join(f"{k} {f4(v)}" for k, v in P["delta_pair"].items()), "",
          "| TCM ⊕ CMH vs T ⊕ CMH | median n_pred/n_true (TCM / T) | single-cluster share (TCM / T) | guard |",
          "|---|---|---|---|"]
    for k, g in P["guard"].items():
        L.append(f"| {k} | {f4u(g['TCM']['ratio_median'])} / {f4u(g['T']['ratio_median'])} | "
                 f"{f4u(g['TCM']['single_share'])} / {f4u(g['T']['single_share'])} | {g['ok']} |")
    L += ["", "Δ by quartile of n_true: " + ", ".join(f"Q{i + 1} {f4(v)}" for i, v in
                                                  enumerate(res["delta_by_ntrue_quartile"])), "",
          "## Gates detail", "",
          "| trunk | last-epoch training co-membership term | semantic recall | event recall | OC potential |",
          "|---|---|---|---|---|"]
    for a in ARMS:
        for t in SEEDS:
            h = res["health"][trunk(a, t)]
            L.append(f"| {trunk(a, t)} | {f4u(res['gates']['comem_last'].get(trunk(a, t), NAN))} | "
                     f"{f4u(h['semantic'])} | {f4u(h['event'])} | {f4u(h['potential'])} |")
    L += ["", f"Health TCM − T (mean over seeds): semantic {f4(res['health_delta']['semantic'])}, event "
          f"{f4(res['health_delta']['event'])}. T arms' test hard-seed ARI {f4u(res['hard_T_mean'])} "
          f"(Stage-1b T {S1B_T_TEST_HARD}).", "",
          "## Secondary (reported, never decided on): TCM − T, paired on test", "",
          "| seed | hard-seed decoder | joint soft layer NM | joint soft layer DEP | CTH head NM | CMH_h1 DEP | CMH_h2 DEP |",
          "|---|---|---|---|---|---|---|"]
    for t in SEEDS:
        s = res["secondary"][t]
        cell = lambda d: f"{f4(d['mean'])} ± {f4u(d['se'])}"
        L.append(f"| {t} | {cell(s['hard_seed'])} | {cell(s['soft_NM'])} | {cell(s['soft_DEP'])} | "
                 f"{cell(s['CTH_NM'])} | {cell(s['CMH_DEP']['CMH_h1'])} | {cell(s['CMH_DEP']['CMH_h2'])} |")
    L += ["", "## Absolute test ARI", "", "| trunk | " + " | ".join(f"{h} {d}" for h in HEADS + ("soft",)
                                                                   for d in DECODES) + " |",
          "|---" * (len(HEADS + ("soft",)) * len(DECODES) + 1) + "|"]
    for a in ARMS:
        for t in SEEDS:
            L.append(f"| {trunk(a, t)} | " + " | ".join(f4u(res["levels"][f"{trunk(a, t)} {h} {d}"])
                                                        for h in HEADS + ("soft",) for d in DECODES) + " |")
    L += ["", "Frozen references (Stage-1b final, test): "
          + "; ".join(f"{k} {' / '.join(map(str, v))}" for k, v in REFS.items()), ""]
    if res["mechanism"]:
        L += ["## Mechanism (dev-A, CMH_h1, decode level): co-assignment / false co-assignment", "",
              "| trunk | RR | RU | UU |", "|---|---|---|---|"]
        for k, v in res["mechanism"].items():
            L.append(f"| {k} | " + " | ".join(f"{f4u(v[c]['coassign'])} / {f4u(v[c]['false'])}"
                                              for c in ("RR", "RU", "UU")) + " |")
    return "\n".join(L) + "\n"


def cmd_smoke(a):
    fails, info = smoke_check(a.root)
    print(json.dumps(info))
    for f in fails:
        print("FAIL:", f)
    if fails:
        raise SystemExit(1)
    print("SMOKE PASS: resume identity 60/60, matched order, co-membership logged in both arms and trained only in TCM")


def cmd_train_gates(a):
    fails, info = train_gates(a.root)
    print(json.dumps(info))
    for f in fails:
        print("FAIL:", f)
    if fails:
        raise SystemExit(1)
    print("K1 / K2: PASS")


def cmd_head_gates(a):
    _, info = train_gates(a.root)
    fails = head_gates(a.root, info.get("commit"))
    for f in fails:
        print("FAIL:", f)
    if fails:
        raise SystemExit(1)
    print("K4: PASS")


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
    for name, fn in (("smoke-check", cmd_smoke), ("train-gates", cmd_train_gates), ("head-gates", cmd_head_gates)):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True)
        s.set_defaults(fn=fn)
    s = sub.add_parser("report")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
