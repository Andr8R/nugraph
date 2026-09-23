#!/usr/bin/env python
"""stage1b_eval.py -- the pre-registered Stage-1b dev-B evaluation (design §5.3), run once.

For each of the four finished fine-tunes (T_s1, C_s1, T_s2, C_s2; last epoch 4) this re-caches the
trunk's outputs on the evaluation split and applies the FROZEN evaluation tools unchanged:

  cache      mat_bench.py cache (ox / beta / h.x + per-batch NuGraph metrics for the health gate)
  soft       soft_particle.py eval: the arm's own co-trained soft layer (decode v2, no null,
             merge 0.5) AND the hard seed decoder (frozen rule quant:0.995:0.05:0.5, t_d 0.3,
             t_assign inf) on the same events
  seed05     mat_bench.py bench, Thread-A seed backend tb0.5 / td0.3 / t_assign inf
  db03/db02  mat_bench.py bench, canonical (0.3/15) and tuned (0.2/15) DBSCAN
  headroom   stage0_seed_diag.py headroom (top-k / all-candidate oracle ARI, unreachable hits)
  rules      stage0_seed_diag.py rules (seed recall; fixed:0.5 alongside)

`report` then decides, exactly as pre-registered (paired per event, d = mean over seeds of T_s - C_s,
SE over events): G (health), R1 (hard seed decoder, T - C > 2 SE), R2 (seed tb0.5, canonical and
tuned DBSCAN, T - C >= -2 SE each), S (T soft - C soft > 2 SE and T soft meets C1-C4), and prints the
pre-registered interpretation. Clarifications fixed before dev-B was read are listed in CLARIFICATIONS.

  stage1b_eval.py plan   --runs $S1B --out $S1B/eval_devB --split devB --data-path ... --device 0
  stage1b_eval.py run    ...same...            (resumable: finished steps are skipped)
  stage1b_eval.py report --runs $S1B --out $S1B/eval_devB --split devB [--stage1a-events CSV]

`--split test` is the pre-registered FINAL evaluation on NuGraph's own test samples, with the same
criteria; `--ref-trunk/--ref-head` add the ep29 + frozen Stage-1a reference arm, reported beside the
four arms and never part of the criteria. `--split smoke` (dev-A batches 0:2) exercises the whole chain
without touching dev-B or test; its report is labelled as a mechanics check, never a decision.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = HERE / "soft_particle.py"
S0 = HERE / "stage0_seed_diag.py"
MB = HERE.parent / "dune_materialization" / "mat_bench.py"

ARMS = ("T_s1", "C_s1", "T_s2", "C_s2")
PAIRS = (("T_s1", "C_s1"), ("T_s2", "C_s2"))
FINAL_EPOCH = 4
RULE = "quant:0.995:0.05:0.5"
SPLITS = {                          # name -> NuGraph split, batch range, extra guard flags
    "devB": {"split": "validation", "batches": "150:314", "guard": []},
    "smoke": {"split": "validation", "batches": "0:2", "guard": []},          # dev-A, mechanics only
    # the pre-registered FINAL split: NuGraph's own test samples, never used in this workspace.
    # The dev-split guards of soft_particle/stage0 count validation batches, so they are waived here.
    "test": {"split": "test", "batches": "0:", "guard": ["--allow-heldout"]},
}
REF = "ref_ep29"                    # optional reference arm: ep29 trunk + frozen Stage-1a head
SEED_TAG = "seed_tb0.5_td0.3_tainf"
DB03_TAG = "dbscan_eps0.3_min15"
DB02_TAG = "dbscan_eps0.2_min15"
HEALTH = {"semantic": "semantic/recall-test", "event": "event/recall-test",
          "potential": "instance/potential-loss-test", "legacy_ari": "instance/adjusted-rand-test"}
NAN = float("nan")

CLARIFICATIONS = [
    "Final epoch = 4 (the pre-registered 5 epochs, last epoch evaluated) for every arm.",
    "Paired statistic: per event d = mean over s=1,2 of (T_s - C_s), on events with a finite value in "
    "all four arms; SE = sd(d)/sqrt(n). Per-seed differences are reported alongside.",
    "Particle ARI of every arm is computed over that arm's own predicted foreground (as in every earlier "
    "stage); foreground sizes per arm are reported.",
    "S / C1-C4 apply to each T seed's soft output separately; both T seeds must meet C1-C4.",
    "C3 compares each T arm's soft output with fixed:0.5 on the SAME latent, i.e. that arm's seed tb0.5 "
    "backend (identical algorithm to fixed:0.5), on the arm's own beta_max <= 0.5 events; the ep29 "
    "fixed:0.5 reference is reported, not gating.",
    "G on dev-B: T - C (mean over seeds) semantic and event recall >= -0.01 each; every reported per-event "
    "and health value finite; OC potential 'bounded' = finite and T <= 1.5 x C (seed means).",
    "Size bins use C_s1's n_true per event and beta_max bins use C_s1's beta_max (common reference).",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def keyed(rows, col, cond=None):
    out = {}
    for r in rows:
        if cond is not None and not cond(r):
            continue
        out[(int(float(r["batch"])), int(float(r["event"])))] = fnum(r.get(col))
    return out


def rule_fs(rule):
    return rule.replace(":", "_")


def split_cfg(split):
    return SPLITS[split]["split"], SPLITS[split]["batches"], SPLITS[split]["guard"]


def arm_paths(out, arm):
    d = Path(out) / arm
    return {"dir": d, "cache": d / "cache", "soft": d / "soft", "bench": d / "bench",
            "stage0": d / "stage0", "logs": d / "logs"}


# --------------------------------------------------------------------------- #
# inputs: the four finished runs
# --------------------------------------------------------------------------- #
def check_runs(runs):
    """Every arm finished all epochs at the same commit and lambda; returns provenance."""
    prov, commits, lams = {}, set(), set()
    for arm in ARMS:
        d = Path(runs) / arm
        man_p, hist_p = d / "manifest.json", d / "history.csv"
        trunk, soft = d / f"trunk_ep{FINAL_EPOCH}.ckpt", d / f"soft_ep{FINAL_EPOCH}.pt"
        for p in (man_p, hist_p, trunk, soft):
            if not p.exists():
                raise SystemExit(f"{arm}: missing {p}")
        man = json.loads(man_p.read_text())
        if "finished" not in man:
            raise SystemExit(f"{arm}: run not finished ({man_p} has no 'finished')")
        hist = read_csv(hist_p)
        if not hist or int(float(hist[-1]["epoch"])) != FINAL_EPOCH:
            raise SystemExit(f"{arm}: history does not end at epoch {FINAL_EPOCH}")
        cfg = man.get("config", {})
        want_arm, want_seed = arm[0], int(arm.split("_s")[1])
        if cfg.get("arm") != want_arm or int(cfg.get("seed", -1)) != want_seed:
            raise SystemExit(f"{arm}: manifest says arm {cfg.get('arm')} seed {cfg.get('seed')}")
        if cfg.get("git_dirty"):
            raise SystemExit(f"{arm}: trained from a dirty tree")
        commits.add(cfg.get("git_commit"))
        lams.add(cfg.get("lam"))
        prov[arm] = {"trunk": str(trunk), "trunk_sha256": sha256(trunk), "soft": str(soft),
                     "soft_sha256": sha256(soft), "git_commit": cfg.get("git_commit"),
                     "lam": cfg.get("lam"), "seed": want_seed, "started": man.get("started"),
                     "finished": man.get("finished"), "resumes": len(man.get("resumes", []))}
    if len(commits) != 1:
        raise SystemExit(f"arms were trained at different commits: {sorted(map(str, commits))}")
    if len(lams) != 1:
        raise SystemExit(f"arms were trained with different lambda: {sorted(map(str, lams))}")
    return prov


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def step_plan(out, arm, trunk, soft, split, data_path, device, py=sys.executable):
    """Ordered (name, argv, done_check) for one arm."""
    nusplit, rng, guard = split_cfg(split)
    tag, p = split, arm_paths(out, arm)
    dev = str(device)
    bench = [py, str(MB), "bench", "--cache", str(p["cache"]), "--out", str(p["bench"]),
             "--batches", rng, "--device", dev]

    def bench_done(t):
        j = p["bench"] / f"{t}.json"
        return lambda: j.exists() and json.loads(j.read_text()).get("complete") is True

    def resume_flag(t):
        return ["--resume"] if (p["bench"] / f"{t}.csv").exists() else []

    return [
        ("cache", lambda: [py, str(MB), "cache", "--checkpoint", str(trunk), "--data-path",
                           str(data_path), "--out", str(p["cache"]), "--split", nusplit,
                           "--batches", rng, "--store-hit-x", "--batch-size", "8", "--device", dev],
         lambda: (p["cache"] / "manifest.json").exists()
         and "finished" in json.loads((p["cache"] / "manifest.json").read_text())),
        ("soft", lambda: [py, str(SP), "eval", "--head", str(soft), "--cache", str(p["cache"]),
                          "--batches", rng, "--out", str(p["soft"]), "--tag", tag,
                          "--decode", "v2", "--no-null", "--merge-thr", "0.5", "--device", dev]
                         + guard,
         lambda: (p["soft"] / f"{tag}_summary.json").exists()),
        ("seed05", lambda: bench + ["--backends", "seed", "--t-beta", "0.5", "--t-d", "0.3",
                                    "--t-assign", "inf"] + resume_flag(SEED_TAG),
         bench_done(SEED_TAG)),
        ("db03", lambda: bench + ["--backends", "dbscan", "--dbscan-eps", "0.3",
                                  "--dbscan-min-samples", "15"] + resume_flag(DB03_TAG),
         bench_done(DB03_TAG)),
        ("db02", lambda: bench + ["--backends", "dbscan", "--dbscan-eps", "0.2",
                                  "--dbscan-min-samples", "15"] + resume_flag(DB02_TAG),
         bench_done(DB02_TAG)),
        ("headroom", lambda: [py, str(S0), "headroom", "--cache", str(p["cache"]), "--out",
                              str(p["stage0"]), "--batches", rng, "--rule", RULE, "--device", dev]
                             + guard,
         lambda: (p["stage0"] / f"headroom_events_{rule_fs(RULE)}.csv").exists()),
        ("rules", lambda: [py, str(S0), "rules", "--cache", str(p["cache"]), "--out",
                           str(p["stage0"]), "--batches", rng, "--rules", RULE, "fixed:0.5",
                           "--device", dev] + guard,
         lambda: (p["stage0"] / "rules_events.csv").exists()),
    ]


def cmd_plan(a, execute=False):
    if a.split not in SPLITS:
        raise SystemExit(f"--split must be one of {sorted(SPLITS)}")
    prov = check_runs(a.runs)
    ref_trunk, ref_head = getattr(a, "ref_trunk", None), getattr(a, "ref_head", None)
    if bool(ref_trunk) != bool(ref_head):
        raise SystemExit("--ref-trunk and --ref-head must be given together")
    if ref_trunk:                       # reported reference, never part of the criteria
        prov[REF] = {"trunk": str(ref_trunk), "trunk_sha256": sha256(ref_trunk),
                     "soft": str(ref_head), "soft_sha256": sha256(ref_head), "reference": True}
    out = Path(a.out)
    arms = a.arms or (list(ARMS) + ([REF] if ref_trunk else []))
    if execute:
        out.mkdir(parents=True, exist_ok=True)
        pfile = out / "provenance.json"
        rec = json.loads(pfile.read_text()) if pfile.exists() else {}
        if rec.get("runs") and any(rec["runs"][k]["trunk_sha256"] != prov[k]["trunk_sha256"]
                                   or rec["runs"][k]["soft_sha256"] != prov[k]["soft_sha256"]
                                   for k in prov):
            raise SystemExit(f"{pfile}: the trunks/soft layers changed since this evaluation began")
        rec.update({"split": a.split, "batches": SPLITS[a.split]["batches"],
                    "nugraph_split": SPLITS[a.split]["split"], "runs": prov,
                    "driver_sha256": sha256(__file__), "tools": {str(t): sha256(t) for t in
                                                                (SP, S0, MB) if t.exists()},
                    "repo": _git(HERE)})
        rec.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
        pfile.write_text(json.dumps(rec, indent=2))
    for arm in arms:
        for name, argv, done in step_plan(out, arm, prov[arm]["trunk"], prov[arm]["soft"],
                                          a.split, a.data_path, a.device):
            if done():
                print(f"[{arm}] {name}: done, skipped")
                continue
            cmd = argv()
            print(f"[{arm}] {name}: " + " ".join(cmd), flush=True)
            if not execute:
                continue
            logs = arm_paths(out, arm)["logs"]
            logs.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(logs / f"{name}.log", "a") as fh:
                fh.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n")
                fh.flush()
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
            if rc != 0 or not done():
                raise SystemExit(f"[{arm}] {name} failed (exit {rc}); see {logs / (name + '.log')}")
            print(f"[{arm}] {name}: ok ({time.time() - t0:.0f} s)", flush=True)


def _git(path):
    run = lambda *x: subprocess.run(["git", "-C", str(path), *x], capture_output=True,
                                    text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--short"))}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def load_arm(out, arm, split):
    """Per-event metrics {name: {(batch, event): value}} and health means for one arm."""
    p = arm_paths(out, arm)
    soft = read_csv(p["soft"] / f"{split}_events.csv")
    bench = lambda t: read_csv(p["bench"] / f"{t}.csv")
    valid = lambda r: str(r.get("valid")) == "True"
    head = read_csv(p["stage0"] / f"headroom_events_{rule_fs(RULE)}.csv")
    rules = read_csv(p["stage0"] / "rules_events.csv")
    unreach = {}
    for r in head:
        n = fnum(r.get("n_pop"))
        unreach[(int(float(r["batch"])), int(float(r["event"])))] = (
            fnum(r.get("n_unreachable")) / n if n and n > 0 else NAN)
    ratio = {}
    for r in soft:
        nt = fnum(r.get("n_true_all"))
        ratio[(int(float(r["batch"])), int(float(r["event"])))] = (
            fnum(r.get("n_pred")) / nt if nt and nt > 0 else NAN)
    m = {"hard": keyed(soft, "ari_particle_hardseed"), "soft": keyed(soft, "ari_particle"),
         "n_groups": keyed(soft, "n_groups"), "beta_max": keyed(soft, "beta_max"),
         "n_true": keyed(soft, "n_true_all"), "n_fg": keyed(soft, "n_fg"), "pred_true": ratio,
         "seed05": keyed(bench(SEED_TAG), "ari_particle", valid),
         "db03": keyed(bench(DB03_TAG), "ari_particle", valid),
         "db02": keyed(bench(DB02_TAG), "ari_particle", valid),
         "oracle_top3": keyed(head, "ari_particle_top3"),
         "oracle_top5": keyed(head, "ari_particle_top5"),
         "oracle_all": keyed(head, "ari_particle_topall"), "unreachable": unreach,
         "seed_recall": keyed(rules, "seed_recall", lambda r: r.get("rule") == RULE)}
    sums, w = {}, 0
    with open(p["cache"] / "batch_metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            n = rec.get("n_graphs", 1)
            w += n
            for k, key in HEALTH.items():
                sums[k] = sums.get(k, 0.0) + fnum(rec.get(key)) * n
    health = {k: sums.get(k, NAN) / w if w else NAN for k in HEALTH}
    return m, health


def paired(metric, data):
    """d_e = mean over seeds (T_s - C_s) on events finite in all four arms."""
    import numpy as np
    keys = set.intersection(*(set(data[a][metric]) for a in ARMS))
    keys = sorted(k for k in keys if all(math.isfinite(data[a][metric][k]) for a in ARMS))
    if not keys:
        return {"n": 0, "mean": NAN, "se": NAN}
    v = {a: np.array([data[a][metric][k] for k in keys]) for a in ARMS}
    d = 0.5 * ((v["T_s1"] - v["C_s1"]) + (v["T_s2"] - v["C_s2"]))
    se = lambda x: float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else NAN
    out = {"n": len(keys), "mean": float(d.mean()), "se": se(d),
           "T_mean": float(0.5 * (v["T_s1"] + v["T_s2"]).mean()),
           "C_mean": float(0.5 * (v["C_s1"] + v["C_s2"]).mean()),
           "frac_T_better": float(np.mean(d > 1e-12))}
    for i, (t, c) in enumerate(PAIRS, 1):
        ds = v[t] - v[c]
        out[f"seed{i}"] = {"mean": float(ds.mean()), "se": se(ds), "T": float(v[t].mean()),
                           "C": float(v[c].mean())}
    return out


def soft_criteria(m):
    """C1-C4 for one T arm's soft output (C3 against fixed:0.5 = seed tb0.5 on the same latent)."""
    import numpy as np
    soft = np.array([v for v in m["soft"].values() if math.isfinite(v)])
    groups = np.array([v for v in m["n_groups"].values() if math.isfinite(v)])
    ratio = np.array([v for v in m["pred_true"].values() if math.isfinite(v)])
    low = [k for k, b in m["beta_max"].items() if math.isfinite(b) and b <= 0.5
           and math.isfinite(m["soft"].get(k, NAN)) and math.isfinite(m["seed05"].get(k, NAN))]
    dl = np.array([m["soft"][k] - m["seed05"][k] for k in low])
    c = {"C1_ari": float(soft.mean()) if soft.size else NAN,
         "C2_no_cluster": float(np.mean(groups == 0)) if groups.size else NAN,
         "C3_low_beta_n": len(low),
         "C3_delta_vs_fixed05": float(dl.mean()) if dl.size else NAN,
         "C3_se": float(dl.std(ddof=1) / math.sqrt(dl.size)) if dl.size > 1 else NAN,
         "C4_pred_true_med": float(np.median(ratio)) if ratio.size else NAN}
    c["C1"] = c["C1_ari"] >= 0.20
    c["C2"] = c["C2_no_cluster"] == 0
    c["C3"] = (not low) or c["C3_delta_vs_fixed05"] >= 0      # no low-beta events: vacuous
    c["C4"] = 0.8 <= c["C4_pred_true_med"] <= 1.5
    c["pass"] = bool(c["C1"] and c["C2"] and c["C3"] and c["C4"])
    return c


def evaluate(data, health):
    import numpy as np
    res = {"paired": {k: paired(k, data) for k in
                      ("hard", "soft", "seed05", "db03", "db02", "oracle_top3", "oracle_top5",
                       "oracle_all", "unreachable", "seed_recall")}}
    P = res["paired"]
    # G: health gate on dev-B
    tmc = lambda k: 0.5 * ((health["T_s1"][k] - health["C_s1"][k])
                           + (health["T_s2"][k] - health["C_s2"][k]))
    pot_t = 0.5 * (health["T_s1"]["potential"] + health["T_s2"]["potential"])
    pot_c = 0.5 * (health["C_s1"]["potential"] + health["C_s2"]["potential"])
    finite_events = all(math.isfinite(v) for a in ARMS for k in ("hard", "soft")
                        for v in data[a][k].values() if v == v) and all(
        math.isfinite(health[a][k]) for a in ARMS for k in HEALTH)
    g = {"semantic_T_minus_C": tmc("semantic"), "event_T_minus_C": tmc("event"),
         "potential_T": pot_t, "potential_C": pot_c, "all_finite": finite_events}
    g["semantic_ok"] = g["semantic_T_minus_C"] >= -0.01
    g["event_ok"] = g["event_T_minus_C"] >= -0.01
    g["potential_ok"] = math.isfinite(pot_t) and math.isfinite(pot_c) and pot_t <= 1.5 * pot_c
    g["pass"] = bool(g["semantic_ok"] and g["event_ok"] and g["potential_ok"] and g["all_finite"])
    res["G"] = g
    res["R1"] = {"pass": bool(P["hard"]["n"] > 1 and P["hard"]["mean"] > 2 * P["hard"]["se"]),
                 **P["hard"]}
    res["R2"] = {k: {"pass": bool(P[k]["n"] > 1 and P[k]["mean"] >= -2 * P[k]["se"]), **P[k]}
                 for k in ("seed05", "db03", "db02")}
    res["R2"]["pass"] = all(res["R2"][k]["pass"] for k in ("seed05", "db03", "db02"))
    crit = {t: soft_criteria(data[t]) for t in ("T_s1", "T_s2")}
    s_diff = bool(P["soft"]["n"] > 1 and P["soft"]["mean"] > 2 * P["soft"]["se"])
    res["S"] = {"diff_pass": s_diff, **P["soft"], "C1_C4": crit,
                "pass": bool(s_diff and all(c["pass"] for c in crit.values()))}
    # reported, not gating: health values, foreground sizes, size and beta_max bins
    res["health"] = health
    res["foreground_mean"] = {a: float(np.nanmean(list(data[a]["n_fg"].values()) or [NAN]))
                              for a in ARMS}
    res["by_size"] = binned(data, "n_true", [(1, 3), (3, 6), (6, 10), (10, 20), (20, 10**9)])
    res["by_beta_max"] = binned(data, "beta_max", [(0.0, 0.5000001), (0.5000001, 2.0)])
    res["interpretation"] = interpret(res)
    return res


def binned(data, by, edges):
    """R1 / S paired differences within bins of C_s1's `by` value."""
    out = {}
    for lo, hi in edges:
        keys = {k for k, v in data["C_s1"][by].items() if math.isfinite(v) and lo <= v < hi}
        sub = {a: {m: {k: data[a][m][k] for k in keys if k in data[a][m]} for m in ("hard", "soft")}
               for a in ARMS}
        out[f"[{lo:g},{hi:g})"] = {m: paired(m, sub) for m in ("hard", "soft")}
    return out


def interpret(res):
    if not res["G"]["pass"]:
        return ("G fails on dev-B: the fine-tuned trunk's health is not preserved; per the "
                "pre-registration no Stage-1b effect is claimed.")
    if res["R1"]["pass"]:
        tail = ("" if res["R2"]["pass"] else " CAVEAT: R2 fails (a fixed decoder degrades: " +
                ", ".join(k for k in ("seed05", "db03", "db02") if not res["R2"][k]["pass"]) + ").")
        return ("G + R1 pass: the particle-level signal improves the reconstruction itself "
                "(the fixed hard seed decoder is better on the T latent); report it as a trunk-level "
                "result and consider Stage 1c." + tail)
    if res["S"]["pass"]:
        return ("R1 fails, S passes: the gain is in the jointly trained decoder only "
                "(materialization), not the latent.")
    return ("R1 and S fail: end-to-end gradient adds nothing over Stage 1a on this trunk; the next "
            "lever is candidate generation / coverage (59.8 % of hits unreachable).")


def stage1a_reference(data, path):
    """T soft output (mean over seeds) vs the frozen Stage-1a head on the ep29 latent (reported)."""
    import numpy as np
    ref = keyed(read_csv(path), "ari_particle")
    keys = sorted(k for k in set(ref) & set(data["T_s1"]["soft"]) & set(data["T_s2"]["soft"])
                  if all(math.isfinite(x) for x in (ref[k], data["T_s1"]["soft"][k],
                                                    data["T_s2"]["soft"][k])))
    if len(keys) < 2:
        return None
    t = np.array([0.5 * (data["T_s1"]["soft"][k] + data["T_s2"]["soft"][k]) for k in keys])
    r = np.array([ref[k] for k in keys])
    d = t - r
    return {"n": len(keys), "T_soft": float(t.mean()), "stage1a_frozen": float(r.mean()),
            "mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(d.size))}


def _mean_on(metric, keys):
    import numpy as np
    v = [metric[k] for k in keys if k in metric and math.isfinite(metric[k])]
    return float(np.mean(v)) if v else NAN


def _paired_vs_ref(data, ref_metric, m, keys):
    """T (mean over seeds) - reference, paired per event (reported, never gating)."""
    import numpy as np
    ref = ref_metric
    ks = sorted(k for k in keys if k in ref and math.isfinite(ref[k])
                and all(math.isfinite(data[x][m].get(k, NAN)) for x in ("T_s1", "T_s2")))
    if len(ks) < 2:
        return {"n": len(ks), "mean": NAN, "se": NAN}
    d = np.array([0.5 * (data["T_s1"][m][k] + data["T_s2"][m][k]) - ref[k] for k in ks])
    return {"n": len(ks), "mean": float(d.mean()),
            "se": float(d.std(ddof=1) / math.sqrt(d.size))}


def cost_per_step(runs):
    out = {}
    for arm in ARMS:
        rows = read_csv(Path(runs) / arm / "steps.csv")
        t = [fnum(r.get("step_s")) for r in rows]
        out[arm] = {"steps": len(rows), "step_s_mean": sum(t) / len(t) if t else NAN}
    return out


def fmt(x, n=4):
    return "nan" if not isinstance(x, (int, float)) or not math.isfinite(x) else f"{x:.{n}f}"


def render(res, split):
    L = [f"# Stage 1b evaluation — {split} ({SPLITS[split]['split']} {SPLITS[split]['batches']})", ""]
    if split == "smoke":
        L += ["**SMOKE RUN on dev-A: mechanics check only, NOT a decision.**", ""]
    elif split == "test":
        L += ["**FINAL pre-registered evaluation on NuGraph's test split (never used before).**", ""]
    ok = lambda b: "PASS" if b else "FAIL"
    g = res["G"]
    L += ["## Decision (pre-registered, design §5.3)", "",
          f"- **G** {ok(g['pass'])}: semantic recall T−C {fmt(g['semantic_T_minus_C'])} (≥ −0.01), "
          f"event recall T−C {fmt(g['event_T_minus_C'])} (≥ −0.01), OC potential T {fmt(g['potential_T'])} "
          f"/ C {fmt(g['potential_C'])} (T ≤ 1.5·C), all finite {g['all_finite']}",
          f"- **R1** {ok(res['R1']['pass'])}: hard seed decoder T−C {fmt(res['R1']['mean'])} ± "
          f"{fmt(res['R1']['se'])} (n = {res['R1']['n']}; needs > 2 SE)",
          f"- **R2** {ok(res['R2']['pass'])}: " + "; ".join(
              f"{k} T−C {fmt(res['R2'][k]['mean'])} ± {fmt(res['R2'][k]['se'])} {ok(res['R2'][k]['pass'])}"
              for k in ("seed05", "db03", "db02")) + " (each ≥ −2 SE)",
          f"- **S** {ok(res['S']['pass'])}: soft T−C {fmt(res['S']['mean'])} ± {fmt(res['S']['se'])} "
          f"({ok(res['S']['diff_pass'])}); C1–C4: " + "; ".join(
              f"{t} C1 {fmt(c['C1_ari'])} C2 {fmt(c['C2_no_cluster'], 3)} C3 {fmt(c['C3_delta_vs_fixed05'])} "
              f"(n={c['C3_low_beta_n']}) C4 {fmt(c['C4_pred_true_med'], 2)} → {ok(c['pass'])}"
              for t, c in res["S"]["C1_C4"].items()), "",
          f"**Interpretation:** {res['interpretation']}", "",
          "## Paired T − C (mean over seeds; per seed)", "",
          "| metric | T | C | T−C ± SE | seed 1 | seed 2 | n |", "|---|---|---|---|---|---|---|"]
    for k, v in res["paired"].items():
        if not v.get("n"):
            continue
        L.append(f"| {k} | {fmt(v['T_mean'])} | {fmt(v['C_mean'])} | {fmt(v['mean'])} ± {fmt(v['se'])} | "
                 f"{fmt(v['seed1']['mean'])} | {fmt(v['seed2']['mean'])} | {v['n']} |")
    L += ["", "## Trunk health (event-weighted, NuGraph test-stage metrics)", "",
          "| arm | semantic recall | event recall | OC potential | legacy ARI | mean foreground hits |",
          "|---|---|---|---|---|---|"]
    for a in ARMS:
        h = res["health"][a]
        L.append(f"| {a} | {fmt(h['semantic'])} | {fmt(h['event'])} | {fmt(h['potential'])} | "
                 f"{fmt(h['legacy_ari'])} | {fmt(res['foreground_mean'][a], 1)} |")
    for title, key in (("true multiplicity (C_s1 n_true)", "by_size"), ("beta_max (C_s1)", "by_beta_max")):
        L += ["", f"## By {title}", "", "| bin | hard T−C ± SE | soft T−C ± SE | n |", "|---|---|---|---|"]
        for b, v in res[key].items():
            L.append(f"| {b} | {fmt(v['hard']['mean'])} ± {fmt(v['hard']['se'])} | "
                     f"{fmt(v['soft']['mean'])} ± {fmt(v['soft']['se'])} | {v['hard']['n']} |")
    if res.get("stage1a_reference"):
        r = res["stage1a_reference"]
        L += ["", f"T soft output (mean over seeds) {fmt(r['T_soft'])} vs frozen Stage-1a head on the ep29 "
              f"latent {fmt(r['stage1a_frozen'])}: {fmt(r['mean'])} ± {fmt(r['se'])} (n = {r['n']}; reported)"]
    if res.get("reference_ep29"):
        L += ["", "## Reference: ep29 trunk (frozen Stage-1a head for 'soft'), same events", "",
              "| metric | ep29 | T (mean over seeds) | T − ep29 ± SE | n |", "|---|---|---|---|---|"]
        for m in ("hard", "soft", "seed05", "db03", "db02"):
            r = res["reference_ep29"][m]
            L.append(f"| {m} | {fmt(r['mean'])} | {fmt(r['T_mean'])} | {fmt(r['T_minus_ref']['mean'])} ± "
                     f"{fmt(r['T_minus_ref']['se'])} | {r['T_minus_ref']['n']} |")
    if res.get("cost"):
        L += ["", "Cost per training step (in-step seconds): " + ", ".join(
            f"{a} {fmt(v['step_s_mean'], 2)}" for a, v in res["cost"].items())]
    L += ["", "## Clarifications fixed before dev-B was read", ""] + [f"- {c}" for c in CLARIFICATIONS]
    return "\n".join(L) + "\n"


def cmd_report(a):
    out = Path(a.out)
    data, health = {}, {}
    for arm in ARMS:
        data[arm], health[arm] = load_arm(out, arm, a.split)
    res = evaluate(data, health)
    if a.stage1a_events:
        res["stage1a_reference"] = stage1a_reference(data, a.stage1a_events)
    if a.runs:
        res["cost"] = cost_per_step(a.runs)
    ref_dir = out / REF
    if (ref_dir / "soft" / f"{a.split}_events.csv").exists():
        ref, ref_health = load_arm(out, REF, a.split)
        keys = set.intersection(*(set(data[x]["soft"]) for x in ARMS))
        res["reference_ep29"] = {
            m: {"mean": _mean_on(ref[m], keys), "T_mean": res["paired"][m]["T_mean"],
                "T_minus_ref": _paired_vs_ref(data, ref[m], m, keys)}
            for m in ("hard", "soft", "seed05", "db03", "db02")}
        res["reference_ep29"]["health"] = ref_health
    res.update({"split": a.split, "batches": SPLITS[a.split]["batches"],
                "nugraph_split": SPLITS[a.split]["split"], "decision": a.split in ("devB", "test"),
                "clarifications": CLARIFICATIONS,
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (out / "report.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "report.md").write_text(render(res, a.split))
    print(render(res, a.split))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        s = sub.add_parser(name)
        s.add_argument("--runs", required=True, help="stage1b dir holding T_s1 C_s1 T_s2 C_s2")
        s.add_argument("--out", required=True)
        s.add_argument("--split", required=True, choices=sorted(SPLITS))
        s.add_argument("--data-path", required=True)
        s.add_argument("--device", default="0")
        s.add_argument("--arms", nargs="+", choices=list(ARMS) + [REF], default=None)
        s.add_argument("--ref-trunk", default=None,
                       help="reference trunk (ep29) evaluated alongside; never part of the criteria")
        s.add_argument("--ref-head", default=None, help="frozen Stage-1a head for the reference arm")
    s = sub.add_parser("report")
    s.add_argument("--out", required=True)
    s.add_argument("--split", required=True, choices=sorted(SPLITS))
    s.add_argument("--runs", default=None, help="stage1b dir (for cost per step)")
    s.add_argument("--stage1a-events", default=None,
                   help="frozen Stage-1a per-event CSV on the same split (reported comparison)")
    a = p.parse_args(argv)
    if a.cmd == "report":
        return cmd_report(a)
    return cmd_plan(a, execute=(a.cmd == "run"))


if __name__ == "__main__":
    main()
