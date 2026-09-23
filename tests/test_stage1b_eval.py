"""Tests for the Stage-1b evaluation driver (scripts/particle_hierarchy/stage1b_eval.py).

numpy + stdlib only (runs on GPVM too): synthetic tool outputs in the real per-event CSV formats of
soft_particle eval, mat_bench bench, stage0_seed_diag headroom/rules and mat_bench cache.
"""
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "stage1b_eval", ROOT / "scripts/particle_hierarchy/stage1b_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

N = 60                                                    # events per arm
KEYS = [(150 + i // 8, i % 8) for i in range(N)]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_arm(out, arm, split, hard, soft, seed05=None, db03=None, db02=None, n_groups=None,
             pred_true=1.0, beta_max=None, health=None, missing=()):
    """Synthetic outputs of every tool for one arm; values are per-event arrays of length N."""
    p = ev.arm_paths(out, arm)
    beta_max = np.full(N, 0.9) if beta_max is None else beta_max
    n_groups = np.full(N, 5) if n_groups is None else n_groups
    soft_rows, bench_rows = [], {ev.SEED_TAG: [], ev.DB03_TAG: [], ev.DB02_TAG: []}
    head_rows, rule_rows = [], []
    for i, (b, e) in enumerate(KEYS):
        if (b, e) in missing:
            continue
        soft_rows.append({"batch": b, "event": e, "ari_particle": soft[i],
                          "ari_particle_hardseed": hard[i], "n_groups": n_groups[i],
                          "beta_max": beta_max[i], "n_true_all": 8, "n_pred": 8 * pred_true,
                          "n_fg": 1000})
        for tag, arr in ((ev.SEED_TAG, seed05), (ev.DB03_TAG, db03), (ev.DB02_TAG, db02)):
            v = hard[i] if arr is None else arr[i]
            bench_rows[tag].append({"batch": b, "event": e, "ari_particle": v, "valid": True})
        head_rows.append({"batch": b, "event": e, "ari_particle_top3": 0.25, "ari_particle_top5": 0.3,
                          "ari_particle_topall": 0.6, "n_pop": 100, "n_unreachable": 60})
        rule_rows += [{"batch": b, "event": e, "rule": ev.RULE, "seed_recall": 0.5},
                      {"batch": b, "event": e, "rule": "fixed:0.5", "seed_recall": 0.4}]
    _write(p["soft"] / f"{split}_events.csv", soft_rows)
    for tag, rows in bench_rows.items():
        _write(p["bench"] / f"{tag}.csv", rows)
    _write(p["stage0"] / f"headroom_events_{ev.rule_fs(ev.RULE)}.csv", head_rows)
    _write(p["stage0"] / "rules_events.csv", rule_rows)
    h = {"semantic": 0.79, "event": 0.71, "potential": 0.47, "legacy_ari": 0.22, **(health or {})}
    p["cache"].mkdir(parents=True, exist_ok=True)
    with open(p["cache"] / "batch_metrics.jsonl", "w") as f:
        for bi in range(3):                                # three batches of 8 graphs
            f.write(json.dumps({"batch": 150 + bi, "n_graphs": 8, **{
                ev.HEALTH[k]: v for k, v in h.items()}}) + "\n")


def scenario(tmp_path, t_hard_gain=0.02, t_soft_gain=0.03, split="devB", **kw):
    rng = np.random.default_rng(0)
    base = 0.17 + 0.05 * rng.standard_normal(N)
    sbase = 0.25 + 0.05 * rng.standard_normal(N)
    noise = lambda: 0.005 * rng.standard_normal(N)
    for arm in ev.ARMS:
        t = arm.startswith("T")
        make_arm(tmp_path, arm, split,
                 hard=base + (t_hard_gain if t else 0) + noise(),
                 soft=sbase + (t_soft_gain if t else 0) + noise(),
                 **kw.get(arm, {}))
    return ev.cmd_report(type("A", (), {"out": str(tmp_path), "split": split, "runs": None,
                                        "stage1a_events": None})())


# --------------------------------------------------------------------------- #
def test_paired_statistic_matches_hand_computation():
    data = {a: {"x": {}} for a in ev.ARMS}
    vals = {"T_s1": [0.3, 0.5, float("nan"), 0.2], "C_s1": [0.1, 0.4, 0.3, 0.2],
            "T_s2": [0.4, 0.2, 0.1, 0.3], "C_s2": [0.2, 0.2, 0.1, 0.1]}
    for a, v in vals.items():
        data[a]["x"] = {(150, i): x for i, x in enumerate(v)}
    del data["C_s2"]["x"][(150, 3)]                               # missing in one arm
    r = ev.paired("x", data)
    d = np.array([0.5 * ((0.3 - 0.1) + (0.4 - 0.2)), 0.5 * ((0.5 - 0.4) + (0.2 - 0.2))])
    assert r["n"] == 2
    assert math.isclose(r["mean"], d.mean())
    assert math.isclose(r["se"], d.std(ddof=1) / math.sqrt(2))
    assert math.isclose(r["seed1"]["mean"], np.mean([0.2, 0.1]))


def test_all_criteria_pass_when_T_improves_latent_and_decoder(tmp_path):
    res = scenario(tmp_path)
    assert res["G"]["pass"] and res["R1"]["pass"] and res["R2"]["pass"] and res["S"]["pass"]
    assert res["R1"]["mean"] == pytest.approx(0.02, abs=0.004)
    assert res["interpretation"].startswith("G + R1 pass")
    assert res["decision"] is True
    assert (tmp_path / "report.md").exists() and (tmp_path / "report.json").exists()


def test_decoder_only_gain_is_R1_fail_S_pass(tmp_path):
    res = scenario(tmp_path, t_hard_gain=0.0)
    assert not res["R1"]["pass"] and res["S"]["pass"]
    assert res["interpretation"].startswith("R1 fails, S passes")


def test_no_gain_anywhere(tmp_path):
    res = scenario(tmp_path, t_hard_gain=0.0, t_soft_gain=0.0)
    assert not res["R1"]["pass"] and not res["S"]["pass"]
    assert res["interpretation"].startswith("R1 and S fail")


def test_health_gate_failure_overrides_everything(tmp_path):
    res = scenario(tmp_path, T_s1={"health": {"semantic": 0.76}}, T_s2={"health": {"semantic": 0.77}})
    assert res["G"]["semantic_T_minus_C"] == pytest.approx(-0.025)
    assert not res["G"]["pass"] and res["R1"]["pass"]
    assert res["interpretation"].startswith("G fails")


def test_potential_must_stay_bounded(tmp_path):
    res = scenario(tmp_path, T_s1={"health": {"potential": 2.0}}, T_s2={"health": {"potential": 2.0}})
    assert not res["G"]["potential_ok"] and not res["G"]["pass"]


def test_missing_health_metric_fails_the_gate(tmp_path):
    res = scenario(tmp_path, T_s2={"health": {"event": float("nan")}})
    assert not res["G"]["all_finite"] and not res["G"]["pass"]


def test_S_needs_C1_to_C4_on_each_T_seed(tmp_path):
    res = scenario(tmp_path, T_s2={"pred_true": 0.5})              # C4 fails on seed 2 only
    assert res["S"]["diff_pass"] and not res["S"]["pass"]
    assert res["S"]["C1_C4"]["T_s1"]["pass"] and not res["S"]["C1_C4"]["T_s2"]["C4"]
    groups = np.full(N, 5)
    groups[3] = 0
    res = scenario(tmp_path / "b", T_s1={"n_groups": groups})       # one empty event: C2 fails
    assert not res["S"]["C1_C4"]["T_s1"]["C2"] and not res["S"]["pass"]


def test_C3_uses_the_arms_own_fixed05_on_low_beta_events(tmp_path):
    low = np.full(N, 0.9)
    low[:10] = 0.3
    seed05 = np.full(N, 0.95)                                        # fixed:0.5 beats soft there
    res = scenario(tmp_path, T_s1={"beta_max": low, "seed05": seed05})
    c = res["S"]["C1_C4"]["T_s1"]
    assert c["C3_low_beta_n"] == 10 and c["C3_delta_vs_fixed05"] < 0 and not c["C3"]
    assert not res["S"]["pass"]


def test_R2_detects_a_degraded_fixed_decoder(tmp_path):
    rng = np.random.default_rng(1)
    bad = 0.10 + 0.01 * rng.standard_normal(N)
    res = scenario(tmp_path, T_s1={"db02": bad}, T_s2={"db02": bad})
    assert not res["R2"]["db02"]["pass"] and res["R2"]["seed05"]["pass"]
    assert not res["R2"]["pass"] and res["R1"]["pass"]
    assert "CAVEAT: R2 fails" in res["interpretation"]


def test_events_missing_in_one_arm_are_dropped_from_pairs(tmp_path):
    res = scenario(tmp_path, C_s2={"missing": {KEYS[0], KEYS[1]}})
    assert res["paired"]["hard"]["n"] == N - 2


def test_smoke_report_is_labelled_not_a_decision(tmp_path):
    res = scenario(tmp_path, split="smoke")
    assert res["decision"] is False
    assert "NOT a decision" in (tmp_path / "report.md").read_text()


def test_stage1a_reference_comparison(tmp_path):
    scenario(tmp_path)
    ref = tmp_path / "frozen_events.csv"
    _write(ref, [{"batch": b, "event": e, "ari_particle": 0.24} for b, e in KEYS])
    res = ev.cmd_report(type("A", (), {"out": str(tmp_path), "split": "devB", "runs": None,
                                       "stage1a_events": str(ref)})())
    r = res["stage1a_reference"]
    assert r["n"] == N and r["stage1a_frozen"] == pytest.approx(0.24)


# --------------------------------------------------------------------------- #
# inputs and the command plan
# --------------------------------------------------------------------------- #
def fake_runs(root, commit="a61a01b", lam=1.0, unfinished=None):
    for arm in ev.ARMS:
        d = root / arm
        d.mkdir(parents=True)
        (d / f"trunk_ep{ev.FINAL_EPOCH}.ckpt").write_bytes(b"trunk-" + arm.encode())
        (d / f"soft_ep{ev.FINAL_EPOCH}.pt").write_bytes(b"soft-" + arm.encode())
        man = {"started": "x", "config": {"arm": arm[0], "seed": int(arm[-1]), "lam": lam,
                                          "git_commit": commit, "git_dirty": False}}
        if arm != unfinished:
            man["finished"] = "y"
        (d / "manifest.json").write_text(json.dumps(man))
        _write(d / "history.csv", [{"epoch": e} for e in range(-1, ev.FINAL_EPOCH + 1)])
        _write(d / "steps.csv", [{"global_step": 1, "step_s": 2.4}])
    return root


def test_plan_prints_the_frozen_tool_commands_for_devB(tmp_path, capsys):
    runs = fake_runs(tmp_path / "runs")
    a = type("A", (), {"runs": str(runs), "out": str(tmp_path / "eval"), "split": "devB",
                       "data_path": "/data.h5", "device": "0", "arms": None})()
    ev.cmd_plan(a, execute=False)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[")]
    assert len(lines) == 4 * 7
    assert all("--batches 150:314" in ln for ln in lines)
    t1 = [ln for ln in lines if ln.startswith("[T_s1]")]
    assert "--checkpoint " + str(runs / "T_s1" / "trunk_ep4.ckpt") in t1[0] and "--store-hit-x" in t1[0]
    assert "--head " + str(runs / "T_s1" / "soft_ep4.pt") in t1[1]
    assert "--decode v2 --no-null --merge-thr 0.5" in t1[1] and "--tag devB" in t1[1]
    assert "--backends seed --t-beta 0.5 --t-d 0.3 --t-assign inf" in t1[2]
    assert "--dbscan-eps 0.3 --dbscan-min-samples 15" in t1[3]
    assert "--dbscan-eps 0.2 --dbscan-min-samples 15" in t1[4]
    assert "headroom" in t1[5] and f"--rule {ev.RULE}" in t1[5]
    assert f"--rules {ev.RULE} fixed:0.5" in t1[6]
    assert not (tmp_path / "eval").exists()                         # plan writes nothing


def test_smoke_split_stays_on_dev_A(tmp_path, capsys):
    runs = fake_runs(tmp_path / "runs")
    a = type("A", (), {"runs": str(runs), "out": str(tmp_path / "eval"), "split": "smoke",
                       "data_path": "/d.h5", "device": "0", "arms": ["C_s2"]})()
    ev.cmd_plan(a, execute=False)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[")]
    assert len(lines) == 7 and all("--batches 0:2" in ln for ln in lines)
    assert ev.SPLITS == {"devB": "150:314", "smoke": "0:2"}


@pytest.mark.parametrize("bad", [dict(unfinished="C_s2"), dict(commit=None)])
def test_runs_must_be_finished_and_consistent(tmp_path, bad):
    runs = fake_runs(tmp_path / "runs", **({} if "commit" in bad else bad))
    if "commit" in bad:                                           # one arm at another commit
        m = json.loads((runs / "T_s2" / "manifest.json").read_text())
        m["config"]["git_commit"] = "deadbee"
        (runs / "T_s2" / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SystemExit):
        ev.check_runs(runs)


def test_run_refuses_changed_trunks_midway(tmp_path, monkeypatch):
    runs = fake_runs(tmp_path / "runs")
    out = tmp_path / "eval"
    a = type("A", (), {"runs": str(runs), "out": str(out), "split": "smoke",
                       "data_path": "/d.h5", "device": "0", "arms": ["T_s1"]})()
    calls = []

    def fake_call(cmd, stdout=None, stderr=None):                  # every step "succeeds"
        calls.append(cmd)
        return 1                                                    # ...but reports failure
    monkeypatch.setattr(ev.subprocess, "call", fake_call)
    with pytest.raises(SystemExit):                                 # failed step stops the run
        ev.cmd_plan(a, execute=True)
    assert len(calls) == 1 and json.loads((out / "provenance.json").read_text())["split"] == "smoke"
    (runs / "T_s1" / "trunk_ep4.ckpt").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="changed"):
        ev.cmd_plan(a, execute=True)
