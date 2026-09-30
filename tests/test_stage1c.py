"""Tests for the Stage 1c gates and report (scripts/particle_hierarchy/stage1c.py). Pure python.

Run:  pytest -q tests/test_stage1c.py
"""
import csv
import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage1c", ROOT / "scripts/particle_hierarchy/stage1c.py")
s1c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1c)
N_EV = 240
EVAL_F = ["batch", "event", "ari_particle", "ari_particle_hardseed", "n_pred", "n_true_all"]


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build(tmp, gain=(0.03, 0.03), pair_gain=None, comem_last=None, config_over=None, steps_over=None,
          head_over=None, k3=None, health_delta=0.0, hard_T=0.19, hard_shift=None, ratio=None, finished=True):
    """A synthetic Stage-1c root. gain[s] = TCM - T for the CMH heads of trunk seed s (NM)."""
    root = tmp / "stage1c"
    steps = [{"global_step": i + 1, "epoch": i // 10, "batch": i % 10, "ids_hash": f"h{i}", "loss": 1.0}
             for i in range(50)]
    cl = {"T": 2.0, "TCM": 1.5, **(comem_last or {})}
    for a in s1c.ARMS:
        for t in s1c.SEEDS:
            d = root / "trunks" / f"{a}_s{t}"
            d.mkdir(parents=True)
            cfg = {"arm": "T", "seed": t, "epochs": 5, "comem": s1c.LAMBDA[a], "comem_log": True, "comem_pairs": 2048,
                   "git_commit": "abc", "git_dirty": False, "lam": 1.0}
            cfg.update((config_over or {}).get((a, t), {}))
            man = {"config": cfg, **({"finished": "x"} if finished else {})}
            (d / "manifest.json").write_text(json.dumps(man))
            st = [dict(r) for r in steps]
            if steps_over and (a, t) in steps_over:
                st[3]["ids_hash"] = "other"
            write_csv(d / "steps.csv", st)
            write_csv(d / "history.csv", [{"epoch": e, "train/comem": cl[a] + (4 - e) * 0.1} for e in range(-1, 5)])
            (d / "trunk_ep4.ckpt").write_text("x")
            (d / "soft_ep4.pt").write_text("x")
            with open(_mk(root / f"cache_test_{a}_s{t}"), "w") as f:
                sem = 0.79 + (health_delta if a == "TCM" else 0.0)
                f.write(json.dumps({"n_graphs": 8, "semantic/recall-test": sem, "event/recall-test": 0.70,
                                    "instance/potential-loss-test": 0.46}) + "\n")
            for h in s1c.HEADS:
                hd = root / "heads" / f"{a}_s{t}_{h}"
                hd.mkdir(parents=True)
                recipe, seed = h.split("_h")
                m = {"comem": s1c.HEAD_LAMBDA[recipe], "seed": int(seed), "m": 20, "class_channels": "none",
                     "git_dirty": False, "git_commit": "abc", "init_sha256": f"i{h}", "perm_sha256": [f"p{h}"],
                     "n_train_events": 3990}
                m.update((head_over or {}).get((a, t, h), {}))
                (hd / "train_meta.json").write_text(json.dumps(m))
    (root / "gates").mkdir()
    for split in ("train", "devA", "test"):
        (root / "gates" / f"K3_{split}.txt").write_text((k3 or {}).get(split, "PASS: 4 caches, identical events"))
    rng = random.Random(7)
    base = [(0.28 + 0.1 * rng.random(), hard_T + 0.02 * (rng.random() - 0.5), 5 + e % 20) for e in range(N_EV)]
    for a in s1c.ARMS:
        for t in s1c.SEEDS:
            for h in s1c.HEADS + ("soft",):
                for dec in s1c.DECODES:
                    g = 0.0
                    if a == "TCM":
                        g = (pair_gain or {}).get((t, h), gain[t - 1] if h in s1c.PRIMARY_HEADS else 0.01)
                    rows = []
                    for e, (a0, hs, nt) in enumerate(base):
                        hsv = hs + (hard_shift if hard_shift and (a, t, h) == ("TCM", 2, "CTH_h1") and e == 5 else 0.0)
                        npred = nt if not (ratio and a == "TCM" and h == "CMH_h1") else max(1, round(nt * ratio))
                        rows.append({"batch": e // 8, "event": e % 8, "ari_particle": round(a0 + g + 0.004 * ((e * 3 + t) % 5 - 2), 9),
                                     "ari_particle_hardseed": hsv, "n_pred": npred, "n_true_all": nt})
                    write_csv(root / "eval" / f"{a}_s{t}" / h / dec / "test_events.csv", rows, EVAL_F)
    return root


def _mk(d):
    d.mkdir(parents=True, exist_ok=True)
    return d / "batch_metrics.jsonl"


# ---- readings ------------------------------------------------------------------------------ #
def test_adopt(tmp_path):
    res = s1c.evaluate(build(tmp_path))
    assert res["gates"]["pass"], res["gates"]["failures"]
    assert res["reading"] == "ADOPT"
    assert res["primary"]["delta"] == pytest.approx(0.03, abs=0.002)
    assert res["primary"]["delta_seed"][1] == pytest.approx(0.03, abs=0.002)
    assert "CMH head + NM" in s1c.render(res)


@pytest.mark.parametrize("kw, reading", [
    ({"gain": (0.0, 0.002)}, "HEAD-SUFFICIENT"),
    ({"gain": (-0.02, -0.01)}, "HARMFUL"),
    ({"gain": (0.03, 0.01)}, "PARTIAL"),                                   # one trunk seed below +0.015
    ({"pair_gain": {(1, "CMH_h2"): -0.002}}, "PARTIAL"),                    # one (seed, head) pair not positive
    ({"ratio": 0.6}, "PARTIAL"),                                           # collapse guard fails
])
def test_readings(tmp_path, kw, reading):
    assert s1c.evaluate(build(tmp_path, **kw))["reading"] == reading


# ---- gates ------------------------------------------------------------------------------------ #
@pytest.mark.parametrize("kw, gate", [
    ({"steps_over": {("TCM", 2): 1}}, "K1 seed 2: per-step"),
    ({"config_over": {("TCM", 1): {"lr": 1e-3}}}, "K1 seed 1: resume configs differ"),
    ({"config_over": {("T", 2): {"git_dirty": True}}}, "K1 T_s2: dirty"),
    ({"config_over": {("TCM", 1): {"git_commit": "zzz"}}}, "K1: trunks trained at different commits"),
    ({"config_over": {("T", 1): {"comem_log": False}}}, "K1 T_s1: comem"),
    ({"finished": False}, "K1 T_s1: not finished"),
    ({"comem_last": {"TCM": 1.9}}, "K2 seed 1"),
    ({"k3": {"test": "FAIL: events differ"}}, "K3 test"),
    ({"head_over": {("TCM", 2, "CMH_h1"): {"init_sha256": "x"}}}, "K4 CMH_h1: init_sha256"),
    ({"head_over": {("T", 2, "CTH_h1"): {"perm_sha256": ["x"]}}}, "K4 CTH_h1: training order"),
    ({"head_over": {("TCM", 1, "CMH_h2"): {"comem": 0.0}}}, "K4 TCM_s1_CMH_h2"),
    ({"hard_shift": 1e-6}, "K5 TCM_s2 CTH_h1/NM"),
    ({"health_delta": -0.02}, "K6"),
    ({"hard_T": 0.25}, "K7"),
])
def test_each_gate_fails_on_its_planted_defect(tmp_path, kw, gate):
    res = s1c.evaluate(build(tmp_path, **kw))
    assert not res["gates"]["pass"] and res["reading"] == "INVALID"
    assert any(f.startswith(gate) for f in res["gates"]["failures"]), res["gates"]["failures"]


def test_different_usable_event_counts_may_have_different_orders(tmp_path):
    over = {("T", 2, "CTH_h1"): {"perm_sha256": ["other"], "n_train_events": 3985}}
    res = s1c.evaluate(build(tmp_path, head_over=over))
    assert res["gates"]["pass"], res["gates"]["failures"]


def test_train_and_head_gates_run_before_the_test_lock(tmp_path):
    root = build(tmp_path)
    fails, info = s1c.train_gates(root)
    assert fails == [] and info["commit"] == "abc" and info["comem_last"]["TCM_s1"] == pytest.approx(1.5)
    assert s1c.head_gates(root, "abc") == []
    assert any(f.startswith("K4 T_s1_CMH_h1: commit") for f in s1c.head_gates(root, "zzz"))


# ---- smoke ------------------------------------------------------------------------------------- #
def smoke_steps(n, lam, jump=0.0, comem=0.5, ids_off=None):
    rows = []
    for i in range(n):
        mem, ex, mg = 1.0, 0.2, 0.1
        rows.append({"global_step": i + 1, "epoch": i // 30, "batch": i % 30,
                     "ids_hash": "x" if ids_off == i else f"h{i}", "loss": 2.0 + (jump if i >= 45 else 0.0) + 0.001 * (i % 3),
                     "soft": mem + ex + mg + lam * comem, "mem": mem, "exist": ex, "merge": mg, "comem": comem})
    return rows


def smoke_root(tmp, **kw):
    s = tmp / "smoke"
    write_csv(s / "R" / "steps.csv", smoke_steps(60, 0.3))
    write_csv(s / "I" / "steps.csv", smoke_steps(60, 0.3, jump=kw.get("jump", 0.0), ids_off=kw.get("ids_off")))
    write_csv(s / "C" / "steps.csv", smoke_steps(20, kw.get("c_lam", 0.0)))
    (s / "I" / "manifest.json").write_text(json.dumps({"resumes": [{"at": "x"}], "finished": "y"}))
    return s


def test_smoke_passes(tmp_path):
    fails, info = s1c.smoke_check(smoke_root(tmp_path))
    assert fails == [], fails


@pytest.mark.parametrize("kw, msg", [
    ({"ids_off": 50}, "resume:"), ({"jump": 1.0}, "loss gap jumps"), ({"c_lam": 0.3}, "T: soft"),
])
def test_smoke_fails(tmp_path, kw, msg):
    fails, _ = s1c.smoke_check(smoke_root(tmp_path, **kw))
    assert any(msg in f for f in fails), fails
