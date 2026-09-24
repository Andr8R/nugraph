"""Tests for the T/C x M=20 analysis driver (scripts/particle_hierarchy/retrieval_tc.py).

The report is pure python, so these run anywhere; the identity fingerprint test needs torch.
Run:  pytest -q tests/test_retrieval_tc.py
"""
import csv
import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("retrieval_tc", ROOT / "scripts/particle_hierarchy/retrieval_tc.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

COLS = ["batch", "event", "ari_particle", "completeness", "purity", "ari_particle_hardseed",
        "pick_own", "own_in_nb", "pop"]
N_EV = 400


def write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def meta(arm, t, m, over=None):
    base = {"m": m, "seed": 1, "rule": "quant:0.995:0.05:0.5", "r": 2, "width": 64, "epochs": 8,
            "target": "own", "exist_target": "real", "merge_scope": "local", "dist_only": False,
            "train_cache": f"/x/retrieval_tc/cache_train_{arm}_s{t}"}
    return {**base, **(over or {})}


def build(tmp, t_gain=0.05, c_gain=0.02, own_t=0.03, own_c=0.01, hard_shift=0.0, drop_event=None,
          meta_over=None):
    """Synthetic experiment: widening adds t_gain ARI on T and c_gain on C (plus noise)."""
    root, s1b = tmp / "root", tmp / "s1b"
    rng = random.Random(7)
    base = [(0.20 + 0.1 * rng.random(), 0.25 * rng.random(), 100 + rng.randrange(900)) for _ in range(N_EV)]
    for arm in rt.ARMS:
        for t in rt.TRUNK_SEEDS:
            hard = [0.15 + 0.05 * rng.random() for _ in range(N_EV)]
            ref_rows = []
            for m in rt.MS:
                rows = []
                for i, (ari0, own0, pop) in enumerate(base):
                    if drop_event is not None and (arm, t, m) == drop_event[:3] and i == drop_event[3]:
                        continue
                    wide = m == 20
                    gain = (t_gain if arm == "T" else c_gain) * wide
                    og = (own_t if arm == "T" else own_c) * wide
                    avail = 0.3 + 0.1 * wide
                    rows.append({"batch": i // 8, "event": i % 8,
                                 "ari_particle": ari0 + gain + rng.gauss(0, 0.01),
                                 "completeness": 0.5 + gain + rng.gauss(0, 0.01),
                                 "purity": 0.6 + rng.gauss(0, 0.01),
                                 "ari_particle_hardseed": hard[i],
                                 "pick_own": round(pop * (own0 + og)), "own_in_nb": round(pop * avail),
                                 "pop": pop})
                write_events(root / "eval" / rt.run_name(arm, t, m) / "devB_events.csv", rows)
                hd = root / "heads" / rt.run_name(arm, t, m)
                hd.mkdir(parents=True, exist_ok=True)
                (hd / "train_meta.json").write_text(json.dumps(meta(arm, t, m, (meta_over or {}).get((arm, t, m)))))
            ref_rows = [{"batch": i // 8, "event": i % 8, "ari_particle": 0, "completeness": 0, "purity": 0,
                         "ari_particle_hardseed": hard[i] + (hard_shift if i == 3 else 0.0),
                         "pick_own": 0, "own_in_nb": 0, "pop": 1} for i in range(N_EV)]
            write_events(s1b / f"{arm}_s{t}" / "soft" / "devB_events.csv", ref_rows)
    return root, s1b


def test_planted_interaction_is_recovered_and_passes(tmp_path):
    root, s1b = build(tmp_path, t_gain=0.05, c_gain=0.02)
    res = rt.evaluate(root, s1b)
    assert res["gates"]["pass"], res["gates"]["failures"]
    m, se = res["interaction"]["ari"]["combined"]
    assert m == pytest.approx(0.03, abs=0.004) and res["interaction"]["ari"]["pass"]
    assert res["interaction"]["comp"]["pass"]
    assert res["widening"]["ari"]["T"]["combined"][0] == pytest.approx(0.05, abs=0.004)
    assert res["verdict"].startswith("PRIMARY SUPPORTED")


def test_additive_effects_do_not_pass(tmp_path):
    root, s1b = build(tmp_path, t_gain=0.03, c_gain=0.03, own_t=0.01, own_c=0.01)
    res = rt.evaluate(root, s1b)
    assert not res["interaction"]["ari"]["pass"]
    assert res["verdict"].startswith("PRIMARY NOT SUPPORTED")


def test_owner_identification_mechanism_needs_kappa_and_own_pick(tmp_path):
    root, s1b = build(tmp_path, t_gain=0.05, c_gain=0.02, own_t=0.05, own_c=0.005)
    res = rt.evaluate(root, s1b)
    assert res["interaction"]["own"]["pass"]
    for t in rt.TRUNK_SEEDS:
        k = res["kappa"][f"s{t}"]
        assert k["kappa_T"] > k["kappa_C"] and k["ci95"][0] > 0
    assert "owner identification" in res["verdict"]


def test_kappa_is_the_ratio_of_sum_changes():
    r5 = {1: {"pick_own": 10, "own_in_nb": 40}, 2: {"pick_own": 20, "own_in_nb": 60}}
    r20 = {1: {"pick_own": 13, "own_in_nb": 50}, 2: {"pick_own": 21, "own_in_nb": 70}}
    assert rt.kappa(r5, r20, [1, 2]) == pytest.approx(4 / 20)
    assert math.isnan(rt.kappa(r5, r5, [1, 2]))                      # nothing newly available


def test_hard_seed_mismatch_fails_the_gate(tmp_path):
    root, s1b = build(tmp_path, hard_shift=1e-6)
    res = rt.evaluate(root, s1b)
    assert not res["gates"]["pass"]
    assert any(f.startswith("G3") and "hard seed differs on 1 events" in f for f in res["gates"]["failures"])
    assert res["verdict"].startswith("INVALID")


def test_nan_against_a_number_is_a_mismatch():
    assert rt.same_value(float("nan"), float("nan"))
    assert not rt.same_value(float("nan"), 0.2) and not rt.same_value(0.2, float("nan"))
    assert rt.same_value(0.2, 0.2 + 1e-12) and not rt.same_value(0.2, 0.2 + 1e-6)


def test_a_missing_event_that_carries_values_fails(tmp_path):
    """Dropping a real event from one run leaves it carrying values in the others: G2 must fail."""
    root, s1b = build(tmp_path, drop_event=("C", 2, 20, 17))
    res = rt.evaluate(root, s1b)
    assert any(f.startswith("G2 ") and "carry values" in f for f in res["gates"]["failures"])


def _blank_event(path, key):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        if (int(r["batch"]), int(r["event"])) == key:
            r.update(ari_particle="nan", completeness="nan", purity="nan", pick_own=0, own_in_nb=0, pop=0)
    write_events(path, rows)


def test_a_trunk_skipped_empty_event_is_excluded_not_failed(tmp_path):
    """The real case: an event with no population hits is skipped by one trunk (empty predicted
    foreground). It carries nothing anywhere, so it is excluded and every number is unchanged."""
    root, s1b = build(tmp_path)
    ref = rt.evaluate(root, s1b)
    key = (2, 1)                                                     # event index 17
    for a in rt.ARMS:
        for t in rt.TRUNK_SEEDS:
            _blank_event(s1b / f"{a}_s{t}" / "soft" / "devB_events.csv", key)
            for m in rt.MS:
                _blank_event(root / "eval" / rt.run_name(a, t, m) / "devB_events.csv", key)
    for m in rt.MS:                                                  # T_s2 skips it entirely
        p = root / "eval" / rt.run_name("T", 2, m) / "devB_events.csv"
        write_events(p, [r for r in csv.DictReader(open(p)) if (int(r["batch"]), int(r["event"])) != key])
    p = s1b / "T_s2" / "soft" / "devB_events.csv"
    write_events(p, [r for r in csv.DictReader(open(p)) if (int(r["batch"]), int(r["event"])) != key])
    res = rt.evaluate(root, s1b)
    assert res["gates"]["pass"], res["gates"]["failures"]
    assert res["gates"]["excluded_noncontributing"]["T_s1_m5"] == 1
    assert res["gates"]["excluded_noncontributing"]["T_s2_m5"] == 0
    blanked = rt.evaluate(root, s1b)["interaction"]["ari"]["combined"]
    assert blanked[0] == pytest.approx(ref["interaction"]["ari"]["combined"][0], abs=5e-4)


def test_a_head_off_the_registered_recipe_fails(tmp_path):
    root, s1b = build(tmp_path, meta_over={("T", 1, 20): {"m": 10},
                                           ("C", 2, 5): {"train_cache": "/x/cache_train_hx"}})
    fails = rt.evaluate(root, s1b)["gates"]["failures"]
    assert any(f.startswith("G1 T_s1_m20") and "'m': 10" in f for f in fails)
    assert any(f.startswith("G1 C_s2_m5") and "train cache" in f for f in fails)


def test_missing_run_is_refused(tmp_path):
    root, s1b = build(tmp_path)
    (root / "eval" / "T_s2_m5" / "devB_events.csv").unlink()
    with pytest.raises(SystemExit, match="missing"):
        rt.evaluate(root, s1b)


def test_report_renders_and_writes(tmp_path):
    root, s1b = build(tmp_path)
    a = type("A", (), {"root": str(root), "stage1b_devb": str(s1b), "out": str(tmp_path / "rep")})()
    rt.cmd_report(a)
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "Gates:** PASS" in md and "| particle ARI (deployed decode) |" in md
    assert json.loads((tmp_path / "rep" / "report.json").read_text())["gates"]["pass"]


def test_compare_fingerprints():
    a = [(0, 0, 5, "x", "y"), (0, 1, 7, "p", "q")]
    assert rt.compare_fingerprints({"a": a, "b": list(a)})[0]
    ok, msg = rt.compare_fingerprints({"a": a, "b": [a[0], (0, 1, 7, "p", "Q")]})
    assert not ok and "first at index 1" in msg
    assert not rt.compare_fingerprints({"a": a, "b": a[:1]})[0]
    assert not rt.compare_fingerprints({"a": [], "b": []})[0]


def test_fingerprints_read_truth_only(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    evs = [{"batch": 0, "event": g, "y_i": torch.tensor([0, 1, -1], dtype=torch.int32) + g,
            "y_sem": torch.tensor([1, 2, 6], dtype=torch.int8), "ox": torch.randn(3, 8)} for g in range(3)]
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        other = [dict(e, ox=torch.randn(3, 8)) for e in evs]      # trunk outputs differ, truth doesn't
        torch.save(other, tmp_path / name / "shard_0000.pt")
    fa, fb = rt.fingerprints(tmp_path / "a"), rt.fingerprints(tmp_path / "b")
    assert fa == fb and len(fa) == 3 and fa[0][2] == 3
    (tmp_path / "c").mkdir()                                     # same truth, older dtypes
    torch.save([dict(e, y_i=e["y_i"].long(), y_sem=e["y_sem"].int()) for e in evs],
               tmp_path / "c" / "shard_0000.pt")
    assert rt.fingerprints(tmp_path / "c") == fa
