"""Tests for the Stage 1a-R5 report driver (scripts/particle_hierarchy/class_probe.py). Pure python.

Run:  pytest -q tests/test_class_probe_report.py
"""
import csv
import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("class_probe", ROOT / "scripts/particle_hierarchy/class_probe.py")
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)
KS = (1, 2, 3, 5, 10, 20)
N_EV = 200


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def meta(arm, t, **over):
    m = {"m": 20, "seed": t, "rule": "quant:0.995:0.05:0.5", "width": 64, "hit_dim": 136, "epochs": 8,
         "class_channels": cp.MODE[arm], "oracle_upper_bound": arm == "ORACLE", "init_sha256": f"init{t}",
         "perm_sha256": [f"p{t}{e}" for e in range(8)], "best_epoch": 5 if arm == "CTRL" else 6,
         "best_dev_ari": 0.26, "git_commit": "abc", "git_dirty": False, "materialize_sha256": "m",
         "argv": [arm]}
    m.update(over)
    return m


def rank_summary(diff_class):
    miss = 1000
    st = {"n_pop": 10000, "n_avail": 4000, "n_correct": 2000, "n_miss": miss, "owner_missing": 0, "demoted": 100,
          "recall_head": [2000, 2600, 2900, 3200, 3700, 4000], "recall_dist": [1600, 2300, 2700, 3000, 3500, 4000],
          "miss_rank_head": [300, 150, 200, 230, 120], "miss_rank_dist": [140, 300, 150, 200, 110, 100],
          "p1_ok": [0] * 150 + [2000] + [0] * 49, "p1_miss": [0] * 60 + [miss] + [0] * 139,
          "margin_ok_low": 900, "margin_miss_low": 700, "err_spurious": 80,
          "err_diff_class": int(diff_class * miss), "err_same_class": miss - 80 - int(diff_class * miss)}
    ari = {"pick_only": 0.25, **{f"head@{k}": 0.40 + 0.01 * i for i, k in enumerate(KS)},
           **{f"dist@{k}": 0.36 + 0.01 * i for i, k in enumerate(KS)}}
    return {"ari": ari, "stats": st, "n_events": N_EV,
            "gates": {"R1_pick_only_fail": [], "R3_head_eq_dist_at_M_fail": [], "R4_owner_missing": 0}}


def build(tmp, gain=0.05, diff_oracle=0.2, meta_over=None, hard_shift=0.0):
    root, mb = tmp / "root", tmp / "mb"
    rng = random.Random(3)
    base = [(0.25 + 0.1 * rng.random(), 0.15 + 0.05 * rng.random()) for _ in range(N_EV)]
    for t in cp.SEEDS:
        for arm in cp.ARMS:
            d = root / "heads" / cp.name(arm, t)
            d.mkdir(parents=True)
            (d / "train_meta.json").write_text(json.dumps(meta(arm, t, **((meta_over or {}).get((arm, t), {})))))
            g = gain if arm == "ORACLE" else 0.0
            for sub in ("diag_last", "diag_best"):
                rows = []
                for e, (a0, h) in enumerate(base):
                    hh = h + (hard_shift if (arm == "ORACLE" and e == 5) else 0.0)
                    for var in ("v2_nonull", "pick_only"):
                        rows.append({"batch": e // 8, "event": e % 8, "variant": var,
                                     "ari_particle": a0 + g + rng.gauss(0, 0.01), "ari_hardseed": hh})
                write_csv(root / sub / cp.name(arm, t) / "diagnose_events.csv",
                          ["batch", "event", "variant", "ari_particle", "ari_hardseed"], rows)
            r = root / "rank" / cp.name(arm, t)
            r.mkdir(parents=True)
            (r / "rank_summary.json").write_text(json.dumps(rank_summary(diff_oracle if arm == "ORACLE" else 0.5)))
        write_csv(mb / "retrieval" / f"diag_m20_s{t}" / "diagnose_events.csv",
                  ["batch", "event", "variant", "ari_particle", "ari_hardseed"],
                  [{"batch": e // 8, "event": e % 8, "variant": "v2_nonull", "ari_particle": a0, "ari_hardseed": h}
                   for e, (a0, h) in enumerate(base)])
    return root, mb


def test_planted_class_effect_reads_class(tmp_path):
    res = cp.evaluate(*build(tmp_path, gain=0.05, diff_oracle=0.2))
    assert res["gates"]["pass"], res["gates"]["failures"]
    for t in cp.SEEDS:
        assert res["seeds"][t]["last:v2_nonull"]["mean"] == pytest.approx(0.05, abs=0.004)
    assert res["verdict"]["reading"] == "CLASS" and res["verdict"]["flag"] is None
    md = cp.render(res)
    assert "ORACLE UPPER BOUND" in md and "**Reading:** CLASS" in md


def test_gain_without_fewer_class_errors_is_flagged(tmp_path):
    v = cp.evaluate(*build(tmp_path, gain=0.05, diff_oracle=0.45))["verdict"]
    assert v["reading"] == "PARTIAL" and "explain before acting" in v["flag"]


def test_no_gain_reads_not_class(tmp_path):
    v = cp.evaluate(*build(tmp_path, gain=0.0, diff_oracle=0.5))["verdict"]
    assert v["reading"] == "NOT-CLASS" and "hit-to-hit context" in v["next"]


def test_small_gain_reads_partial(tmp_path):
    assert cp.evaluate(*build(tmp_path, gain=0.02, diff_oracle=0.2))["verdict"]["reading"] == "PARTIAL"


@pytest.mark.parametrize("over, gate", [
    ({("ORACLE", 1): {"init_sha256": "other"}}, "C1 seed 1"),
    ({("ORACLE", 2): {"perm_sha256": ["x"] * 8}}, "C2 seed 2"),
    ({("ORACLE", 1): {"lr": 0.01}}, "C3 seed 1: recipe differs in ['lr']"),
    ({("ORACLE", 2): {"git_commit": "def"}}, "C3 seed 2: recipe differs in ['git_commit']"),
    ({("CTRL", 1): {"git_dirty": True}}, "C3 CTRL_s1: trained from a dirty"),
    ({("CTRL", 2): {"class_channels": "none"}}, "C3 CTRL_s2: class_channels"),
])
def test_each_gate_fails_on_its_planted_defect(tmp_path, over, gate):
    res = cp.evaluate(*build(tmp_path, meta_over=over))
    assert not res["gates"]["pass"] and res["verdict"]["reading"] == "INVALID"
    assert any(f.startswith(gate) for f in res["gates"]["failures"]), res["gates"]["failures"]


def test_c4_catches_a_changed_hard_seed(tmp_path):
    res = cp.evaluate(*build(tmp_path, hard_shift=1e-6))
    assert any(f.startswith("C4 seed 1") for f in res["gates"]["failures"])


def test_missing_head_is_refused(tmp_path):
    root, mb = build(tmp_path)
    (root / "heads" / "ORACLE_s2" / "train_meta.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        cp.evaluate(root, mb)
