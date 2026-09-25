"""Tests for the predicted-class disagreement diagnostic (ranking_diag --pred-class counters and
scripts/particle_hierarchy/pred_disagree.py). numpy / pure python.

Run:  pytest -q tests/test_pred_disagree.py
"""
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rd = _load("ranking_diag_pd", ROOT / "scripts/particle_hierarchy/ranking_diag.py")
pd = _load("pred_disagree", ROOT / "scripts/particle_hierarchy/pred_disagree.py")


def toy():
    """Four hits of particle 10 (true class 1), head's first choice in column 0.

    h0 correct, trunk agrees              h2 miss onto cand 1 (class 2): different class, trunk disagrees
    h1 correct, hit mispredicted (3)       h3 miss onto cand 2 (class 1): same class, trunk agrees
    """
    W = np.tile([0.7, 0.3], (4, 1))
    nb = np.array([[0, 1], [0, 1], [1, 0], [2, 0]])
    cand_pid, cand_cls, cand_pcls = np.array([10, 20, 30]), np.array([1, 2, 1]), np.array([1, 2, 1])
    own = cand_pid[nb] == 10
    hit_cls, hit_pcls = np.array([1, 1, 1, 1]), np.array([1, 3, 1, 1])
    return (W, nb, own, np.ones(4, bool), np.zeros(4, int), cand_pid, cand_cls, hit_cls), hit_pcls, cand_pcls


def test_counters_on_a_hand_built_event():
    args, hp, cpc = toy()
    _, st, _ = rd.analyse_event(*args, ari_fn=lambda v: 0.0, hit_pcls=hp, cand_pcls=cpc)
    assert (st["n_correct"], st["n_miss"]) == (2, 2)
    assert st["ok_pred_diff"] == 1 and st["miss_pred_diff"] == 1
    assert st["err_diff_pred_diff"] == 1 and st["err_same_pred_diff"] == 0 and st["err_spurious_pred_diff"] == 0
    assert st["ok_hit_mispredicted"] == 1 and st["miss_hit_mispredicted"] == 0
    assert st["ok_dis_hit_mispredicted"] == 1
    assert st["ok_by_cls"][1] == 2 and st["ok_dis_by_cls"][1] == 1
    assert st["miss_by_cls"][1] == 2 and st["miss_dis_by_cls"][1] == 1
    assert sum(st["ok_by_cls"]) == st["n_correct"] and sum(st["miss_by_cls"]) == st["n_miss"]


def test_counters_absent_without_predicted_classes():
    args, _, _ = toy()
    _, st, _ = rd.analyse_event(*args, ari_fn=lambda v: 0.0)
    assert "ok_pred_diff" not in st and "ok_by_cls" not in st


def stats(p_ok, p_miss, n_ok=1000, n_miss=2000):
    return {"n_correct": n_ok, "n_miss": n_miss, "ok_pred_diff": int(p_ok * n_ok), "miss_pred_diff": int(p_miss * n_miss),
            "err_diff_class": 1000, "err_diff_pred_diff": 600, "err_same_class": 800, "err_same_pred_diff": 100,
            "err_spurious": 200, "err_spurious_pred_diff": 50, "ok_hit_mispredicted": 100,
            "miss_hit_mispredicted": 400, "ok_dis_hit_mispredicted": int(p_ok * n_ok * 0.8),
            "ok_by_cls": [n_ok] + [0] * 6, "ok_dis_by_cls": [int(p_ok * n_ok)] + [0] * 6,
            "miss_by_cls": [n_miss] + [0] * 6, "miss_dis_by_cls": [int(p_miss * n_miss)] + [0] * 6}


@pytest.mark.parametrize("p_ok, p_miss, reading", [(0.30, 0.40, "WEAK"), (0.05, 0.40, "STRONG-UNUSED"),
                                                   (0.20, 0.40, "PARTIAL"), (0.15, 0.50, "PARTIAL")])
def test_reading_bands_fixed_in_advance(p_ok, p_miss, reading):
    h = pd.head(stats(p_ok, p_miss))
    assert h["reading"] == reading
    assert h["LR"] == pytest.approx(p_miss / p_ok)
    assert h["veto_precision"] == pytest.approx(p_miss * 2000 / (p_miss * 2000 + p_ok * 1000))


def write(root, name, st):
    d = root / name
    d.mkdir(parents=True)
    (d / "rank_summary.json").write_text(json.dumps({"stats": st}))


def test_contradiction_flag_ignores_oracle_heads(tmp_path):
    write(tmp_path, "CTRL_s1", stats(0.30, 0.40))
    write(tmp_path, "ORACLE_s1", stats(0.05, 0.40))                  # oracle picks use truth: not a contradiction
    res = pd.evaluate(tmp_path)
    assert not res["contradiction_with_R6"]
    write(tmp_path, "PRED_s1", stats(0.05, 0.40))
    res = pd.evaluate(tmp_path)
    assert res["contradiction_with_R6"] and res["contradicting_heads"] == ["PRED_s1"]
    assert "Contradiction with R6:** YES" in pd.render(res)


def test_summaries_without_the_counters_are_refused(tmp_path):
    write(tmp_path, "CTRL_s1", {"n_correct": 1, "n_miss": 1})
    with pytest.raises(SystemExit, match="lacks the disagreement counters"):
        pd.evaluate(tmp_path)
