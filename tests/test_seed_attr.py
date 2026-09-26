"""Tests for Stage 1a-R7 Step 1: the seed-side / hit-side / both-correct split (ranking_diag --pred-class counters)
and scripts/particle_hierarchy/seed_attr.py. numpy / pure python.

Run:  pytest -q tests/test_seed_attr.py
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


rd = _load("ranking_diag_sa", ROOT / "scripts/particle_hierarchy/ranking_diag.py")
sa = _load("seed_attr", ROOT / "scripts/particle_hierarchy/seed_attr.py")


def event():
    """Six hits, head's first choice in column 0. Candidates (particle, true class, predicted class):
    c0 (10, pion, pion)  c1 (20, hadron, PION)  c2 (30, hadron, hadron)  c3 (40, pion, HADRON)  c4 (10, pion, pion)

    h0 pion of 10 picks c1          -> different class, seed predicted as the hit's class: candidate disguised
    h1 hadron of 30, predicted pion, picks c0 -> different class, hit predicted as the candidate's class: hit disguised
    h2 pion of 10 picks c2          -> different class, both predicted correctly
    h3 pion of 40 picks c0          -> same-class miss; its owner c3 has a mispredicted seed
    h4 pion of 10 picks c4          -> correct
    h5 pion of 40 picks c3          -> correct, but the picked (owner) seed is mispredicted
    """
    cand_pid, cand_cls, cand_pcls = np.array([10, 20, 30, 40, 10]), np.array([1, 3, 3, 1, 1]), np.array([1, 1, 3, 3, 1])
    nb = np.array([[1, 0], [0, 2], [2, 0], [0, 3], [4, 1], [3, 0]])
    y = np.array([10, 30, 10, 40, 10, 40])
    own = cand_pid[nb] == y[:, None]
    hit_cls, hit_pcls = np.array([1, 3, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 1])
    W = np.tile([0.7, 0.3], (6, 1))
    _, st, _ = rd.analyse_event(W, nb, own, np.ones(6, bool), np.zeros(6, int), cand_pid, cand_cls, hit_cls,
                                ari_fn=lambda v: 0.0, hit_pcls=hit_pcls, cand_pcls=cand_pcls)
    return st


def test_counters_on_a_hand_built_event():
    st = event()
    assert (st["n_miss"], st["n_correct"]) == (4, 2)
    assert (st["dm_cand_disguised"], st["dm_hit_disguised"], st["dm_both_correct"]) == (1, 1, 1)
    assert st["dm_both_wrong"] == st["dm_seed_wrong_third"] == st["dm_hit_wrong_third"] == 0
    assert st["dm_by_cls"][1] == 2 and st["dm_by_cls"][3] == 1
    assert st["dm_cand_disguised_by_cls"][1] == 1 and st["dm_hit_disguised_by_cls"][3] == 1
    assert st["miss_owner_seed_wrong"] == 1 and st["ok_owner_seed_wrong"] == 1


def test_cells_partition_the_different_class_misses():
    rng = np.random.default_rng(4)
    for _ in range(30):
        n, m, K = 60, 5, 12
        cand_pid = rng.integers(-1, 6, K)
        cand_cls, cand_pcls = rng.integers(0, 6, K), rng.integers(0, 6, K)
        nb = rng.integers(0, K, (n, m))
        y = rng.integers(0, 6, n)
        own = cand_pid[nb] == y[:, None]
        hit_cls, hit_pcls = rng.integers(0, 6, n), rng.integers(0, 6, n)
        _, st, _ = rd.analyse_event(rng.random((n, m)), nb, own, np.ones(n, bool), np.zeros(n, int), cand_pid,
                                    cand_cls, hit_cls, ari_fn=lambda v: 0.0, hit_pcls=hit_pcls, cand_pcls=cand_pcls)
        total = sum(st[f"dm_{k}"] for k in sa.MAIN + sa.OTHER)
        assert total == sum(st["dm_by_cls"])


def stats(cand, hit, both, lr=2.0, n=1000):
    rest = n - int(cand * n) - int(hit * n) - int(both * n)
    z7 = [0] * 7
    return {"dm_cand_disguised": int(cand * n), "dm_hit_disguised": int(hit * n), "dm_both_correct": int(both * n),
            "dm_both_wrong": rest, "dm_seed_wrong_third": 0, "dm_hit_wrong_third": 0,
            "n_miss": 1000, "n_correct": 1000, "miss_owner_seed_wrong": int(100 * lr), "ok_owner_seed_wrong": 100,
            "dm_by_cls": z7, "dm_cand_disguised_by_cls": z7, "dm_hit_disguised_by_cls": z7,
            "dm_both_correct_by_cls": z7, "miss_by_cls": z7, "ok_by_cls": z7,
            "miss_owner_seed_wrong_by_cls": z7, "ok_owner_seed_wrong_by_cls": z7}


@pytest.mark.parametrize("args, reading", [
    ((0.50, 0.20, 0.20, 2.0), "SEED"),
    ((0.50, 0.20, 0.20, 1.2), "MIXED"),            # seed-side errors, but owner seeds are not worse on misses
    ((0.20, 0.50, 0.20, 2.0), "HIT"),
    ((0.20, 0.20, 0.50, 2.0), "HEAD/GEOMETRY"),
    ((0.30, 0.30, 0.30, 2.0), "MIXED"),            # nothing reaches 0.40
])
def test_readings_fixed_in_advance(args, reading):
    assert sa.head(stats(*args))["reading"] == reading


def write(root, name, st):
    d = root / name
    d.mkdir(parents=True)
    (d / "rank_summary.json").write_text(json.dumps({"stats": st}))


def test_ctrl_seeds_must_agree_and_oracle_is_reference_only(tmp_path):
    write(tmp_path, "CTRL_s1", stats(0.5, 0.2, 0.2))
    write(tmp_path, "CTRL_s2", stats(0.5, 0.2, 0.2))
    write(tmp_path, "ORACLE_s1", stats(0.2, 0.2, 0.5))
    res = sa.evaluate(tmp_path)
    assert res["reading"] == "SEED" and res["decided_on"] == ["CTRL_s1", "CTRL_s2"] and "ORACLE-SEED" in res["next"]
    (tmp_path / "CTRL_s2" / "rank_summary.json").write_text(json.dumps({"stats": stats(0.2, 0.5, 0.2)}))
    assert sa.evaluate(tmp_path)["reading"] == "MIXED"
    assert "Reading (decided on CTRL_s1, CTRL_s2):** MIXED" in sa.render(sa.evaluate(tmp_path))


def test_old_summaries_and_missing_ctrl_are_refused(tmp_path):
    write(tmp_path, "CTRL_s1", {"n_miss": 1})
    with pytest.raises(SystemExit, match="Step-1 counters"):
        sa.evaluate(tmp_path)
    (tmp_path / "CTRL_s1" / "rank_summary.json").write_text(json.dumps({"stats": stats(0.5, 0.2, 0.2)}))
    with pytest.raises(SystemExit, match="both CTRL"):
        sa.evaluate(tmp_path)
