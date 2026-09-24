"""Tests for the owner-rank diagnostic (scripts/particle_hierarchy/ranking_diag.py).

The per-event analysis and the report are numpy / pure python and run anywhere; the end-to-end
`run` test needs torch + the NuGraph materializer (EAF).
Run:  pytest -q tests/test_ranking_diag.py
"""
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ranking_diag", ROOT / "scripts/particle_hierarchy/ranking_diag.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def toy():
    """Four hits, three candidates each (columns in distance order).

    hit 0: head's first choice is own                      -> correct
    hit 1: own is the head's #2 (and the nearest)          -> miss, head rank 2, demoted
    hit 2: own is the head's #3 (distance rank 2)          -> miss, head rank 3
    hit 3: background, no own                              -> not available
    """
    W = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.6, 0.1],
                  [0.2, 0.1, 0.7],
                  [0.5, 0.3, 0.2]])
    nb = np.array([[0, 1, 2], [0, 1, 2], [2, 0, 1], [1, 2, 0]])
    own = np.array([[True, False, False], [True, False, False],
                    [False, True, False], [False, False, False]])
    pop = np.array([True, True, True, False])
    owner = np.array([0, 0, 0, -1])
    cand_pid = np.array([7, 8, -1])            # candidate 2 is spurious
    cand_cls = np.array([1, 1, 4])
    hit_cls = np.array([1, 2, 1, 0])
    return W, nb, own, pop, owner, cand_pid, cand_cls, hit_cls


def run_toy():
    return rd.analyse_event(*toy(), ari_fn=lambda lab: float(len(set(lab.tolist()))))


def test_first_true_is_one_based_and_zero_when_absent():
    f = np.array([[False, True, True], [False, False, False], [True, False, False]])
    assert rd.first_true(f).tolist() == [2, 0, 1]


def test_labels_follow_the_definitions():
    ev, st, lab = run_toy()
    # head order: hit0 [0,1,2], hit1 [1,0,2], hit2 [2,0,1], hit3 [0,1,2]
    assert lab["pick_only"].tolist() == [0, 1, 1, 1]            # nb[i, head first choice]
    assert lab["head@1"].tolist() == [0, 1, 1, 1]               # only hit 0 resolved (already right)
    assert lab["head@2"].tolist() == [0, 0, 1, 1]               # hit 1's owner is head #2
    assert lab["head@3"].tolist() == [0, 0, 0, 1]
    assert lab["dist@1"].tolist() == [0, 0, 1, 1]               # hits 0, 1 have the nearest own
    assert lab["dist@2"].tolist() == [0, 0, 0, 1]
    assert lab["head@20"].tolist() == lab["dist@20"].tolist()   # R3 by construction
    assert ev["pick_only"] == 2.0 and ev["head@3"] == 2.0       # the stub ARI saw the labels


def test_hit_level_counts():
    _, st, _ = run_toy()
    assert (st["n_pop"], st["n_avail"], st["n_correct"], st["n_miss"]) == (3, 3, 1, 2)
    assert st["owner_missing"] == 0
    assert st["miss_rank_head"] == [1, 1, 0, 0, 0]               # ranks 2 and 3
    assert st["miss_rank_dist"] == [1, 1, 0, 0, 0, 0]            # nearest own (hit 1), rank 2 (hit 2)
    assert st["demoted"] == 1
    assert st["recall_head"][:3] == [1, 2, 3] and st["recall_dist"][:2] == [2, 3]
    # misses: hit 1 picked candidate nb[1,1] = 1 (class 1 vs the hit's class 2) -> different class;
    #         hit 2 picked candidate nb[2,2] = 1 (class 1 = the hit's class 1)     -> same class
    assert st["err_diff_class"] == 1 and st["err_same_class"] == 1 and st["err_spurious"] == 0
    assert sum(st["p1_ok"]) == 1 and sum(st["p1_miss"]) == 2
    assert st["margin_miss_low"] == 0 and st["margin_ok_low"] == 0


def test_spurious_error_type():
    W, nb, own, pop, owner, pid, ccls, hcls = toy()
    W[1] = [0.1, 0.2, 0.7]                                          # hit 1 now picks nb[1,2] = cand 2
    _, st, _ = rd.analyse_event(W, nb, own, pop, owner, pid, ccls, hcls, ari_fn=lambda lab: 0.0)
    assert st["err_spurious"] == 1


def test_ties_resolve_like_argmax():
    W = np.full((1, 3), 1 / 3)
    nb = np.array([[5, 6, 7]])
    own = np.array([[False, False, True]])
    _, st, lab = rd.analyse_event(W, nb, own, np.array([True]), np.array([7]), np.array([0, 0, 0, 0, 0, 1, 2, 3]),
                                  np.zeros(8, int), np.zeros(1, int), ari_fn=lambda lab: 0.0)
    assert lab["pick_only"].tolist() == [5]                      # first maximum, as torch argmax
    assert st["margin_miss_low"] == 1


def test_head_equals_distance_at_full_width_on_random_events():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n, m = 30, 7
        W = rng.random((n, m))
        own = rng.random((n, m)) < 0.2
        nb = rng.integers(0, 12, (n, m))
        pop = rng.random(n) < 0.8
        owner = np.where(own.any(1), 11, -1)
        _, _, lab = rd.analyse_event(W, nb, own, pop, owner, np.arange(12), np.zeros(12, int),
                                     np.zeros(n, int), ari_fn=lambda lab: 0.0, ks=(1, 3, 7))
        assert lab["head@7"].tolist() == lab["dist@7"].tolist()
        assert (lab["head@1"] == np.where(pop & own[np.arange(n), W.argmax(1)], 11, lab["pick_only"])).all()


def test_same_partition():
    assert rd.same_partition([3, 3, 9, 1], [0, 0, 1, 2])
    assert not rd.same_partition([3, 3, 9], [0, 1, 1])
    assert not rd.same_partition([3, 4], [0, 0])


def test_auroc_from_histograms():
    assert rd.auroc([0, 0, 5], [5, 0, 0]) == 1.0
    assert rd.auroc([2, 2], [2, 2]) == pytest.approx(0.5)
    assert rd.auroc([0, 5, 0], [5, 0, 0]) == 1.0 and rd.auroc([5, 0], [0, 5]) == 0.0
    assert math.isnan(rd.auroc([0, 0], [1, 1]))


def numbers(closure2, closure3, rank2, auc, diff=0.3, spur=0.1):
    return {"closure2": closure2, "closure3": closure3, "rank2_share": rank2, "auroc": auc,
            "err": {"diff_class": diff, "spurious": spur, "same_class": 1 - diff - spur}}


def test_readings_follow_the_preregistered_thresholds():
    assert rd.reading(numbers(0.6, 0.8, 0.55, 0.80)) == "AMBIG"
    assert rd.reading(numbers(0.6, 0.8, 0.55, 0.70)) == "MIXED"    # confidence not informative
    assert rd.reading(numbers(0.2, 0.30, 0.20, 0.90)) == "INFO"
    assert rd.reading(numbers(0.2, 0.40, 0.20, 0.90)) == "MIXED"   # owner not rare in the top-3


def test_verdict_needs_five_of_six_and_picks_the_probe():
    nums = {h: dict(numbers(0.2, 0.3, 0.2, 0.9, diff=0.6), reading="INFO") for h in rd.HEADS}
    v = rd.verdict(nums)
    assert v["overall"] == "INFO" and "class-information probe" in v["next"]
    nums["T_s1"]["reading"] = nums["T_s2"]["reading"] = "AMBIG"
    v = rd.verdict(nums)
    assert v["overall"] == "MIXED" and "multiple membership after" in v["next"]
    for x in nums.values():
        x["err"] = {"diff_class": 0.2, "spurious": 0.4, "same_class": 0.4}
    v = rd.verdict(nums)
    assert "hit-to-hit context probe" in v["next"] and v["spurious_flag"]


# ---- report end to end on synthetic summaries ------------------------------------------ #
def summary(tag, H, D, pick, ranks=(60, 20, 10, 5, 5), n_events=10):
    st = {"n_pop": 1000, "n_avail": 400, "n_correct": 200, "n_miss": 100, "owner_missing": 0,
          "demoted": 10, "recall_head": [200, 260, 290, 320, 360, 400],
          "recall_dist": [160, 230, 270, 300, 350, 400], "miss_rank_head": list(ranks),
          "miss_rank_dist": [10, 30, 20, 20, 10, 10],
          "p1_ok": [0] * 150 + [200] + [0] * 49, "p1_miss": [0] * 60 + [100] + [0] * 139,
          "margin_ok_low": 20, "margin_miss_low": 70, "err_spurious": 10, "err_same_class": 30,
          "err_diff_class": 60}
    ari = {"pick_only": pick, **{f"head@{k}": H[i] for i, k in enumerate(rd.KS)},
           **{f"dist@{k}": D[i] for i, k in enumerate(rd.KS)}}
    return {"ari": ari, "stats": st, "n_events": n_events,
            "gates": {"R1_pick_only_fail": [], "R3_head_eq_dist_at_M_fail": [], "R4_owner_missing": 0}}


def build_report_tree(tmp, flips=None, r2_shift=0.0, r2_n=0):
    rank_root, mb_root = tmp / "rank", tmp / "mb"
    H = [0.26, 0.42, 0.47, 0.50, 0.53, 0.54]                         # AMBIG-shaped
    D = [0.25, 0.33, 0.38, 0.43, 0.50, 0.54]
    for h in rd.HEADS:
        s = summary(h, H, D, 0.25)
        if flips and h in flips:
            s["gates"]["R1_pick_only_fail"] = flips[h]
        d = rank_root / h
        d.mkdir(parents=True)
        (d / "rank_summary.json").write_text(json.dumps(s))
        with open(d / "rank_events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["batch", "event", "pick_only"])
            w.writeheader()
            w.writerows({"batch": 0, "event": e, "pick_only": 0.25} for e in range(10))
    for h, sub in rd.EP29_DIAG.items():
        d = mb_root / sub
        d.mkdir(parents=True)
        with open(d / "diagnose_events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["batch", "event", "variant", "ari_particle"])
            w.writeheader()
            for e in range(10):
                w.writerow({"batch": 0, "event": e, "variant": "pick_only",
                            "ari_particle": 0.25 + (r2_shift if e < r2_n else 0.0)})
                w.writerow({"batch": 0, "event": e, "variant": "v2_nonull", "ari_particle": 0.9})
    return rank_root, mb_root


def test_report_passes_and_reads_ambig(tmp_path):
    rank_root, mb_root = build_report_tree(tmp_path)
    res = rd.evaluate(rank_root, mb_root)
    assert res["gates"]["pass"], res["gates"]["failures"]
    n = res["heads"]["T_s1"]
    assert n["closure2"] == pytest.approx((0.42 - 0.26) / (0.54 - 0.26))
    assert n["rank2_share"] == pytest.approx(0.6) and n["auroc"] == 1.0
    assert n["reading"] == "AMBIG" and res["verdict"]["overall"] == "AMBIG"
    md = rd.render(res)
    assert "**Gates:** PASS" in md and "| reading | AMBIG" in md and "tolerated near-tie flips" in md


def test_r1_tolerates_two_flips_but_not_three(tmp_path):
    rank_root, mb_root = build_report_tree(tmp_path / "a", flips={"C_s2": [[0, 1], [0, 2]]})
    assert rd.evaluate(rank_root, mb_root)["gates"]["pass"]
    rank_root, mb_root = build_report_tree(tmp_path / "b", flips={"C_s2": [[0, 1], [0, 2], [0, 3]]})
    res = rd.evaluate(rank_root, mb_root)
    assert not res["gates"]["pass"] and res["verdict"]["overall"] == "INVALID"


def test_r2_catches_a_changed_pick_only(tmp_path):
    rank_root, mb_root = build_report_tree(tmp_path / "a", r2_shift=4e-4, r2_n=2)
    assert rd.evaluate(rank_root, mb_root)["gates"]["pass"]          # 2 events, mean within 1e-4
    rank_root, mb_root = build_report_tree(tmp_path / "b", r2_shift=4e-4, r2_n=3)
    fails = rd.evaluate(rank_root, mb_root)["gates"]["failures"]
    assert any(f.startswith("R2 ep29_s1") for f in fails)


def test_missing_head_is_refused(tmp_path):
    rank_root, mb_root = build_report_tree(tmp_path)
    (rank_root / "C_s1" / "rank_summary.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        rd.evaluate(rank_root, mb_root)


# ---- run end to end (torch, materializer: EAF) -------------------------------------------- #
def test_run_end_to_end_on_a_synthetic_cache(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    import sys
    sp_spec = importlib.util.spec_from_file_location("soft_particle_t", ROOT / "scripts/particle_hierarchy/soft_particle.py")
    sp = importlib.util.module_from_spec(sp_spec)
    sp_spec.loader.exec_module(sp)
    torch.manual_seed(0)
    cache = tmp_path / "cache"
    cache.mkdir()
    evs = []
    for i in range(3):
        g = torch.Generator().manual_seed(i)
        k, n = 4, 30
        y = torch.arange(k).repeat_interleave(n)
        ox = torch.zeros(k, 8)
        ox[:, 0] = torch.arange(k, dtype=torch.float) * 3.0
        ox = ox[y] + 0.4 * torch.randn(k * n, 8, generator=g)
        beta = 0.1 * torch.rand(k * n, generator=g)
        beta[torch.arange(k) * n] = 0.9
        beta[torch.arange(k) * n + 1] = 0.8                       # duplicates
        evs.append({"batch": 0, "event": i, "ox": ox, "beta": beta,
                    "mask": torch.ones(k * n, dtype=torch.bool), "y_i": y.to(torch.int32),
                    "y_sem": (y % 2).to(torch.int8), "x_nugraph": torch.zeros(k * n, dtype=torch.int32),
                    "hit_x_fg": torch.randn(k * n, 3, generator=g).half()})
    torch.save(evs, cache / "shard_00000.pt")
    model = sp.build_model(3, 8, 16, 2)
    cfg = {"rule": "fixed1:0.5", "m": 20, "r": 2, "t_d": 0.3, "t_assign": float("inf"),
           "hit_dim": 3, "lat_dim": 8, "width": 16, "epoch": 0}
    head = tmp_path / "head.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, head)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["ranking_diag.py", "run", "--head", str(head), "--cache", str(cache),
                                      "--batches", "0:1", "--out", str(out), "--device", "cpu"])
    rd.main()
    res = json.loads((out / "rank_summary.json").read_text())
    assert res["n_events"] == 3 and res["gates"]["R1_pick_only_fail"] == []
    assert res["gates"]["R4_owner_missing"] == 0
    assert res["ari"]["head@20"] == pytest.approx(res["ari"]["dist@20"])
    st = res["stats"]
    assert st["n_correct"] + st["n_miss"] == st["n_avail"] > 0
    assert sum(st["miss_rank_head"]) == st["n_miss"]
    # the pick_only ARI equals the evaluated diagnose path's pick_only on the same events
    rows = list(csv.DictReader(open(out / "rank_events.csv")))
    for r in rows:
        ev = next(e for e in evs if e["event"] == int(r["event"]))
        d = sp.prepare(ev, sp._import_materialize(), sp.s0.make_rule("fixed1:0.5"), 20, 0.3,
                       float("inf"), torch.device("cpu"))
        ref = [x for x in sp.diagnose_event(model.eval(), d) if x["variant"] == "pick_only"][0]
        assert float(r["pick_only"]) == pytest.approx(ref["ari_particle"], abs=1e-12)


def test_run_refuses_beyond_dev_a(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import sys
    monkeypatch.setattr(sys, "argv", ["ranking_diag.py", "run", "--head", "x", "--cache", "y",
                                      "--batches", "150:314", "--out", str(tmp_path), "--device", "cpu"])
    with pytest.raises(SystemExit, match="dev-A only"):
        rd.main()
