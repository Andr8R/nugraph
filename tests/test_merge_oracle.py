"""Tests for the merging-vs-un-sharing oracle (scripts/particle_hierarchy/merge_oracle.py).

Per-event oracles and the report are numpy / pure python; the end-to-end `run` needs torch + the
NuGraph materializer (EAF).  Run:  pytest -q tests/test_merge_oracle.py
"""
import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("merge_oracle", ROOT / "scripts/particle_hierarchy/merge_oracle.py")
mo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mo)


def ari(pred, true):
    """Adjusted Rand index, numpy only (independent of sklearn)."""
    pred, true = np.asarray(pred), np.asarray(true)
    _, p = np.unique(pred, return_inverse=True)
    _, t = np.unique(true, return_inverse=True)
    c = np.zeros((p.max() + 1, t.max() + 1))
    np.add.at(c, (p, t), 1)
    comb = lambda x: (x * (x - 1) / 2).sum()
    sij, sa, sb, n = comb(c), comb(c.sum(1)), comb(c.sum(0)), comb(np.array([len(pred)]))
    exp = sa * sb / n
    return (sij - exp) / (0.5 * (sa + sb) - exp) if (0.5 * (sa + sb) - exp) else 1.0


# particles: A = 10 (candidates 0 primary, 1 duplicate), B = 20 (2 primary, 3 duplicate); 4 spurious
CAND_PID = np.array([10, 10, 20, 20, -1])
CAND_PRIMARY = np.array([0, 0, 2, 2, -1])
TRUE = np.array([10, 10, 20, 20, 10, 20])


def toy():
    """Six hits, two candidates each, head's first choice in column 0.

    h0 A picks 0 (primary)        correct           h3 B picks 1 (A's duplicate)  wrong, on a particle
    h1 A picks 1 (duplicate)      correct           h4 A picks 4 (spurious)       wrong, on spurious
    h2 B picks 2 (primary)        correct           h5 B picks 3 (duplicate)      correct
    h3 and h4 have their owner at head rank 2.
    """
    W = np.tile([0.8, 0.2], (6, 1))
    nb = np.array([[0, 2], [1, 3], [2, 0], [1, 2], [4, 0], [3, 1]])
    own = CAND_PID[nb] == TRUE[:, None]
    pop = np.ones(6, bool)
    owner = np.array([0, 0, 2, 2, 0, 2])
    local_root = np.array([0, 0, 2, 3, 4])        # only (0,1) is a reachable same-particle pair
    return W, nb, own, pop, owner, CAND_PID, CAND_PRIMARY, local_root


def test_oracle_labels_follow_the_definitions():
    ev, st, checks, lab = mo.merge_oracles(*toy(), ari_fn=lambda v: ari(v, TRUE))
    assert lab["P"].tolist() == [0, 1, 2, 1, 4, 3]
    assert lab["M"].tolist() == [0, 0, 2, 0, 4, 2]      # every hit follows its candidate's particle
    assert lab["H1"].tolist() == [0, 0, 2, 1, 4, 2]     # correct hits consolidate; h3 stays on 1
    assert lab["ML"].tolist() == [0, 0, 2, 0, 4, 3]     # (2,3) is not reachable locally
    assert lab["M+H@2"].tolist() == [0, 0, 2, 2, 0, 2]  # plus a perfect top-2 resolver = truth
    assert all(checks.values())
    assert ev["M+H@2"] == pytest.approx(1.0)
    # the decomposition this experiment is about
    assert ev["H1"] - ev["M"] == pytest.approx(ari(lab["H1"], TRUE) - ari(lab["M"], TRUE))
    assert ev["M"] >= ev["ML"] >= ev["P"] - 1e-12


def test_hit_level_counts():
    _, st, _, _ = mo.merge_oracles(*toy(), ari_fn=lambda v: 0.0)
    assert st == {"n_pop": 6, "n_correct": 4, "n_wrong_on_particle": 1, "n_wrong_on_spurious": 1,
                  "n_correct_on_duplicate": 2}


def test_learned_labels_are_scored_when_given():
    ev, _, _, lab = mo.merge_oracles(*toy(), ari_fn=lambda v: ari(v, TRUE), learned=np.array([0, 0, 1, 0, 2, 3]))
    assert "L" in ev and ev["L"] == pytest.approx(ari([0, 0, 1, 0, 2, 3], TRUE))


def test_construction_checks_catch_planted_defects():
    W, nb, own, pop, owner, pid, prim, root = toy()
    bad_root = np.array([0, 0, 0, 3, 4])                  # merges A's and B's candidates
    _, _, checks, _ = mo.merge_oracles(W, nb, own, pop, owner, pid, prim, bad_root, ari_fn=lambda v: 0.0)
    assert not checks["q3_ml_within_particle"]
    bad_prim = np.array([0, 0, 0, 2, -1])                 # candidate 2's "primary" belongs to A
    _, _, checks, _ = mo.merge_oracles(W, nb, own, pop, owner, pid, bad_prim, root, ari_fn=lambda v: 0.0)
    assert not checks["q4_primary"]
    spur_root = np.array([0, 0, 2, 3, 0])                 # spurious candidate absorbed
    _, _, checks, _ = mo.merge_oracles(W, nb, own, pop, owner, pid, prim, spur_root, ari_fn=lambda v: 0.0)
    assert not checks["q3_spurious_singletons"]


def test_local_components_merge_only_same_particle_pairs():
    pid = np.array([10, 10, 20, 20, -1, 10])
    assert mo.local_components(pid, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5)], 6) == [0, 0, 2, 2, 4, 0]
    assert mo.local_components(pid, [], 6) == [0, 1, 2, 3, 4, 5]
    assert mo.local_components(pid, [(4, 4)], 6)[4] == 4            # spurious never merges


# ---- readings and report --------------------------------------------------------------------- #
def summary(P=0.25, L=0.26, ML=0.33, M=0.36, H1=0.41, n_events=10):
    ari_ = {"P": P, "L": L, "ML": ML, "M": M, "H1": H1, "M+H@2": 0.45, "M+H@3": 0.5, "M+H@5": 0.55}
    st = {"n_pop": 1000, "n_correct": 190, "n_wrong_on_particle": 700, "n_wrong_on_spurious": 110,
          "n_correct_on_duplicate": 125}
    return {"ari": ari_, "stats": st, "n_events": n_events, "gates": {"construction_fail": []}}


def test_decomposition_and_readings():
    n = mo.head_numbers(summary())
    assert n["S"] == pytest.approx((0.36 - 0.25) / (0.41 - 0.25))
    assert n["unsharing"] == pytest.approx(0.05) and n["scope_limit"] == pytest.approx(0.03)
    assert n["realised"] == pytest.approx(0.01 / 0.08)
    assert n["reading"] == "MERGE" and not n["scope_ok"]              # local reach 0.727 < 0.80
    assert n["scope"] == pytest.approx(0.08 / 0.11)
    assert mo.head_numbers(summary(M=0.32))["reading"] == "UNSHARE"   # S = 0.4375
    assert mo.head_numbers(summary(M=0.33, ML=0.33))["reading"] == "MERGE"   # S = 0.5 exactly


def test_verdict_rules():
    mk = lambda s: {h: mo.head_numbers(s) for h in mo.HEADS}
    v = mo.verdict(mk(summary()))
    assert v["overall"] == "MERGE" and not v["scope_ok"] and "widen the merge scope" in v["next"]
    v = mo.verdict(mk(summary(ML=0.355)))
    assert v["scope_ok"] and "local scope suffices" in v["next"]
    v = mo.verdict(mk(summary(M=0.30)))
    assert v["overall"] == "UNSHARE" and "class-information probe" in v["next"]
    nums = mk(summary())
    for h in mo.HEADS[:2]:
        nums[h] = mo.head_numbers(summary(M=0.30))
    assert mo.verdict(nums)["overall"] == "NOT ESTABLISHED"


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_tree(tmp, q1_diff=0, q2_diff=0):
    merge_root, rank_root, mb_root = tmp / "merge", tmp / "rank", tmp / "mb"
    for h in mo.HEADS:
        (merge_root / h).mkdir(parents=True)
        (merge_root / h / "merge_summary.json").write_text(json.dumps(summary()))
        write_csv(merge_root / h / "merge_events.csv", ["batch", "event", "P", "H1", "L"],
                  [{"batch": 0, "event": e, "P": 0.25, "H1": 0.41, "L": 0.26} for e in range(10)])
        write_csv(rank_root / h / "rank_events.csv", ["batch", "event", "pick_only", "head@1"],
                  [{"batch": 0, "event": e, "pick_only": 0.25 + (0.01 if e < q1_diff else 0), "head@1": 0.41}
                   for e in range(10)])
    for h, sub in mo.EP29_DIAG.items():
        write_csv(mb_root / sub / "diagnose_events.csv", ["batch", "event", "variant", "ari_particle"],
                  [{"batch": 0, "event": e, "variant": v, "ari_particle": x}
                   for e in range(10) for v, x in (("pick_merge", 0.26 + (0.01 if e < q2_diff else 0)),
                                                   ("pick_only", 0.25))])
    return merge_root, rank_root, mb_root


def test_report_passes_and_renders(tmp_path):
    res = mo.evaluate(*build_tree(tmp_path))
    assert res["gates"]["pass"], res["gates"]["failures"]
    assert res["verdict"]["overall"] == "MERGE"
    md = mo.render(res)
    assert "**Gates:** PASS" in md and "| **S** = (M − P)/(H1 − P) |" in md


def test_q1_and_q2_fail_beyond_the_flip_tolerance(tmp_path):
    assert mo.evaluate(*build_tree(tmp_path / "a", q1_diff=2))["gates"]["pass"]
    fails = mo.evaluate(*build_tree(tmp_path / "b", q1_diff=3))["gates"]["failures"]
    assert any(f.startswith("Q1 ") and " P:" in f for f in fails)
    fails = mo.evaluate(*build_tree(tmp_path / "c", q2_diff=3))["gates"]["failures"]
    assert any(f.startswith("Q2 ep29_s1") for f in fails)


def test_construction_failure_invalidates(tmp_path):
    root = build_tree(tmp_path)
    s = summary()
    s["gates"]["construction_fail"] = [{"event": [0, 1], "q4_primary": False}]
    (root[0] / "T_s1" / "merge_summary.json").write_text(json.dumps(s))
    res = mo.evaluate(*root)
    assert not res["gates"]["pass"] and res["verdict"]["overall"] == "INVALID"


def test_missing_head_is_refused(tmp_path):
    root = build_tree(tmp_path)
    (root[0] / "C_s2" / "merge_summary.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        mo.evaluate(*root)


# ---- run end to end (torch, materializer: EAF) -------------------------------------------- #
def test_run_end_to_end_on_a_synthetic_cache(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    import sys
    spec = importlib.util.spec_from_file_location("soft_particle_m", ROOT / "scripts/particle_hierarchy/soft_particle.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
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
    monkeypatch.setattr(sys, "argv", ["merge_oracle.py", "run", "--head", str(head), "--cache", str(cache),
                                      "--batches", "0:1", "--out", str(out), "--device", "cpu"])
    mo.main()
    res = json.loads((out / "merge_summary.json").read_text())
    assert res["n_events"] == 3 and res["gates"]["construction_fail"] == []
    A = res["ari"]
    assert A["M"] >= A["ML"] - 1e-12 and not math.isnan(A["L"])
    # the learned-merge column is the evaluated diagnose path's pick_merge
    rows = list(csv.DictReader(open(out / "merge_events.csv")))
    for r in rows:
        ev = next(e for e in evs if e["event"] == int(r["event"]))
        d = sp.prepare(ev, sp._import_materialize(), sp.s0.make_rule("fixed1:0.5"), 20, 0.3,
                       float("inf"), torch.device("cpu"))
        ref = [x for x in sp.diagnose_event(model.eval(), d) if x["variant"] == "pick_merge"][0]
        assert float(r["L"]) == pytest.approx(ref["ari_particle"], abs=1e-12)
        po = [x for x in sp.diagnose_event(model.eval(), d) if x["variant"] == "pick_only"][0]
        assert float(r["P"]) == pytest.approx(po["ari_particle"], abs=1e-12)


def test_run_refuses_beyond_dev_a(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import sys
    monkeypatch.setattr(sys, "argv", ["merge_oracle.py", "run", "--head", "x", "--cache", "y",
                                      "--batches", "150:314", "--out", str(tmp_path), "--device", "cpu"])
    with pytest.raises(SystemExit, match="dev-A only"):
        mo.main()
