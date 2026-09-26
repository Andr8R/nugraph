"""Tests for Stage 1a-R8 (class-truth decomposition): encoder-block isolation, split channels, geometry counters,
and the report driver (gates, shares, interaction, readings).

The report and counters are numpy / pure python; the isolation and end-to-end tests need torch (EAF).
Run:  pytest -q tests/test_class_decomp.py
"""
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cd = _load("class_decomp", ROOT / "scripts/particle_hierarchy/class_decomp.py")
rd = cd.rd
N_EV = 200


# ---- geometry counters (numpy) --------------------------------------------------------------- #
def geo_event(picked_col_first):
    """One different-class, both-predicted-correctly miss: hit (pion of particle 10) links to c1 (hadron of 20);
    its owner c0 is a correctly predicted pion. Candidate columns are in distance order: c1 is column 0 (nearer)
    when picked_col_first, else column 1 (farther). A second hit's owner c2 has a MISPREDICTED seed."""
    cand_pid, cand_cls, cand_pcls = np.array([10, 20, 30]), np.array([1, 3, 1]), np.array([1, 3, 3])
    if picked_col_first:
        nb, W = np.array([[1, 0], [1, 0], [1, 2]]), np.tile([0.7, 0.3], (3, 1))
    else:
        nb, W = np.array([[0, 1], [0, 1], [2, 1]]), np.tile([0.3, 0.7], (3, 1))
    y = np.array([10, 20, 30])
    own = cand_pid[nb] == y[:, None]
    hit_cls, hit_pcls = np.array([1, 3, 1]), np.array([1, 3, 1])
    _, st, _ = rd.analyse_event(W, nb, own, np.ones(3, bool), np.zeros(3, int), cand_pid, cand_cls, hit_cls,
                                ari_fn=lambda v: 0.0, hit_pcls=hit_pcls, cand_pcls=cand_pcls)
    return st


def test_geometry_counters_on_hand_built_events():
    near = geo_event(True)
    # hit 0: both-correct miss, picked nearer, owner seed correct; hit 2 (pion of 30) picks c1 (hadron, correct
    # seed) while its owner c2's seed is mispredicted -> both-correct miss with owner_seed_wrong
    assert near["dm_both_correct"] == 2
    assert near["dm_both_correct_owner_seed_wrong"] == 1
    assert near["dm_both_correct_picked_nearer"] == 2
    assert near["dm_both_correct_nearer_owner_seed_ok"] == 1
    far = geo_event(False)
    assert far["dm_both_correct"] == 2 and far["dm_both_correct_picked_nearer"] == 0
    assert far["dm_both_correct_nearer_owner_seed_ok"] == 0 and far["dm_both_correct_owner_seed_wrong"] == 1


# ---- report (pure python) --------------------------------------------------------------------- #
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "event", "variant", "ari_particle", "ari_hardseed"])
        w.writeheader()
        w.writerows(rows)


def meta(arm, t, **over):
    m = {"m": 20, "seed": t, "hit_dim": 144, "class_channels": cd.MODE[arm], "split_channels": 8,
         "oracle_upper_bound": "oracle" in cd.MODE[arm], "init_sha256": f"i{t}",
         "perm_sha256": [f"p{t}{e}" for e in range(8)], "best_epoch": 5, "best_dev_ari": 0.3, "git_commit": "c",
         "git_dirty": False, "latent_checkpoint_sha256": "ck", "argv": [arm]}
    m.update(over)
    return m


def rank(n_bc=400, osw=0.3, nearer_ok=0.5):
    st = {"n_pop": 10000, "n_avail": 4000, "n_correct": 2000, "n_miss": 1000, "owner_missing": 0, "demoted": 100,
          "recall_head": [2000, 2600, 2900, 3200, 3700, 4000], "recall_dist": [1600, 2300, 2700, 3000, 3500, 4000],
          "miss_rank_head": [300, 150, 200, 230, 120], "miss_rank_dist": [140, 300, 150, 200, 110, 100],
          "p1_ok": [0] * 150 + [2000] + [0] * 49, "p1_miss": [0] * 60 + [1000] + [0] * 139, "margin_ok_low": 900,
          "margin_miss_low": 700, "err_spurious": 80, "err_same_class": 400, "err_diff_class": 520,
          "dm_cand_disguised": 150, "dm_hit_disguised": 50, "dm_both_wrong": 20, "dm_seed_wrong_third": 0,
          "dm_hit_wrong_third": 0, "miss_owner_seed_wrong": 600, "ok_owner_seed_wrong": 500,
          "dm_both_correct": n_bc, "dm_both_correct_owner_seed_wrong": int(osw * n_bc),
          "dm_both_correct_picked_nearer": int(0.8 * n_bc), "dm_both_correct_nearer_owner_seed_ok": int(nearer_ok * n_bc)}
    return {"ari": {"pick_only": 0.25, **{f"head@{k}": 0.40 + 0.01 * i for i, k in enumerate(rd.KS)},
                    **{f"dist@{k}": 0.36 + 0.01 * i for i, k in enumerate(rd.KS)}},
            "stats": st, "n_events": N_EV,
            "gates": {"R1_pick_only_fail": [], "R3_head_eq_dist_at_M_fail": [], "R4_owner_missing": 0}}


def build(tmp, gains=None, gains2=None, meta_over=None, r6_gain=0.17, hard_shift=0.0, geo=None, readout_pass=True):
    """Planted per-arm ARI gains vs CTRL (seed 1; seed 2 = gains2 or the same)."""
    root, r6 = tmp / "root", tmp / "r6"
    g1 = {"CTRL": 0.0, "ORACLE": 0.17, "SEED": 0.10, "HIT": 0.03, "HARD": 0.0, **(gains or {})}
    g = {1: g1, 2: {**g1, **(gains2 or {})}}
    rng = random.Random(5)
    base = [(0.25 + 0.1 * rng.random(), 0.15 + 0.05 * rng.random()) for _ in range(N_EV)]

    def rows(gain, shift=False):
        out = []
        for e, (a0, h) in enumerate(base):
            for var in ("v2_nonull", "pick_only"):
                out.append({"batch": e // 8, "event": e % 8, "variant": var,
                            "ari_particle": a0 + gain + rng.gauss(0, 0.01),
                            "ari_hardseed": h + (hard_shift if shift and e == 3 else 0.0)})
        return out

    for split in ("train", "devA"):
        p = root / "readout" / f"{split}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"pass": readout_pass, "fg_reclassified_diffuse": 0.0 if readout_pass else 0.01,
                                 "checkpoint_sha256": "ck"}))
    for t in cd.SEEDS:
        for arm in cd.ARMS:
            d = root / "heads" / cd.name(arm, t)
            d.mkdir(parents=True)
            (d / "train_meta.json").write_text(json.dumps(meta(arm, t, **((meta_over or {}).get((arm, t), {})))))
            for sub in ("diag_last", "diag_best"):
                write_csv(root / sub / cd.name(arm, t) / "diagnose_events.csv", rows(g[t][arm], arm == "HARD"))
            r = root / "rank" / cd.name(arm, t)
            r.mkdir(parents=True)
            (r / "rank_summary.json").write_text(json.dumps(rank(**((geo or {}) if arm == "CTRL" else {}))))
        for arm, gg in (("CTRL", 0.0), ("ORACLE", r6_gain)):
            write_csv(r6 / "diag_last" / f"{arm}_s{t}" / "diagnose_events.csv", rows(gg))
    return root, r6


def test_shares_and_interaction_are_recovered(tmp_path):
    res = cd.evaluate(*build(tmp_path, gains={"SEED": 0.10, "HIT": 0.03, "HARD": 0.0}))
    assert res["gates"]["pass"], res["gates"]["failures"]
    for t in cd.SEEDS:
        p, ci = res["seeds"][t]["decomp"]["point"], res["seeds"][t]["decomp"]["ci95"]
        assert p["G"] == pytest.approx(0.17, abs=0.005)
        assert p["S_seed"] == pytest.approx(0.10 / 0.17, abs=0.03)
        assert p["S_hit"] == pytest.approx(0.03 / 0.17, abs=0.03)
        assert p["I_over_G"] == pytest.approx(0.04 / 0.17, abs=0.03)   # complementary: 0.10 + 0.03 < 0.17
        for k in ("S_seed", "S_hit", "I_over_G"):
            assert ci[k][0] < p[k] < ci[k][1]
    assert res["verdict"]["reading"] == "SEED" and res["verdict"]["hard"] == "NO-EFFECT"


def test_redundant_sides_give_negative_interaction(tmp_path):
    res = cd.evaluate(*build(tmp_path, gains={"SEED": 0.12, "HIT": 0.12}))
    p = res["seeds"][1]["decomp"]["point"]
    assert p["I_over_G"] == pytest.approx((0.17 - 0.24) / 0.17, abs=0.03) and p["I_over_G"] < 0
    assert res["verdict"]["reading"] == "EITHER" and "seeds first" in res["verdict"]["next"]


@pytest.mark.parametrize("gains, reading", [
    ({"SEED": 0.10, "HIT": 0.03}, "SEED"),
    ({"SEED": 0.03, "HIT": 0.12}, "HIT"),
    ({"SEED": 0.06, "HIT": 0.12}, "PARTIAL"),        # S_hit >= 0.5 but S_seed in [0.30, 0.50)
    ({"SEED": 0.12, "HIT": 0.12}, "EITHER"),
    ({"SEED": 0.03, "HIT": 0.03}, "JOINT"),          # I/G = 0.65
    ({"SEED": 0.07, "HIT": 0.07}, "PARTIAL"),        # I/G = 0.18
])
def test_readings_fixed_in_advance(tmp_path, gains, reading):
    assert cd.evaluate(*build(tmp_path, gains=gains))["verdict"]["reading"] == reading


def test_seeds_must_agree(tmp_path):
    res = cd.evaluate(*build(tmp_path, gains={"SEED": 0.10, "HIT": 0.03}, gains2={"SEED": 0.03, "HIT": 0.12}))
    assert [res["seeds"][t]["reading"] for t in cd.SEEDS] == ["SEED", "HIT"]
    assert res["verdict"]["reading"] == "PARTIAL"


@pytest.mark.parametrize("hard, hard2, want", [
    (0.05, 0.05, "DECISIVE"), (0.0, 0.005, "NO-EFFECT"), (0.015, 0.015, "INCONCLUSIVE"), (0.05, 0.0, "INCONCLUSIVE")])
def test_pred_hard_is_read_independently(tmp_path, hard, hard2, want):
    res = cd.evaluate(*build(tmp_path, gains={"HARD": hard}, gains2={"HARD": hard2}))
    assert res["verdict"]["hard"] == want and res["verdict"]["reading"] == "SEED"


@pytest.mark.parametrize("geo, want", [
    ({"osw": 0.6, "nearer_ok": 0.3}, "SEED-EFFECT"), ({"osw": 0.2, "nearer_ok": 0.75}, "GEOMETRY"),
    ({"osw": 0.3, "nearer_ok": 0.5}, "UNRESOLVED")])
def test_geometry_reading(tmp_path, geo, want):
    res = cd.evaluate(*build(tmp_path, geo=geo))
    assert res["verdict"]["geometry"] == want


@pytest.mark.parametrize("kw, gate", [
    ({"meta_over": {("SEED", 1): {"init_sha256": "x"}}}, "D1 seed 1: SEED"),
    ({"meta_over": {("HIT", 2): {"perm_sha256": ["x"] * 8}}}, "D2 seed 2: HIT"),
    ({"meta_over": {("HARD", 1): {"lr": 0.01}}}, "D3 seed 1: HARD recipe differs"),
    ({"meta_over": {("ORACLE", 2): {"split_channels": 0}}}, "D3 ORACLE_s2"),
    ({"meta_over": {("CTRL", 1): {"git_dirty": True}}}, "D3 CTRL_s1"),
    ({"meta_over": {("HARD", 2): {"latent_checkpoint_sha256": "other"}}}, "D0 HARD_s2"),
    ({"readout_pass": False}, "D0 train"),
    ({"hard_shift": 1e-6}, "D4 seed 1: HARD hard-seed"),
    ({"r6_gain": 0.10}, "D5 seed 1"),
])
def test_each_gate_fails_on_its_planted_defect(tmp_path, kw, gate):
    res = cd.evaluate(*build(tmp_path, **kw))
    assert not res["gates"]["pass"] and res["verdict"]["reading"] == "INVALID"
    assert any(f.startswith(gate) for f in res["gates"]["failures"]), res["gates"]["failures"]


def test_render_labels_follow_the_arms(tmp_path):
    res = cd.evaluate(*build(tmp_path, gains={"SEED": 0.10, "HIT": 0.03, "HARD": 0.0}))
    md = cd.render(res)
    hdr = next(l for l in md.splitlines() if l.startswith("| ARI (last epoch)"))
    assert [x.strip() for x in hdr.split("|")[2:-1]] == list(cd.ARMS)
    row = next(l for l in md.splitlines() if l.startswith("| seed 1 |"))
    vals = dict(zip(cd.ARMS, (float(x) for x in row.split("|")[2:-1])))
    assert vals["ORACLE"] - vals["CTRL"] == pytest.approx(0.17, abs=0.01)
    assert vals["SEED"] - vals["CTRL"] == pytest.approx(0.10, abs=0.01)
    assert "**Reading:** SEED" in md and "inductive-bias probe" in md
    m = res["seeds"][1]["mech"]["CTRL"]
    assert m["cand_disguised"] == pytest.approx(150 / 620) and m["LR_seed"] == pytest.approx(0.6 / 0.25)


# ---- isolation and split channels (torch, EAF) ------------------------------------------------ #
def _torch_sp():
    torch = pytest.importorskip("torch")
    return torch, _load("soft_particle_cd", ROOT / "scripts/particle_hierarchy/soft_particle.py")


def test_encoder_blocks_are_isolated():
    torch, sp = _torch_sp()
    torch.manual_seed(0)
    k, F0, lat = sp.CLASS_BUCKETS, 3, 8
    m = sp.build_model(F0 + 2 * k, lat, 16, 2)
    m.split_k = k
    n = 12
    hit_x, ox, beta = torch.randn(n, F0 + 2 * k), torch.randn(n, lat), torch.rand(n)
    seed_idx = torch.tensor([0, 5, 9])
    h0, c0 = m.encode(hit_x, ox, beta, seed_idx)
    x = hit_x.clone()
    x[:, F0 + k:] = torch.randn(n, k)                    # change only the candidate block
    h1, c1 = m.encode(x, ox, beta, seed_idx)
    assert torch.equal(h0, h1) and not torch.allclose(c0, c1)
    x = hit_x.clone()
    x[:, F0:F0 + k] = torch.randn(n, k)                  # change only the hit block
    h2, c2 = m.encode(x, ox, beta, seed_idx)
    assert torch.equal(c0, c2) and not torch.allclose(h0, h2)
    m.split_k = 0                                        # unsplit heads (every earlier stage) see both
    h3, c3 = m.encode(x, ox, beta, seed_idx)
    assert not torch.allclose(c0, c3)


def split_event(torch):
    fg = torch.tensor([True] * 8 + [False, False])
    return {"fg_idx": fg.nonzero().squeeze(1), "hit_x": torch.rand(8, 3) + 0.1, "cp": torch.tensor([1, 4]),
            "y_sem": torch.tensor([0, 1, 2, 3, 4, 5, 6, -1, 2, 2], dtype=torch.int8)}


@pytest.mark.parametrize("mode", ["s_zeros", "s_oracle", "s_oracle_seed", "s_oracle_hit", "s_pred_hard"])
def test_split_blocks_per_mode(mode):
    torch, sp = _torch_sp()
    assert set(sp.SPLIT_MODES) == set(cd.MODE.values())
    d = split_event(torch)
    ro = (torch.eye(7, 3) * 5, torch.zeros(7), "sha")     # argmax of the read-out = the argmax of hit_x[:, :3]
    ch = sp.class_channels(d, mode, ro)
    k = sp.CLASS_BUCKETS
    assert ch.shape == (8, 2 * k)
    hb, cb = ch[:, :k], ch[:, k:]
    hit_spec, cand_spec = sp.SPLIT_MODES[mode]
    true = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]), k).float()
    hard = torch.nn.functional.one_hot(d["hit_x"].argmax(1), k).float()
    want = {None: torch.zeros(8, k), "true": true, "hard": hard}
    assert torch.equal(hb, want[hit_spec])
    seed = torch.zeros(8, 1)
    seed[d["cp"]] = 1
    assert torch.equal(cb, want[cand_spec] * seed)               # candidate block on seed rows only
    assert not cb[[0, 2, 3, 5, 6, 7]].any()
    if cand_spec == "hard":
        assert not hard[:, 7].any()                              # PRED-HARD never uses the unlabelled bucket


def test_pred_hard_needs_the_readout():
    torch, sp = _torch_sp()
    with pytest.raises(ValueError):
        sp.class_channels(split_event(torch), "s_pred_hard")


def test_five_arms_share_init_and_order_and_restore_the_split(tmp_path, monkeypatch):
    """End to end: all five R8 arms at one width share init and order (D1/D2); load_head restores split_k."""
    torch, sp = _torch_sp()
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    tcp = _load("test_class_probe_cd", ROOT / "tests/test_class_probe.py")
    cache = tcp.synthetic_cache(tmp_path)
    ck = tmp_path / "trunk.ckpt"
    torch.save({"state_dict": {"semantic_decoder.net.weight": torch.randn(7, 3),
                               "semantic_decoder.net.bias": torch.zeros(7)}}, ck)
    (cache / "manifest.json").write_text(json.dumps({"checkpoint": str(ck), "checkpoint_sha256": sp.mb.sha256(ck)}))
    metas = {}
    for arm, mode in cd.MODE.items():
        sp._READOUT.clear()
        out, metas[arm] = tcp.train(tmp_path, cache, mode, monkeypatch)
        assert metas[arm]["split_channels"] == sp.CLASS_BUCKETS
        assert metas[arm]["hit_dim"] == 3 + 2 * sp.CLASS_BUCKETS
        assert sp.load_head(torch.load(out / "head.pt", weights_only=False), "cpu").split_k == sp.CLASS_BUCKETS
    for arm in cd.ARMS[1:]:
        assert metas[arm]["init_sha256"] == metas["CTRL"]["init_sha256"]
        assert metas[arm]["perm_sha256"] == metas["CTRL"]["perm_sha256"]
        diff = {k for k in set(metas[arm]) | set(metas["CTRL"]) if metas[arm].get(k) != metas["CTRL"].get(k)}
        assert diff <= cd.cp.OUTCOME_KEYS | {"argv", "out"}, diff
    assert [metas[a]["oracle_upper_bound"] for a in cd.ARMS] == [False, True, True, True, False]
    # every tool rebuilds the split inputs from the head config
    out = tmp_path / "head_s_oracle_seed"
    monkeypatch.setattr(sys, "argv", ["soft_particle.py", "eval", "--head", str(out / "head.pt"), "--cache",
                                      str(cache), "--batches", "0:1", "--device", "cpu", "--out", str(out / "ev"),
                                      "--no-null"])
    sp.main()
    assert json.loads((out / "ev" / "dev_summary.json").read_text())["dev/n_events"] == 3
