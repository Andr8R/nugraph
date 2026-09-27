"""Tests for Stage 1a-R9 (consistency vs accuracy): label channels, co-assignment / channel counters, report driver.

The counters and report are numpy / pure python; the channel construction and end-to-end tests need torch (EAF).
Run:  pytest -q tests/test_cons_acc.py
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


ca = _load("cons_acc", ROOT / "scripts/particle_hierarchy/cons_acc.py")
rd = ca.rd
N_EV = 200


# ---- counters (numpy) -------------------------------------------------------------------------- #
def test_pair_counts():
    part = np.array([1, 1, 1, 2, 2, 3])
    lab = np.array([5, 5, 6, 7, 8, 9])
    assert rd.pair_counts(part, lab) == (4, 1)
    assert rd.pair_counts(np.array([1]), np.array([0])) == (0, 0)


def test_coassignment_and_channel_counters_on_a_hand_built_event():
    """Candidates c0 (particle 10), c1 (20), c2 (10, duplicate). Hits of particle 30 have no own candidate."""
    cand_pid, cand_cls, cand_pcls = np.array([10, 20, 10]), np.array([0, 3, 0]), np.array([0, 3, 0])
    hit_pid = np.array([10, 10, 10, 20, 20, 30, 30])
    nb = np.array([[0, 1], [2, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1]])
    own = cand_pid[nb] == hit_pid[:, None]
    W = np.tile([0.7, 0.3], (7, 1))                      # everyone picks column 0: c0 c2 c0 c1 c0 c0 c0
    hit_cls = np.array([0, 0, 0, 3, 3, 4, 4])
    hit_pcls = np.array([0, 1, 0, 3, 2, 4, 4])           # trunk wrong on h1 (pion) and h4 (hadron)
    owner = np.array([0, 0, 0, 1, 1, -1, -1])
    _, st, _ = rd.analyse_event(W, nb, own, np.ones(7, bool), owner, cand_pid, cand_cls, hit_cls,
                                ari_fn=lambda v: 0.0, hit_pcls=hit_pcls, cand_pcls=cand_pcls, hit_pid=hit_pid,
                                chan_cls=hit_cls.copy())
    assert (st["coassign_avail_pairs"], st["coassign_avail_same"]) == (4, 1)     # {c0,c2,c0} and {c1,c0}
    assert (st["coassign_unavail_pairs"], st["coassign_unavail_same"]) == (1, 1)
    assert (st["ch_n"], st["ch_correct"], st["trunk_correct"]) == (7, 7, 5)
    assert (st["trunk_err"], st["ch_fixed"], st["trunk_err_track"], st["ch_fixed_track"]) == (2, 2, 2, 2)
    assert st["ch_broken"] == 0
    assert (st["ch_pairs"], st["ch_pairs_same"], st["trunk_pairs_same"]) == (5, 5, 2)


def test_counters_absent_by_default():
    _, st, _ = rd.analyse_event(np.array([[0.7, 0.3]]), np.array([[0, 1]]), np.array([[False, True]]),
                                np.array([True]), np.array([1]), np.array([5, 6]), np.array([0, 1]),
                                np.array([1]), ari_fn=lambda v: 0.0)
    assert "coassign_avail_pairs" not in st and "ch_n" not in st


# ---- report (pure python) --------------------------------------------------------------------- #
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "event", "variant", "ari_particle", "ari_hardseed"])
        w.writeheader()
        w.writerows(rows)


def meta(arm, t, **over):
    m = {"m": 20, "seed": t, "hit_dim": 136, "class_channels": ca.MODE[arm], "split_channels": 0,
         "oracle_upper_bound": arm != "CTRL", "init_sha256": f"i{t}", "perm_sha256": [f"p{t}{e}" for e in range(8)],
         "best_epoch": 5, "best_dev_ari": 0.3, "git_commit": "c", "git_dirty": False,
         "latent_checkpoint_sha256": "ck", "argv": [arm]}
    m.update(over)
    return m


CH = {  # valid construction per label arm: (accuracy, fixed, fixed_track, broken, pair_same)
    "TRUECLS": (1000, 180, 100, 0, 5000), "CONS": (880, 120, 70, 20, 5000), "FIX50": (910, 90, 50, 0, 4800),
    "FIXTRACK": (920, 100, 100, 0, 4850)}


def rank(arm, ch=None, co=None):
    st = {"n_pop": 10000, "n_avail": 4000, "n_correct": 2000, "n_miss": 1000, "owner_missing": 0, "demoted": 100,
          "recall_head": [2000, 2600, 2900, 3200, 3700, 4000], "recall_dist": [1600, 2300, 2700, 3000, 3500, 4000],
          "miss_rank_head": [300, 150, 200, 230, 120], "miss_rank_dist": [140, 300, 150, 200, 110, 100],
          "p1_ok": [0] * 150 + [2000] + [0] * 49, "p1_miss": [0] * 60 + [1000] + [0] * 139, "margin_ok_low": 900,
          "margin_miss_low": 700, "err_spurious": 80, "err_same_class": 400, "err_diff_class": 520}
    ua, uu = co or ((0.40, 0.30) if arm == "CTRL" else (0.50, 0.55))
    st.update({"coassign_avail_pairs": 1000, "coassign_avail_same": int(ua * 1000),
               "coassign_unavail_pairs": 1000, "coassign_unavail_same": int(uu * 1000)})
    if arm in ca.LABEL_ARMS:
        acc, fixed, fixed_track, broken, same = ch or CH[arm]
        st.update({"ch_n": 1000, "ch_correct": acc, "trunk_correct": 820, "trunk_err": 180, "ch_fixed": fixed,
                   "trunk_err_track": 100, "ch_fixed_track": fixed_track, "ch_broken": broken, "ch_pairs": 5000,
                   "ch_pairs_same": same, "trunk_pairs_same": 4500})
    return {"ari": {"pick_only": 0.25, **{f"head@{k}": 0.40 + 0.01 * i for i, k in enumerate(rd.KS)},
                    **{f"dist@{k}": 0.36 + 0.01 * i for i, k in enumerate(rd.KS)}},
            "stats": st, "n_events": N_EV,
            "gates": {"R1_pick_only_fail": [], "R3_head_eq_dist_at_M_fail": [], "R4_owner_missing": 0}}


def build(tmp, gains=None, gains2=None, meta_over=None, r6_gain=0.17, hard_shift=0.0, ch_over=None, co_over=None,
          readout_pass=True):
    root, r6 = tmp / "root", tmp / "r6"
    g1 = {"CTRL": 0.0, "ORACLE": 0.17, "TRUECLS": 0.16, "CONS": 0.12, "FIX50": 0.02, "FIXTRACK": 0.01,
          **(gains or {})}
    g = {1: g1, 2: {**g1, **(gains2 or {})}}
    rng = random.Random(5)
    base = [(0.25 + 0.1 * rng.random(), 0.15 + 0.05 * rng.random()) for _ in range(N_EV)]

    def rows(gain, shift=False):
        return [{"batch": e // 8, "event": e % 8, "variant": var, "ari_particle": a0 + gain + rng.gauss(0, 0.01),
                 "ari_hardseed": h + (hard_shift if shift and e == 3 else 0.0)}
                for e, (a0, h) in enumerate(base) for var in ("v2_nonull", "pick_only")]

    for split in ("train", "devA"):
        p = root / "readout" / f"{split}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"pass": readout_pass, "fg_reclassified_diffuse": 0.0, "checkpoint_sha256": "ck"}))
    for t in ca.SEEDS:
        for arm in ca.ARMS:
            d = root / "heads" / ca.name(arm, t)
            d.mkdir(parents=True)
            (d / "train_meta.json").write_text(json.dumps(meta(arm, t, **((meta_over or {}).get((arm, t), {})))))
            for sub in ("diag_last", "diag_best"):
                write_csv(root / sub / ca.name(arm, t) / "diagnose_events.csv", rows(g[t][arm], arm == "FIX50"))
            rk = root / "rank" / ca.name(arm, t)
            rk.mkdir(parents=True)
            (rk / "rank_summary.json").write_text(json.dumps(rank(arm, (ch_over or {}).get(arm),
                                                                  (co_over or {}).get(arm))))
        for arm, gg in (("CTRL", 0.0), ("ORACLE", r6_gain)):
            write_csv(r6 / "diag_last" / f"{arm}_s{t}" / "diagnose_events.csv", rows(gg))
    return root, r6


def test_shares_are_recovered(tmp_path):
    res = ca.evaluate(*build(tmp_path))
    assert res["gates"]["pass"], res["gates"]["failures"]
    for t in ca.SEEDS:
        p, ci = res["seeds"][t]["decomp"]["point"], res["seeds"][t]["decomp"]["ci95"]
        assert p["G_T"] == pytest.approx(0.16, abs=0.005)
        assert p["C"] == pytest.approx(0.12 / 0.16, abs=0.03) and p["A"] == pytest.approx(0.02 / 0.16, abs=0.03)
        assert p["bucket7_share"] == pytest.approx(0.01 / 0.17, abs=0.03)
        assert ci["C"][0] < p["C"] < ci["C"][1]
        # efficiency: (0.01 / 100 track fixes) / (0.02 / 90 fixes)
        assert res["seeds"][t]["efficiency_track"] == pytest.approx((0.01 / 100) / (0.02 / 90), abs=0.15)
    assert res["verdict"]["reading"] == "CONSISTENCY" and "T without SEM" in res["verdict"]["next"]
    assert res["verdict"]["coassign_hypothesis"] == "SUPPORTED"


@pytest.mark.parametrize("gains, reading", [
    ({"CONS": 0.12, "FIX50": 0.02}, "CONSISTENCY"),
    ({"CONS": 0.02, "FIX50": 0.08}, "ACCURACY"),
    ({"CONS": 0.08, "FIX50": 0.08}, "BOTH"),
    ({"CONS": 0.01, "FIX50": 0.01}, "NEITHER"),
    ({"CONS": 0.06, "FIX50": 0.02}, "PARTIAL"),      # C = 0.375: neither >= 0.50 nor < 0.15
])
def test_readings_fixed_in_advance(tmp_path, gains, reading):
    assert ca.evaluate(*build(tmp_path, gains=gains))["verdict"]["reading"] == reading


def test_seeds_must_agree(tmp_path):
    res = ca.evaluate(*build(tmp_path, gains2={"CONS": 0.02, "FIX50": 0.08}))
    assert [res["seeds"][t]["reading"] for t in ca.SEEDS] == ["CONSISTENCY", "ACCURACY"]
    assert res["verdict"]["reading"] == "PARTIAL"


def test_a_more_accurate_cons_confounds_a_consistency_reading(tmp_path):
    res = ca.evaluate(*build(tmp_path, ch_over={"CONS": (950, 150, 90, 20, 5000)}))
    assert res["gates"]["pass"] and res["verdict"]["reading"] == "CONFOUNDED"


def test_coassignment_not_supported_when_available_pairs_gain_more(tmp_path):
    res = ca.evaluate(*build(tmp_path, co_over={"TRUECLS": (0.60, 0.35)}))
    assert res["verdict"]["coassign_hypothesis"] == "NOT SUPPORTED"


@pytest.mark.parametrize("kw, gate", [
    ({"readout_pass": False}, "D0 train"),
    ({"meta_over": {("CONS", 2): {"latent_checkpoint_sha256": "x"}}}, "D0 CONS_s2"),
    ({"meta_over": {("FIX50", 1): {"init_sha256": "x"}}}, "D1 seed 1: FIX50"),
    ({"meta_over": {("TRUECLS", 2): {"perm_sha256": ["x"] * 8}}}, "D2 seed 2: TRUECLS"),
    ({"meta_over": {("FIXTRACK", 1): {"lr": 0.01}}}, "D3 seed 1: FIXTRACK recipe differs"),
    ({"meta_over": {("CTRL", 2): {"git_dirty": True}}}, "D3 CTRL_s2"),
    ({"hard_shift": 1e-6}, "D4 seed 1: FIX50 hard-seed"),
    ({"r6_gain": 0.10}, "D5 seed 1"),
    ({"ch_over": {"TRUECLS": (999, 180, 100, 0, 5000)}}, "D6 TRUECLS_s1: accuracy"),
    ({"ch_over": {"CONS": (880, 120, 70, 20, 4999)}}, "D6 CONS_s1: within-particle"),
    ({"ch_over": {"FIX50": (910, 95, 50, 0, 4800)}}, "D6 FIX50_s1: corrected fraction"),
    ({"ch_over": {"FIXTRACK": (920, 101, 100, 0, 4850)}}, "D6 FIXTRACK_s1: fixed track"),
    ({"ch_over": {"FIX50": (910, 90, 50, 3, 4800)}}, "D6 FIX50_s1: 3 correct"),
])
def test_each_gate_fails_on_its_planted_defect(tmp_path, kw, gate):
    res = ca.evaluate(*build(tmp_path, **kw))
    assert not res["gates"]["pass"] and res["verdict"]["reading"] == "INVALID"
    assert any(f.startswith(gate) for f in res["gates"]["failures"]), res["gates"]["failures"]


def test_old_rank_summaries_fail_d6(tmp_path):
    root, r6 = build(tmp_path)
    p = root / "rank" / "CONS_s1" / "rank_summary.json"
    s = json.loads(p.read_text())
    del s["stats"]["ch_pairs"]
    p.write_text(json.dumps(s))
    res = ca.evaluate(root, r6)
    assert any(f.startswith("D6 CONS_s1: channel counters missing") for f in res["gates"]["failures"])


def test_render_labels_follow_the_arms(tmp_path):
    res = ca.evaluate(*build(tmp_path))
    md = ca.render(res)
    hdr = next(l for l in md.splitlines() if l.startswith("| ARI (last epoch)"))
    assert [x.strip() for x in hdr.split("|")[2:-1]] == list(ca.ARMS)
    row = next(l for l in md.splitlines() if l.startswith("| seed 1 |"))
    vals = dict(zip(ca.ARMS, (float(x) for x in row.split("|")[2:-1])))
    assert vals["TRUECLS"] - vals["CTRL"] == pytest.approx(0.16, abs=0.01)
    assert vals["CONS"] - vals["CTRL"] == pytest.approx(0.12, abs=0.01)
    assert "ORACLE UPPER BOUND" in md and "**Reading:** CONSISTENCY" in md


# ---- label channels (torch, EAF) --------------------------------------------------------------- #
def _torch_sp():
    torch = pytest.importorskip("torch")
    return torch, _load("soft_particle_ca", ROOT / "scripts/particle_hierarchy/soft_particle.py")


AM = [0, 0, 1, 3, 2, 2, 3, 5, 4, 5]      # trunk argmax per hit


def label_event(torch, batch=0, event=0):
    return {"fg_idx": torch.arange(10), "hit_x": torch.nn.functional.one_hot(torch.tensor(AM), 7).float(),
            "y_i": torch.tensor([0, 0, 0, 1, 1, 1, 1, -1, 2, 2]),
            "y_sem": torch.tensor([0, 0, 0, 3, 3, 3, 3, -1, 4, 4]), "batch": batch, "event": event}




def readout(torch):
    return (torch.eye(7), torch.zeros(7), "sha")                  # argmax(read-out) = argmax(hit_x)


@pytest.mark.parametrize("mode, want", [
    ("trueclass", [0, 0, 0, 3, 3, 3, 3, 5, 4, 4]),
    ("cons", [0, 0, 0, 2, 2, 2, 2, 5, 4, 4]),        # particle 1 ties 2:2 -> lowest; particle 2 ties -> 4
    ("fixtrack", [0, 0, 0, 3, 3, 3, 3, 5, 4, 5]),    # shower error (hit 9) left alone
])
def test_label_classes(mode, want):
    torch, sp = _torch_sp()
    got = sp.label_classes(label_event(torch), mode, readout(torch))
    assert got.tolist() == want
    ch = sp.class_channels(label_event(torch), mode, readout(torch))
    assert ch.shape == (10, sp.CLASS_BUCKETS) and ch.argmax(1).tolist() == want and not ch[:, 7].any()


def test_fix50_corrects_exactly_half_deterministically_without_the_global_rng():
    torch, sp = _torch_sp()
    errors = {2, 4, 5, 9}
    state = torch.get_rng_state()
    a = sp.label_classes(label_event(torch), "fix50", readout(torch)).tolist()
    assert torch.equal(state, torch.get_rng_state())
    assert a == sp.label_classes(label_event(torch), "fix50", readout(torch)).tolist()
    truth = [0, 0, 0, 3, 3, 3, 3, 5, 4, 4]
    fixed = {i for i in errors if a[i] == truth[i]}
    assert len(fixed) == 2
    assert all(a[i] == AM[i] for i in range(10) if i not in fixed)
    picks = {tuple(sp.label_classes(label_event(torch, event=e), "fix50", readout(torch)).tolist()) for e in range(20)}
    assert len(picks) > 1                                          # the choice depends on the event


def test_label_modes_need_the_readout():
    torch, sp = _torch_sp()
    for mode in sp.LABEL_MODES:
        with pytest.raises(ValueError):
            sp.class_channels(label_event(torch), mode)


def test_six_arms_share_init_and_order_and_ranking_counts_the_channels(tmp_path, monkeypatch):
    """End to end: six R9 arms share init and order (D1/D2); ranking_diag records the D6 counters."""
    torch, sp = _torch_sp()
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    tcp = _load("test_class_probe_ca", ROOT / "tests/test_class_probe.py")
    cache = tcp.synthetic_cache(tmp_path)
    ck = tmp_path / "trunk.ckpt"
    torch.manual_seed(3)
    torch.save({"state_dict": {"semantic_decoder.net.weight": torch.randn(7, 3),
                               "semantic_decoder.net.bias": torch.zeros(7)}}, ck)
    (cache / "manifest.json").write_text(json.dumps({"checkpoint": str(ck), "checkpoint_sha256": sp.mb.sha256(ck)}))
    metas = {}
    for arm, mode in ca.MODE.items():
        sp._READOUT.clear()
        _, metas[arm] = tcp.train(tmp_path, cache, mode, monkeypatch)
    for arm in ca.ARMS[1:]:
        assert metas[arm]["init_sha256"] == metas["CTRL"]["init_sha256"]
        assert metas[arm]["perm_sha256"] == metas["CTRL"]["perm_sha256"]
        diff = {k for k in set(metas[arm]) | set(metas["CTRL"]) if metas[arm].get(k) != metas["CTRL"].get(k)}
        assert diff <= ca.cp.OUTCOME_KEYS | {"argv", "out"}, diff
    assert [metas[a]["oracle_upper_bound"] for a in ca.ARMS] == [False] + [True] * 5
    rdm = _load("ranking_diag_ca", ROOT / "scripts/particle_hierarchy/ranking_diag.py")
    for mode in ("cons", "trueclass"):
        out = tmp_path / f"rank_{mode}"
        monkeypatch.setattr(sys, "argv", ["ranking_diag.py", "run", "--head", str(tmp_path / f"head_{mode}" / "head.pt"),
                                          "--cache", str(cache), "--batches", "0:1", "--out", str(out),
                                          "--device", "cpu", "--pred-class"])
        rdm.main()
        st = json.loads((out / "rank_summary.json").read_text())["stats"]
        assert ca.construction(st, ca.ARMS[2 + ("trueclass", "cons").index(mode)]) == []
        assert st["coassign_avail_pairs"] + st["coassign_unavail_pairs"] > 0
