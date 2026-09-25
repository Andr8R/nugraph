"""Tests for Stage 1a-R6 (predicted-class probe): addressability split, report driver, pred channels.

Addressability and the report are numpy / pure python; the channel / read-out tests need torch (EAF).
Run:  pytest -q tests/test_pred_class.py
"""
import copy
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


pc = _load("pred_class", ROOT / "scripts/particle_hierarchy/pred_class.py")
rd = pc.rd
cp = pc.cp
KS = rd.KS
N_EV = 200


# ---- addressability split (numpy) ------------------------------------------------------------ #
def test_addressability_splits_different_class_misses_by_predicted_class():
    W = np.array([[0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]])
    nb = np.array([[1, 0], [1, 0], [2, 0], [0, 1]])
    own = np.array([[False, True], [False, True], [False, True], [True, False]])
    pop = np.ones(4, bool)
    owner = np.array([0, 0, 0, 0])
    cand_pid = np.array([10, 20, 30])
    cand_cls = np.array([1, 2, 1])            # candidate 1 is class 2, candidate 2 class 1
    hit_cls = np.array([1, 1, 1, 1])          # hits 0,1 miss onto class-2 cand 1 (different class); hit 2 onto class-1
    hit_pcls = np.array([2, 1, 1, 1])         # trunk thinks hit 0 is class 2 (same as candidate): trunk-confused
    cand_pcls = np.array([1, 2, 1])
    _, st, _ = rd.analyse_event(W, nb, own, pop, owner, cand_pid, cand_cls, hit_cls, ari_fn=lambda v: 0.0,
                                hit_pcls=hit_pcls, cand_pcls=cand_pcls)
    assert st["err_diff_class"] == 2 and st["err_same_class"] == 1
    assert st["err_diff_pred_same"] == 1 and st["err_diff_pred_diff"] == 1


def test_addressability_keys_absent_by_default():
    W = np.array([[0.7, 0.3]])
    _, st, _ = rd.analyse_event(W, np.array([[0, 1]]), np.array([[False, True]]), np.array([True]),
                                np.array([1]), np.array([5, 6]), np.array([0, 1]), np.array([1]),
                                ari_fn=lambda v: 0.0)
    assert "err_diff_pred_same" not in st


# ---- report (pure python) --------------------------------------------------------------------- #
def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def meta(arm, t, **over):
    m = {"m": 20, "seed": t, "hit_dim": 136, "class_channels": pc.MODE[arm], "oracle_upper_bound": arm == "ORACLE",
         "init_sha256": f"i{t}", "perm_sha256": [f"p{t}{e}" for e in range(8)], "best_epoch": 5,
         "best_dev_ari": 0.3, "git_commit": "c", "git_dirty": False, "latent_checkpoint_sha256": "ck", "argv": [arm]}
    m.update(over)
    return m


def rank(diff_class, same=None, diffp=None):
    s = copy.deepcopy(_RANK)
    s["stats"]["err_diff_class"] = int(diff_class * 1000)
    s["stats"]["err_same_class"] = 1000 - 80 - int(diff_class * 1000)
    if same is not None:
        s["stats"]["err_diff_pred_same"], s["stats"]["err_diff_pred_diff"] = same, diffp
    return s


_RANK = {"ari": {"pick_only": 0.25, **{f"head@{k}": 0.40 + 0.01 * i for i, k in enumerate(KS)},
                 **{f"dist@{k}": 0.36 + 0.01 * i for i, k in enumerate(KS)}},
         "stats": {"n_pop": 10000, "n_avail": 4000, "n_correct": 2000, "n_miss": 1000, "owner_missing": 0,
                   "demoted": 100, "recall_head": [2000, 2600, 2900, 3200, 3700, 4000],
                   "recall_dist": [1600, 2300, 2700, 3000, 3500, 4000], "miss_rank_head": [300, 150, 200, 230, 120],
                   "miss_rank_dist": [140, 300, 150, 200, 110, 100], "p1_ok": [0] * 150 + [2000] + [0] * 49,
                   "p1_miss": [0] * 60 + [1000] + [0] * 139, "margin_ok_low": 900, "margin_miss_low": 700,
                   "err_spurious": 80, "err_same_class": 400, "err_diff_class": 520},
         "n_events": N_EV, "gates": {"R1_pick_only_fail": [], "R3_head_eq_dist_at_M_fail": [], "R4_owner_missing": 0}}


def build(tmp, g_oracle=0.17, g_pred=0.10, meta_over=None, readout_pass=True, r5_gain=0.17, hard_shift=0.0):
    root, r5 = tmp / "root", tmp / "r5"
    rng = random.Random(5)
    base = [(0.25 + 0.1 * rng.random(), 0.15 + 0.05 * rng.random()) for _ in range(N_EV)]
    gain = {"CTRL": 0.0, "ORACLE": g_oracle, "PRED": g_pred}

    def diag_rows(g, shift_arm=False):
        rows = []
        for e, (a0, h) in enumerate(base):
            for var in ("v2_nonull", "pick_only"):
                rows.append({"batch": e // 8, "event": e % 8, "variant": var,
                             "ari_particle": a0 + g + rng.gauss(0, 0.01),
                             "ari_hardseed": h + (hard_shift if shift_arm and e == 3 else 0.0)})
        return rows

    F = ["batch", "event", "variant", "ari_particle", "ari_hardseed"]
    for split in ("train", "devA"):
        p = root / "readout" / f"{split}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"pass": readout_pass, "fg_reclassified_diffuse": 0.0002 if readout_pass else 0.01,
                                 "pop_accuracy": 0.8, "checkpoint_sha256": "ck"}))
    for t in pc.SEEDS:
        for arm in pc.ARMS:
            d = root / "heads" / pc.name(arm, t)
            d.mkdir(parents=True)
            (d / "train_meta.json").write_text(json.dumps(meta(arm, t, **((meta_over or {}).get((arm, t), {})))))
            for sub in ("diag_last", "diag_best"):
                write_csv(root / sub / pc.name(arm, t) / "diagnose_events.csv", F, diag_rows(gain[arm], arm == "PRED"))
            r = root / "rank" / pc.name(arm, t)
            r.mkdir(parents=True)
            dc = {"CTRL": 0.53, "ORACLE": 0.02, "PRED": 0.40}[arm]
            (r / "rank_summary.json").write_text(json.dumps(rank(dc, 300, 230) if arm == "CTRL" else rank(dc)))
        for arm, g in (("CTRL", 0.0), ("ORACLE", r5_gain)):
            write_csv(r5 / "diag_last" / pc.name(arm, t) / "diagnose_events.csv", F, diag_rows(g))
    return root, r5


def test_planted_deployable(tmp_path):
    res = pc.evaluate(*build(tmp_path, g_oracle=0.17, g_pred=0.10))
    assert res["gates"]["pass"], res["gates"]["failures"]
    for t in pc.SEEDS:
        F = res["seeds"][t]["F"]
        assert F["F"] == pytest.approx(0.10 / 0.17, abs=0.03) and F["ci95"][0] < F["F"] < F["ci95"][1]
        assert res["seeds"][t]["addressability"]["trunk_confused"] == pytest.approx(300 / 530)
    assert res["verdict"]["reading"] == "DEPLOYABLE"
    assert "Recovered fraction" in pc.render(res)


def test_planted_trunk_limited(tmp_path):
    v = pc.evaluate(*build(tmp_path, g_pred=0.01))["verdict"]
    assert v["reading"] == "TRUNK-LIMITED" and "trunk-level semantic" in v["next"]


def test_planted_partial_with_adoption(tmp_path):
    v = pc.evaluate(*build(tmp_path, g_pred=0.05))["verdict"]
    assert v["reading"] == "PARTIAL" and v["adopt_pred_channels"] and v["next"].startswith("adopt")


@pytest.mark.parametrize("kw, gate", [
    ({"meta_over": {("PRED", 1): {"init_sha256": "x"}}}, "P1 seed 1: PRED"),
    ({"meta_over": {("PRED", 2): {"perm_sha256": ["x"] * 8}}}, "P2 seed 2: PRED"),
    ({"meta_over": {("ORACLE", 1): {"lr": 0.01}}}, "P3 seed 1: ORACLE recipe differs"),
    ({"meta_over": {("PRED", 2): {"latent_checkpoint_sha256": "other"}}}, "P0 PRED_s2"),
    ({"readout_pass": False}, "P0 train"),
    ({"r5_gain": 0.10}, "P5 seed 1"),
    ({"hard_shift": 1e-6}, "P4 seed 1: PRED hard-seed"),
])
def test_each_gate_fails_on_its_planted_defect(tmp_path, kw, gate):
    res = pc.evaluate(*build(tmp_path, **kw))
    assert not res["gates"]["pass"] and res["verdict"]["reading"] == "INVALID"
    assert any(f.startswith(gate) for f in res["gates"]["failures"]), res["gates"]["failures"]


# ---- pred channels and the read-out (torch, EAF) ------------------------------------------ #
def _torch_sp():
    torch = pytest.importorskip("torch")
    return torch, _load("soft_particle_pc", ROOT / "scripts/particle_hierarchy/soft_particle.py")


def fake_cache(tmp_path, torch, sha_ok=True):
    """A cache whose manifest points at a fake checkpoint holding only the semantic head."""
    torch.manual_seed(0)
    W, b = torch.randn(7, 3), torch.randn(7)
    ck = tmp_path / "trunk.ckpt"
    torch.save({"state_dict": {"semantic_decoder.net.weight": W, "semantic_decoder.net.bias": b}}, ck)
    cache = tmp_path / "cache"
    cache.mkdir()
    sp = _torch_sp()[1]
    (cache / "manifest.json").write_text(json.dumps({"checkpoint": str(ck),
                                                     "checkpoint_sha256": sp.mb.sha256(ck) if sha_ok else "bad"}))
    hx = torch.randn(30, 3).half()
    logits = hx.float() @ W.T + b
    mask = logits.argmax(1) != sp.DIFFUSE
    ev = {"batch": 0, "event": 0, "mask": mask, "hit_x_fg": hx[mask], "y_sem": logits.argmax(1).to(torch.int8),
          "y_i": torch.zeros(30, dtype=torch.int32)}
    torch.save([ev], cache / "shard_0000.pt")
    return cache, W, b, hx, mask


def test_pred_channels_are_the_softmax_of_the_linear_readout(tmp_path):
    torch, sp = _torch_sp()
    cache, W, b, hx, mask = fake_cache(tmp_path, torch)
    sp._READOUT.clear()
    ro = sp.semantic_readout(cache)
    d = {"fg_idx": mask.nonzero().squeeze(1), "hit_x": hx[mask].float(), "y_sem": torch.zeros(30, dtype=torch.int8)}
    ch = sp.class_channels(d, "pred", ro)
    want = (hx[mask].float() @ W.T + b).softmax(1)
    assert ch.shape == (int(mask.sum()), sp.CLASS_BUCKETS)
    assert torch.allclose(ch[:, :7], want, atol=1e-6) and not ch[:, 7].any()
    with pytest.raises(ValueError):
        sp.class_channels(d, "pred")                      # the read-out is required


def test_readout_refuses_a_checkpoint_that_does_not_match_the_cache(tmp_path):
    torch, sp = _torch_sp()
    cache, *_ = fake_cache(tmp_path, torch, sha_ok=False)
    sp._READOUT.clear()
    with pytest.raises(SystemExit, match="sha256"):
        sp.semantic_readout(cache)


def test_readout_check_reproduces_the_foreground_mask(tmp_path, monkeypatch):
    torch, sp = _torch_sp()
    cache, *_ = fake_cache(tmp_path, torch)
    sp._READOUT.clear()
    out = tmp_path / "ro.json"
    monkeypatch.setattr(sys, "argv", ["soft_particle.py", "readout-check", "--cache", str(cache), "--out", str(out)])
    sp.main()
    res = json.loads(out.read_text())
    assert res["pass"] and res["fg_reclassified_diffuse"] == 0.0


def test_pred_head_matches_ctrl_init_and_order(tmp_path, monkeypatch):
    """End to end: a PRED head shares init and training order with CTRL (gates P1/P2)."""
    torch, sp = _torch_sp()
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    tcp = _load("test_class_probe_x", ROOT / "tests/test_class_probe.py")
    cache = tcp.synthetic_cache(tmp_path)
    W = torch.randn(7, 3)
    ck = tmp_path / "trunk.ckpt"
    torch.save({"state_dict": {"semantic_decoder.net.weight": W, "semantic_decoder.net.bias": torch.zeros(7)}}, ck)
    (cache / "manifest.json").write_text(json.dumps({"checkpoint": str(ck), "checkpoint_sha256": sp.mb.sha256(ck)}))
    metas = {}
    for mode in ("zeros", "pred"):
        sp._READOUT.clear()
        _, metas[mode] = tcp.train(tmp_path, cache, mode, monkeypatch)
    assert metas["zeros"]["init_sha256"] == metas["pred"]["init_sha256"]
    assert metas["zeros"]["perm_sha256"] == metas["pred"]["perm_sha256"]
    assert metas["pred"]["oracle_upper_bound"] is False
