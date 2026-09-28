"""Tests for Stage 1a-R10 (co-membership objective): pair statistics, the report driver, pair sampling and the loss.

The pair statistics and report are numpy / pure python; sampling, loss and end-to-end tests need torch (EAF).
Run:  pytest -q tests/test_comem.py
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


cm = _load("comem", ROOT / "scripts/particle_hierarchy/comem.py")
rd = cm.rd
N_EV = 200


# ---- pair statistics (numpy) ------------------------------------------------------------------ #
def test_pair_stats():
    part = np.array([1, 1, 1, 2, 2])
    lab = np.array([7, 7, 8, 7, 9])
    # pairs 10; same particle 3 + 1 = 4; same label: {0,1,3} -> 3; same both: (0,1) -> 1
    assert rd.pair_stats(part, lab) == (10, 4, 3, 1)
    assert rd.pair_stats(np.array([1]), np.array([0])) == (0, 0, 0, 0)


def test_three_way_coassignment_on_a_hand_built_event():
    """Particle 10: h0, h1 reachable (own c0), h2 unreachable. Particle 30: h3, h4 unreachable."""
    cand_pid = np.array([10, 20])
    hit_pid = np.array([10, 10, 10, 30, 30])
    nb = np.array([[0, 1], [0, 1], [1, 1], [1, 0], [0, 1]])
    nb[2] = [1, 1]                                    # h2 sees only c1 (particle 20) -> unreachable
    own = cand_pid[nb] == hit_pid[:, None]
    W = np.tile([0.7, 0.3], (5, 1))                   # picks: c0 c0 c1 c1 c0
    dec = np.array([5, 5, 5, 6, 6])                   # decoded clusters: particle 10 together, 30 together
    _, st, _ = rd.analyse_event(W, nb, own, np.ones(5, bool), np.array([0, 0, -1, -1, -1]), cand_pid,
                                np.array([0, 0]), np.zeros(5, int), ari_fn=lambda v: 0.0, hit_pid=hit_pid,
                                dec_labels=dec)
    c = {lvl: cm.categories(st, lvl) for lvl in ("pick", "dec")}
    # reachable-reachable: (h0,h1) same particle, same pick
    assert c["pick"]["RR"] == {"pairs": 1, "same_part": 1, "same_lab": 1, "same_both": 1}
    # unreachable-unreachable {h2,h3,h4}: same particle (h3,h4); picks c1 c1 c0 -> same label (h2,h3), wrong
    assert c["pick"]["UU"] == {"pairs": 3, "same_part": 1, "same_lab": 1, "same_both": 0}
    # reachable-unreachable: (h0,h2) (h1,h2) same particle; (h0,h4) (h1,h4) same pick c0 but different particle
    assert c["pick"]["RU"] == {"pairs": 6, "same_part": 2, "same_lab": 2, "same_both": 0}
    assert cm.rates(c["pick"]["RU"]) == {"coassign": 0.0, "false": 0.5}
    assert c["dec"]["RU"]["same_both"] == 2 and c["dec"]["UU"]["same_both"] == 1
    assert cm.rates(c["dec"]["UU"]) == {"coassign": 1.0, "false": 0.0}


# ---- report (pure python) --------------------------------------------------------------------- #
def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def meta(arm, t, **over):
    m = {"m": 20, "seed": t, "hit_dim": 128, "class_channels": "none", "oracle_upper_bound": False,
         "comem": cm.LAMBDA[arm], "comem_pairs": 2048, "init_sha256": f"i{t}",
         "perm_sha256": [f"p{t}{e}" for e in range(8)], "best_epoch": 5, "best_dev_ari": 0.3, "git_commit": "c",
         "git_dirty": False, "argv": [arm]}
    m.update(over)
    return m


def co_stats(rr=0.60, ru=0.40, uu=0.55, false=0.05):
    """Rank summary co_* stats for given decode-level co-assignment rates (pick level mirrors decode)."""
    base = {"avail": (1000, 400, rr), "unavail": (1200, 500, uu)}
    st = {}
    tot_pairs, tot_part, tot_lab, tot_both = 0, 0, 0, 0
    for tag, (pairs, part, rate) in base.items():
        both = int(rate * part)
        lab = both + int(false * (pairs - part))
        for lvl in ("pick", "dec"):
            st.update({f"co_{lvl}_{tag}_pairs": pairs, f"co_{lvl}_{tag}_same_part": part,
                       f"co_{lvl}_{tag}_same_lab": lab, f"co_{lvl}_{tag}_same_both": both})
        tot_pairs += pairs; tot_part += part; tot_lab += lab; tot_both += both
    ru_pairs, ru_part = 2000, 100
    ru_both = int(ru * ru_part)
    ru_lab = ru_both + int(false * (ru_pairs - ru_part))
    for lvl in ("pick", "dec"):
        st.update({f"co_{lvl}_pop_pairs": tot_pairs + ru_pairs, f"co_{lvl}_pop_same_part": tot_part + ru_part,
                   f"co_{lvl}_pop_same_lab": tot_lab + ru_lab, f"co_{lvl}_pop_same_both": tot_both + ru_both})
    return st


def rank(co):
    st = {"n_pop": 10000, "n_avail": 4000, "n_correct": 2000, "n_miss": 1000, "owner_missing": 0, "demoted": 100,
          "recall_head": [2000, 2600, 2900, 3200, 3700, 4000], "recall_dist": [1600, 2300, 2700, 3000, 3500, 4000],
          "miss_rank_head": [300, 150, 200, 230, 120], "miss_rank_dist": [140, 300, 150, 200, 110, 100],
          "p1_ok": [0] * 150 + [2000] + [0] * 49, "p1_miss": [0] * 60 + [1000] + [0] * 139, "margin_ok_low": 900,
          "margin_miss_low": 700, "err_spurious": 80, "err_same_class": 400, "err_diff_class": 520, **co}
    return {"ari": {"pick_only": 0.25, **{f"head@{k}": 0.40 + 0.01 * i for i, k in enumerate(rd.KS)},
                    **{f"dist@{k}": 0.36 + 0.01 * i for i, k in enumerate(rd.KS)}},
            "stats": st, "n_events": N_EV,
            "gates": {"R1_pick_only_fail": [], "R3_head_eq_dist_at_M_fail": [], "R4_owner_missing": 0}}


def build(tmp, gains=None, co=None, comem_loss=None, meta_over=None, hard_shift=0.0, ctrl_level=0.25):
    """gains[arm] = per-seed tuple of planted ARI gains; co[arm] = co_stats kwargs; comem_loss[arm] = last-epoch loss."""
    root = tmp / "root"
    gains = {"CTRL": (0, 0, 0), "CM03": (0.03, 0.03, 0.03), "CM1": (0.015, 0.015, 0.015), **(gains or {})}
    co = {"CTRL": {}, "CM03": {"uu": 0.70, "ru": 0.55}, "CM1": {"uu": 0.60}, **(co or {})}
    closs = {"CTRL": 2.0, "CM03": 1.5, "CM1": 1.2, **(comem_loss or {})}
    rng = random.Random(5)
    base = [(ctrl_level + 0.1 * rng.random() - 0.05, 0.15 + 0.05 * rng.random()) for _ in range(N_EV)]
    F = ["batch", "event", "variant", "ari_particle", "ari_hardseed"]
    for t in cm.SEEDS:
        for arm in cm.ARMS:
            d = root / "heads" / cm.name(arm, t)
            d.mkdir(parents=True)
            (d / "train_meta.json").write_text(json.dumps(meta(arm, t, **((meta_over or {}).get((arm, t), {})))))
            write_csv(d / "train_history.csv", ["epoch", "loss", "comem"],
                      [{"epoch": e, "loss": 1.0, "comem": closs[arm] + (7 - e) * 0.1} for e in range(8)])
            g = gains[arm][t - 1]
            for sub in ("diag_last", "diag_best"):
                rows = [{"batch": e // 8, "event": e % 8, "variant": var, "ari_particle": a0 + g + rng.gauss(0, 0.01),
                         "ari_hardseed": h + (hard_shift if arm == "CM1" and e == 3 else 0.0)}
                        for e, (a0, h) in enumerate(base) for var in ("v2_nonull", "pick_only")]
                write_csv(root / sub / cm.name(arm, t) / "diagnose_events.csv", F, rows)
            rk = root / "rank" / cm.name(arm, t)
            rk.mkdir(parents=True)
            (rk / "rank_summary.json").write_text(json.dumps(rank(co_stats(**co[arm]))))
    return root


def test_objective_gap_with_the_gain_in_the_ignored_population(tmp_path):
    res = cm.evaluate(build(tmp_path))
    assert res["gates"]["pass"], res["gates"]["failures"]
    g = res["arms"]["CM03"]
    assert all(x == pytest.approx(0.03, abs=0.005) for x in g["delta"]) and g["manipulated"]
    # decode gain: RU 15 pairs + UU 75 pairs per seed, RR 0 -> unreachable share 1
    assert g["coassign_gain_pairs"]["RR"] == 0 and g["unreachable_share"] == pytest.approx(1.0)
    assert g["reading"] == "OBJECTIVE-GAP" and res["arms"]["CM1"]["reading"] == "PARTIAL"
    assert res["verdict"]["reading"] == "OBJECTIVE-GAP" and res["verdict"]["carried_forward"] == "CM03"
    assert "T + co-membership" in res["verdict"]["next"]


def test_gain_from_reachable_pairs_only_is_flagged(tmp_path):
    res = cm.evaluate(build(tmp_path, co={"CM03": {"rr": 0.80}}))
    assert res["arms"]["CM03"]["reading"] == "OBJECTIVE-GAP (other route)"
    assert res["arms"]["CM03"]["unreachable_share"] == pytest.approx(0.0)
    assert res["verdict"]["reading"] == "PARTIAL"


def test_information_gap_needs_both_lambdas_null(tmp_path):
    res = cm.evaluate(build(tmp_path, gains={"CM03": (0.0, 0.005, 0.0), "CM1": (0.0, 0.0, 0.004)}))
    assert [res["arms"][a]["reading"] for a in ("CM03", "CM1")] == ["NULL", "NULL"]
    assert res["verdict"]["reading"] == "INFORMATION-GAP" and "trunk" in res["verdict"]["next"]
    res = cm.evaluate(build(tmp_path / "b", gains={"CM03": (0.0, 0.0, 0.0), "CM1": (0.02, 0.01, 0.01)}))
    assert res["verdict"]["reading"] == "PARTIAL"


def test_a_seed_below_the_bar_is_not_an_objective_gap(tmp_path):
    res = cm.evaluate(build(tmp_path, gains={"CM03": (0.03, 0.03, 0.012)}))
    assert res["arms"]["CM03"]["reading"] == "PARTIAL"


def test_manipulation(tmp_path):
    res = cm.evaluate(build(tmp_path, comem_loss={"CM03": 1.9, "CM1": 1.95}))
    assert [res["arms"][a]["reading"] for a in ("CM03", "CM1")] == ["MANIPULATION-FAILED"] * 2
    assert res["verdict"]["reading"] == "MANIPULATION-FAILED"


@pytest.mark.parametrize("kw, gate", [
    ({"meta_over": {("CM03", 2): {"init_sha256": "x"}}}, "E1 seed 2: CM03"),
    ({"meta_over": {("CM1", 3): {"perm_sha256": ["x"] * 8}}}, "E2 seed 3: CM1"),
    ({"meta_over": {("CM1", 1): {"lr": 0.01}}}, "E3 seed 1: CM1 recipe differs"),
    ({"meta_over": {("CM03", 1): {"comem": 0.5}}}, "E3 CM03_s1"),
    ({"meta_over": {("CTRL", 3): {"git_dirty": True}}}, "E3 CTRL_s3"),
    ({"meta_over": {("CM1", 2): {"class_channels": "zeros"}}}, "E3 CM1_s2"),
    ({"hard_shift": 1e-6}, "E4 seed 1: CM1 hard-seed"),
    ({"ctrl_level": 0.33}, "E5 seed 1"),
])
def test_each_gate_fails_on_its_planted_defect(tmp_path, kw, gate):
    res = cm.evaluate(build(tmp_path, **kw))
    assert not res["gates"]["pass"] and res["verdict"]["reading"] == "INVALID"
    assert any(f.startswith(gate) for f in res["gates"]["failures"]), res["gates"]["failures"]


def test_old_rank_summaries_are_refused(tmp_path):
    root = build(tmp_path)
    p = root / "rank" / "CM1_s2" / "rank_summary.json"
    s = json.loads(p.read_text())
    s["stats"] = {k: v for k, v in s["stats"].items() if not k.startswith("co_")}
    p.write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="R10 co_"):
        cm.evaluate(root)


def test_render(tmp_path):
    md = cm.render(cm.evaluate(build(tmp_path)))
    assert "**Reading:** OBJECTIVE-GAP (carried forward: CM03)" in md
    hdr = next(l for l in md.splitlines() if l.startswith("| ARI (last epoch)"))
    assert [x.strip() for x in hdr.split("|")[2:-1]] == list(cm.ARMS)
    assert any(l.startswith("| 1 | dec | UU |") for l in md.splitlines())


# ---- sampling and loss (torch, EAF) ---------------------------------------------------------- #
def _torch_sp():
    torch = pytest.importorskip("torch")
    return torch, _load("soft_particle_cm", ROOT / "scripts/particle_hierarchy/soft_particle.py")


def pair_event(torch, pids=(1, 1, 1, 2, 2, 3, -1, 4)):
    n = len(pids)
    y_i = torch.tensor(list(pids))
    return {"fg_idx": torch.arange(n), "y_i": y_i, "pop_fg": (y_i >= 0) & (torch.arange(n) != 7),
            "nb": torch.arange(n * 3).reshape(n, 3) % 4, "m": 3}


def test_pairs_are_population_hits_with_the_right_relation():
    torch, sp = _torch_sp()
    d = pair_event(torch)
    g = torch.Generator().manual_seed(0)
    state = torch.get_rng_state()
    i, j, same = sp.comem_sample(d, 500, g)
    assert torch.equal(state, torch.get_rng_state())                      # global RNG untouched
    pid = d["y_i"]
    assert d["pop_fg"][i].all() and d["pop_fg"][j].all()
    assert (i != j).all()
    assert torch.equal(pid[i] == pid[j], same)
    assert int(same.sum()) == 500 and int((~same).sum()) == 500
    # partners cover every other hit of the particle (particle 1 = hits 0, 1, 2)
    pos = same & (i == 0)
    assert set(j[pos].tolist()) == {1, 2}
    # hit 5 (particle 3, a singleton) and hit 7 (not population) are never anchors of positives
    assert not ((i[same] == 5) | (i[same] == 7)).any()
    i2, j2, same2 = sp.comem_sample(d, 500, torch.Generator().manual_seed(0))
    assert torch.equal(i, i2) and torch.equal(j, j2)                       # deterministic given the generator


def test_degenerate_events():
    torch, sp = _torch_sp()
    i, j, same = sp.comem_sample(pair_event(torch, (1, 1, 1)), 10, torch.Generator().manual_seed(0))
    assert same.all() and i.numel() == 10                                  # one particle: no negatives
    i, j, same = sp.comem_sample(pair_event(torch, (1, 2)), 10, torch.Generator().manual_seed(0))
    assert not same.any()                                                  # singletons: no positives
    i, j, same = sp.comem_sample(pair_event(torch, (1,)), 10, torch.Generator().manual_seed(0))
    assert i.numel() == 0
    lg = torch.zeros(1, 4, requires_grad=True)
    assert float(sp.comem_loss(lg, {"nb": torch.zeros(1, 3, dtype=torch.long)}, i, j, same)) == 0.0


def test_overlap_counts_shared_candidates_only():
    torch, sp = _torch_sp()
    nb = torch.tensor([[5, 6, 7], [7, 8, 5]])
    lg = torch.log(torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.3, 0.3, 0.3, 0.1]]))     # last column = null
    s = sp.comem_overlap(lg, nb, torch.tensor([0]), torch.tensor([1]))
    # shared: 5 (0.5 * 0.3) and 7 (0.2 * 0.3)
    assert float(s) == pytest.approx(0.5 * 0.3 + 0.2 * 0.3, abs=1e-6)


def test_the_loss_pulls_positives_together_and_pushes_negatives_apart():
    torch, sp = _torch_sp()
    nb = torch.tensor([[0, 1], [0, 1], [0, 1]])
    lg = torch.zeros(3, 3, requires_grad=True)
    i, j, same = torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([True, False])
    s0 = sp.comem_overlap(lg, nb, i, j).detach()
    opt = torch.optim.SGD([lg], lr=1.0)
    for _ in range(50):
        opt.zero_grad()
        sp.comem_loss(lg, {"nb": nb}, i, j, same).backward()
        opt.step()
    s1 = sp.comem_overlap(lg, nb, i, j).detach()
    assert s1[0] > s0[0] and s1[1] < s0[1]


def test_lambda_zero_is_the_unmodified_loss_but_logs_the_term(tmp_path):
    torch, sp = _torch_sp()
    tcp = _load("test_class_probe_cm", ROOT / "tests/test_class_probe.py")
    pytest.importorskip("torch_scatter")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    cache = tcp.synthetic_cache(tmp_path)
    mat, dev = sp._setup(type("A", (), {"device": "cpu"})())
    ev = next(sp.mb.iter_cache(cache, 0, 1))
    d = sp.prepare(ev, mat, sp.s0.make_rule("fixed1:0.5"), 20, 0.3, float("inf"), "cpu")
    torch.manual_seed(0)
    model = sp.build_model(d["hit_x"].size(1), d["ox"].size(1), 16, 2)
    kw = dict(target="own", exist_target="real", merge_scope="local")
    base, p0 = sp.event_loss(model, d, **kw)
    ctrl, p1 = sp.event_loss(model, d, **kw, comem=0.0, comem_gen=torch.Generator().manual_seed(1))
    arm, p2 = sp.event_loss(model, d, **kw, comem=1.0, comem_gen=torch.Generator().manual_seed(1))
    assert float(ctrl) == float(base) and "comem" not in p0 and p1["comem"] == p2["comem"] > 0
    assert float(arm) == pytest.approx(float(base) + p2["comem"], rel=1e-6)


def test_three_arms_share_init_and_order_and_ranking_records_the_pair_classes(tmp_path, monkeypatch):
    torch, sp = _torch_sp()
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    tcp = _load("test_class_probe_cm2", ROOT / "tests/test_class_probe.py")
    cache = tcp.synthetic_cache(tmp_path)
    metas = {}
    for arm, lam in cm.LAMBDA.items():
        out = tmp_path / f"head_{arm}"
        monkeypatch.setattr(sys, "argv", ["soft_particle.py", "train", "--train-cache", str(cache),
                                          "--dev-cache", str(cache), "--dev-batches", "0:1", "--epochs", "2",
                                          "--m", "20", "--rule", "fixed1:0.5", "--no-null", "--seed", "1",
                                          "--device", "cpu", "--comem", str(lam), "--out", str(out)])
        sp.main()
        metas[arm] = json.loads((out / "train_meta.json").read_text())
        hist = list(csv.DictReader(open(out / "train_history.csv")))
        assert len(hist) == 2 and float(hist[-1]["comem"]) > 0
    for arm in cm.ARMS[1:]:
        assert metas[arm]["init_sha256"] == metas["CTRL"]["init_sha256"]
        assert metas[arm]["perm_sha256"] == metas["CTRL"]["perm_sha256"]
        diff = {k for k in set(metas[arm]) | set(metas["CTRL"]) if metas[arm].get(k) != metas["CTRL"].get(k)}
        assert diff <= cm.cp.OUTCOME_KEYS | {"argv", "out", "comem"}, diff
    rdm = _load("ranking_diag_cm", ROOT / "scripts/particle_hierarchy/ranking_diag.py")
    out = tmp_path / "rank"
    monkeypatch.setattr(sys, "argv", ["ranking_diag.py", "run", "--head", str(tmp_path / "head_CM1" / "head.pt"),
                                      "--cache", str(cache), "--batches", "0:1", "--out", str(out), "--device", "cpu"])
    rdm.main()
    st = json.loads((out / "rank_summary.json").read_text())["stats"]
    for lvl in ("pick", "dec"):
        c = cm.categories(st, lvl)
        assert all(c[k]["same_part"] >= c[k]["same_both"] >= 0 for k in cm.CATS)
        assert sum(c[k]["same_part"] for k in cm.CATS) > 0
