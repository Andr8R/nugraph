"""Tests for the Stage 1a-R5 class-information probe (ORACLE UPPER BOUND) and its matched control.

The point of these tests is the control: CTRL ('zeros') and ORACLE must share architecture, initialisation
and training order, so that the only difference is the content of the class channels.
Needs torch (+ the NuGraph materializer and sklearn for the end-to-end tests): runs on EAF.
Run:  pytest -q tests/test_class_probe.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load("soft_particle_cp", ROOT / "scripts/particle_hierarchy/soft_particle.py")


def fake_event(n_hit=10):
    fg = torch.tensor([True] * (n_hit - 2) + [False, False])
    return {"fg_idx": fg.nonzero().squeeze(1), "hit_x": torch.randn(n_hit - 2, 3),
            "y_sem": torch.tensor([0, 1, 2, 3, 4, 5, 6, -1, 2, 2], dtype=torch.int8)[:n_hit]}


# ---- channel content ---------------------------------------------------------------------- #
def test_zero_channels_carry_nothing():
    d = fake_event()
    z = sp.class_channels(d, "zeros")
    assert z.shape == (8, sp.CLASS_BUCKETS) and z.dtype == d["hit_x"].dtype and not z.any()


def test_oracle_channels_are_the_true_class_with_an_unlabelled_bucket():
    d = fake_event()
    o = sp.class_channels(d, "oracle")
    assert o.shape == (8, sp.CLASS_BUCKETS) and (o.sum(1) == 1).all()
    assert o.argmax(1).tolist() == [0, 1, 2, 3, 4, 5, 6, 7]         # y_sem -1 -> bucket 7


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        sp.class_channels(fake_event(), "predicted")


# ---- the confound and why the matched control removes it -------------------------------------- #
def build(hit_dim, seed=1):
    torch.manual_seed(seed)
    m = sp.build_model(hit_dim, 8, 16, 2)
    return m, torch.randperm(50)


def test_same_width_gives_identical_init_and_order():
    (a, pa), (b, pb) = build(3 + 8), build(3 + 8)
    assert sp.state_sha256(a) == sp.state_sha256(b)
    assert torch.equal(pa, pb)


def test_adding_channels_naively_changes_init_everywhere_and_the_order():
    """The confound the matched control exists to remove: a wider input shifts every later draw."""
    (a, pa), (b, pb) = build(3), build(3 + 8)
    assert not torch.equal(pa, pb)                                     # training order changes
    # layers built after the first one are shape-identical, yet initialised differently
    assert a.pair_net[0].weight.shape == b.pair_net[0].weight.shape
    assert not torch.equal(a.pair_net[0].weight, b.pair_net[0].weight)


# ---- end to end: CTRL and ORACLE differ only in channel content (EAF) ------------------------- #
def synthetic_cache(tmp_path):
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
        beta[torch.arange(k) * n + 1] = 0.8
        evs.append({"batch": 0, "event": i, "ox": ox, "beta": beta,
                    "mask": torch.ones(k * n, dtype=torch.bool), "y_i": y.to(torch.int32),
                    "y_sem": (y % 3).to(torch.int8), "x_nugraph": torch.zeros(k * n, dtype=torch.int32),
                    "hit_x_fg": torch.randn(k * n, 3, generator=g).half()})
    torch.save(evs, cache / "shard_00000.pt")
    return cache


def train(tmp_path, cache, mode, monkeypatch):
    out = tmp_path / f"head_{mode}"
    monkeypatch.setattr(sys, "argv", ["soft_particle.py", "train", "--train-cache", str(cache),
                                      "--dev-cache", str(cache), "--dev-batches", "0:1", "--epochs", "2",
                                      "--m", "20", "--rule", "fixed1:0.5", "--no-null", "--seed", "1",
                                      "--device", "cpu", "--class-channels", mode, "--out", str(out)])
    sp.main()
    return out, json.loads((out / "train_meta.json").read_text())


@pytest.fixture
def pair(tmp_path, monkeypatch):
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    cache = synthetic_cache(tmp_path)
    return cache, train(tmp_path, cache, "zeros", monkeypatch), train(tmp_path, cache, "oracle", monkeypatch)


def test_ctrl_and_oracle_are_matched_except_for_the_channels(pair):
    _, (_, ctrl), (_, orac) = pair
    assert ctrl["hit_dim"] == orac["hit_dim"] == 3 + sp.CLASS_BUCKETS
    assert ctrl["init_sha256"] == orac["init_sha256"]                          # gate C1
    assert ctrl["perm_sha256"] == orac["perm_sha256"] and len(ctrl["perm_sha256"]) == 2   # gate C2
    outcome = {"class_channels", "oracle_upper_bound", "best_epoch", "best_dev_ari"}
    diff = {k for k in set(ctrl) | set(orac) if ctrl.get(k) != orac.get(k)} - outcome
    diff -= {"git_commit", "git_dirty", "argv", "materialize_sha256", "out"}  # provenance, not recipe
    assert diff == set(), diff                                                  # gate C3
    assert orac["oracle_upper_bound"] is True and ctrl["oracle_upper_bound"] is False


def test_tools_rebuild_the_channels_from_the_head_config(pair, tmp_path, monkeypatch):
    cache, (out_c, _), (out_o, _) = pair
    for out in (out_c, out_o):                                                  # gate C5
        monkeypatch.setattr(sys, "argv", ["soft_particle.py", "eval", "--head", str(out / "head.pt"),
                                          "--cache", str(cache), "--batches", "0:1", "--device", "cpu",
                                          "--out", str(out / "ev"), "--no-null"])
        sp.main()
        assert json.loads((out / "ev" / "dev_summary.json").read_text())["dev/n_events"] == 3


def test_a_head_rebuilt_without_its_channels_fails_loudly(pair, tmp_path, monkeypatch):
    cache, (out_c, _), _ = pair
    ck = torch.load(out_c / "head.pt", weights_only=False)
    ck["cfg"].pop("class_channels")
    torch.save(ck, tmp_path / "stripped.pt")
    monkeypatch.setattr(sys, "argv", ["soft_particle.py", "eval", "--head", str(tmp_path / "stripped.pt"),
                                      "--cache", str(cache), "--batches", "0:1", "--device", "cpu",
                                      "--out", str(tmp_path / "ev"), "--no-null"])
    with pytest.raises(RuntimeError):                                           # input width mismatch
        sp.main()


def test_time_refuses_probe_heads(pair, tmp_path, monkeypatch):
    cache, (out_c, _), _ = pair
    monkeypatch.setattr(sys, "argv", ["soft_particle.py", "time", "--head", str(out_c / "head.pt"),
                                      "--cache", str(cache), "--batches", "0:1", "--out", str(tmp_path / "t"),
                                      "--device", "cpu"])
    with pytest.raises(SystemExit, match="not timed"):
        sp.main()


def test_ranking_diag_runs_on_a_probe_head(pair, tmp_path, monkeypatch):
    cache, _, (out_o, _) = pair
    rd = _load("ranking_diag_cp", ROOT / "scripts/particle_hierarchy/ranking_diag.py")
    out = tmp_path / "rank"
    monkeypatch.setattr(sys, "argv", ["ranking_diag.py", "run", "--head", str(out_o / "head.pt"), "--cache",
                                      str(cache), "--batches", "0:1", "--out", str(out), "--device", "cpu"])
    rd.main()
    assert json.loads((out / "rank_summary.json").read_text())["n_events"] == 3
