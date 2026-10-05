"""Torch-side tests for coverage_anatomy.py (replica parity, k-NN, ARI identity, end-to-end run + report).

CPU only.  Run:  pytest -q tests/test_coverage_anatomy_torch.py
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
mat = pytest.importorskip("nugraph.models.nugraph3.materialize")

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load("coverage_anatomy", ROOT / "scripts/particle_hierarchy/coverage_anatomy.py")
s0 = _load("stage0_seed_diag", ROOT / "scripts/particle_hierarchy/stage0_seed_diag.py")
DEV = torch.device("cpu")


def make_event(k=6, n=60, sep=0.6, std=0.25, n_bkg=20, seed=0, batch=0, event=0):
    """Overlapping blobs (sep comparable to t_d) so NMS suppresses across particles; many seeds above tau."""
    g = torch.Generator().manual_seed(seed)
    cen = torch.zeros(k, 8)
    cen[:, 0] = torch.arange(k, dtype=torch.float) * sep
    y = torch.arange(k).repeat_interleave(n)
    ox = cen[y] + std * torch.randn(k * n, 8, generator=g)
    beta = torch.rand(k * n, generator=g) ** 3
    ox = torch.cat([ox, 3 + torch.randn(n_bkg, 8, generator=g)])
    beta = torch.cat([beta, torch.rand(n_bkg, generator=g)])
    y = torch.cat([y, torch.full((n_bkg,), -1)])
    n_tot = y.numel()
    ysem = torch.zeros(n_tot, dtype=torch.int8)
    return {"batch": batch, "event": event, "ox": ox, "beta": beta, "mask": torch.ones(n_tot, dtype=torch.bool),
            "y_i": y.to(torch.int32), "y_sem": ysem}


@pytest.mark.parametrize("seed", range(6))
def test_within_matrix_reproduces_deployed_survivors(seed):
    ev = make_event(seed=seed, k=8, n=80)
    ox, bf = ev["ox"], ev["beta"]
    tau = s0.make_rule(ca.FROZEN_RULE)(bf.double())
    for t_d in (0.3, 0.15):
        cp, _, _ = s0.seeds_and_labels(mat, ox, bf, tau, t_d, float("inf"), DEV)
        seed_idx, W, _ = ca.within_matrix(mat, ox, bf, tau, t_d, DEV)
        assert seed_idx.numel() > 25                                   # exercises the mm cdist path
        dom, _ = ca.deployed_from_within(W.numpy(), bf.numpy()[seed_idx.numpy()])
        assert set(seed_idx.numpy()[~dom].tolist()) == set(cp.numpy().tolist())


def test_knn_matches_brute_force_topk():
    g = torch.Generator().manual_seed(3)
    Q, C = torch.randn(500, 8, generator=g), torch.randn(40, 8, generator=g)
    got = ca.knn(mat, Q, C, 20)
    ref = torch.topk(torch.cdist(Q, C, compute_mode="donot_use_mm_for_euclid_dist"), 20, dim=1, largest=False).indices
    d = torch.cdist(Q, C)
    assert torch.allclose(torch.gather(d, 1, got), torch.gather(d, 1, ref), atol=1e-5)
    assert ca.knn(mat, Q, C[:3], 20).shape == (500, 3)


def test_ari_identical_to_mat_bench_event_quality():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 5, 300)
    pred = rng.integers(-1, 7, 300)
    q = s0.mb.event_quality(torch.as_tensor(pred), torch.as_tensor(true), torch.zeros(300, dtype=torch.long),
                            torch.ones(300, dtype=torch.bool))
    assert ca.ari(pred, true) == q["ari_particle"]


def test_analyse_event_bookkeeping_and_parity():
    a = SimpleNamespace(t_d=0.3)
    for seed in range(4):
        ev = make_event(seed=seed)
        assert ca.check_replica(mat, ev, s0, DEV, a)
        r = ca.analyse_event(ev, mat, s0, DEV, a, max_attr_seeds=8000)
        row = r["row"]
        assert row["a2_ok"] == 1 and row["attributable"] == 1
        n = row["n_pop"]
        assert sum(row[f"cat.{c}"] for c in ca.CATS) == n
        assert row["FROZEN.n_seedless"] + row["FROZEN.n_oor"] + row["FROZEN.n_reach"] == n
        assert sum(row[f"cat.seedless_{c}"] for c in ca.CAUSES) == row["FROZEN.n_seedless"]
        assert row["ORACLE-SEEDS.n_seedless"] == 0 and row["ORACLE-ADD.n_seedless"] == 0
        assert row["GREEDY.K"] >= row["FROZEN.K"] and row["NONMS.K"] >= row["GREEDY.K"]
        assert row["ORACLE-ADD.K"] == row["FROZEN.K"] + len(r["particles"])
        assert row["cat.seedless_unattributed"] == 0


def test_run_and_report_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    evs = [make_event(seed=s, batch=s // 2, event=s % 2) for s in range(6)]
    torch.save(evs, cache / "shard_0000.pt")
    (cache / "manifest.json").write_text(json.dumps({"split": "validation", "batches": "0:3"}))
    for name in ("T_s1", "T_s2"):
        ca.cmd_run(SimpleNamespace(cache=str(cache), out=str(tmp_path / name), batches="0:3", deva_end=150,
                                   device="cpu", t_d=0.3, check_events=6, max_attr_seeds=8000))
    meta = json.loads((tmp_path / "T_s1" / "meta.json").read_text())
    assert meta["n_events"] == 6 and meta["a1_failed"] == 0 and meta["a1_checked"] == 6
    ca.cmd_report(SimpleNamespace(primary=[str(tmp_path / "T_s1"), str(tmp_path / "T_s2")], descriptive=[],
                                  anchor=None, out=str(tmp_path / "rep")))
    rep = json.loads((tmp_path / "rep" / "report.json").read_text())
    assert rep["gates_pass"]
    assert rep["reading"] in ("SUPPRESSION", "THRESHOLD", "REACH", "MIXED")


def test_run_refuses_beyond_dev_a(tmp_path):
    with pytest.raises(SystemExit):
        ca.cmd_run(SimpleNamespace(cache=str(tmp_path), out=str(tmp_path / "o"), batches="0:", deva_end=150,
                                   device="cpu", t_d=0.3, check_events=0, max_attr_seeds=8000))
