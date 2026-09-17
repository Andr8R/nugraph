"""Unit tests for the Stage-1a soft particle decoder (scripts/particle_hierarchy).

CPU only, no cache and no GPU needed.  Run:  pytest -q tests/test_soft_particle.py
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_scatter")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = _load("soft_particle", ROOT / "scripts/particle_hierarchy/soft_particle.py")
mat = pytest.importorskip("nugraph.models.nugraph3.materialize")
DEV = torch.device("cpu")


def make_event(k=3, n=40, sep=6.0, std=0.2, n_bkg=10, seed=0):
    """Synthetic cached event: k separated blobs, one clear condensation point each."""
    g = torch.Generator().manual_seed(seed)
    cen = torch.zeros(k, 8)
    cen[:, 0] = torch.arange(k, dtype=torch.float) * sep
    y = torch.arange(k).repeat_interleave(n)
    ox = cen[y] + std * torch.randn(k * n, 8, generator=g)
    beta = 0.1 * torch.rand(k * n, generator=g)
    beta[torch.arange(k) * n] = 0.9                       # condensation points
    hit = torch.nn.functional.one_hot(y, k).float()
    if n_bkg:                                             # background hits: y_i = -1
        ox = torch.cat([ox, 20 + torch.randn(n_bkg, 8, generator=g)])
        beta = torch.cat([beta, 0.05 * torch.rand(n_bkg, generator=g)])
        y = torch.cat([y, torch.full((n_bkg,), -1)])
        hit = torch.cat([hit, torch.zeros(n_bkg, k)])
    n_tot = y.numel()
    return {"batch": 0, "event": 0, "ox": ox, "beta": beta,
            "mask": torch.ones(n_tot, dtype=torch.bool), "y_i": y.to(torch.int32),
            "y_sem": torch.zeros(n_tot, dtype=torch.int8),
            "x_nugraph": torch.zeros(n_tot, dtype=torch.int32), "hit_x_fg": hit.half()}


def prep(ev=None, m=3, rule="fixed1:0.5"):
    ev = ev if ev is not None else make_event()
    return sp.prepare(ev, mat, sp.s0.make_rule(rule), m, 0.3, float("inf"), DEV)


def test_prepare_targets_point_at_own_candidate():
    d = prep()
    assert d["K"] == 3 and d["m"] == 3
    pop = d["pop_fg"]
    tgt = d["target"][pop]
    assert (tgt >= 0).all() and (tgt < d["m"]).all()          # no ignored population hits here
    chosen = d["nb"][pop].gather(1, tgt[:, None]).squeeze(1)
    assert torch.equal(chosen, d["correct"][pop])             # target == the hit's own candidate
    seed_true = d["seed_true"]
    assert torch.equal(seed_true[chosen], d["y_i"][d["fg_idx"]][pop])


def test_background_hits_target_null_and_exist_targets():
    d = prep()
    bkg = (~d["pop_fg"])
    assert bkg.any()
    assert (d["target"][bkg] == d["m"]).all()                 # null column index
    assert d["exist_t"].sum() == 3                            # one primary per true particle


def test_unreachable_hits_are_masked():
    ev = make_event()
    ev["beta"][80] = 0.0                                      # drop the third condensation point
    d = prep(ev)
    assert d["K"] == 2
    orphan = (d["correct"] < 0) & d["pop_fg"]
    assert orphan.any()
    assert (d["target"][orphan] == sp.IGNORE).all()


def test_membership_logits_shape_and_normalisation():
    d = prep()
    model = sp.build_model(d["hit_x"].size(1), 8, 16, 2)
    lg, c, cen, e = model(d["hit_x"], d["ox"], d["beta"], d["cp"], d["nb"])
    assert lg.shape == (d["fg_idx"].numel(), d["m"] + 1)
    assert c.shape[0] == d["K"] and cen.shape == (d["K"], 8) and e.shape == (d["K"],)
    w = lg.softmax(dim=1)
    assert torch.allclose(w.sum(dim=1), torch.ones(w.size(0)), atol=1e-5)


def test_loss_is_finite_and_differentiable():
    d = prep()
    model = sp.build_model(d["hit_x"].size(1), 8, 16, 2)
    loss, parts = sp.event_loss(model, d)
    assert torch.isfinite(loss) and set(parts) == {"mem", "exist", "merge"}
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_merge_all_gives_one_cluster_merge_none_keeps_candidates(monkeypatch):
    d = prep()
    model = sp.build_model(d["hit_x"].size(1), 8, 16, 2)
    real = model.merge_logits
    monkeypatch.setattr(model, "merge_logits",
                        lambda c, cen, pairs, w=None, nb=None: torch.full((pairs.size(0),), 9.0))
    lab, diag = sp.predict_labels(model, d, allow_null=False)
    assert diag["n_groups"] == 1 and set(lab.tolist()) == {0}
    monkeypatch.setattr(model, "merge_logits",
                        lambda c, cen, pairs, w=None, nb=None: torch.full((pairs.size(0),), -9.0))
    lab2, diag2 = sp.predict_labels(model, d, allow_null=False)
    assert diag2["n_groups"] == len(set(lab2.tolist())) <= d["K"]
    assert (lab2 >= 0).all()                                  # no null when allow_null=False


def test_existence_threshold_can_drop_candidates():
    d = prep()
    model = sp.build_model(d["hit_x"].size(1), 8, 16, 2)
    lab, diag = sp.predict_labels(model, d, exist_thr=1.1, allow_null=False)   # nothing survives
    assert diag["n_groups"] == 0 and (lab == -1).all()


def test_labels_are_contiguous_and_cover_foreground():
    d = prep()
    model = sp.build_model(d["hit_x"].size(1), 8, 16, 2)
    lab, _ = sp.predict_labels(model, d, exist_thr=0.0, allow_null=False)
    ids = sorted(set(lab.tolist()))
    assert ids == list(range(len(ids)))                       # contiguous from 0, no gaps
    assert lab.numel() == d["fg_idx"].numel()


def test_store_and_device_roundtrip_preserves_values():
    d = prep()
    back = sp.to_device(sp.to_store(d), DEV)
    assert "D" not in sp.to_store(d)
    assert torch.equal(back["target"], d["target"])
    assert torch.allclose(back["hit_x"], d["hit_x"], atol=1e-2)   # fp16 storage
    assert back["hit_x"].dtype == torch.float32


def test_perfect_features_are_learnable_end_to_end():
    """With one-hot true-particle features the decoder must reach a high ARI."""
    events = [prep(make_event(seed=s)) for s in range(6)]
    model = sp.build_model(events[0]["hit_x"].size(1), 8, 32, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(25):
        for d in events:
            loss, _ = sp.event_loss(model, d)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        d = events[0]
        lab, _ = sp.predict_labels(model, d)
        full = torch.full((d["n_hit"],), -1, dtype=torch.long)
        full[d["fg_idx"]] = lab
        q = sp.mb.event_quality(full, d["y_i"], d["y_sem"], d["fg"] & (d["y_sem"] >= 0))
    assert q["ari_particle"] > 0.8
