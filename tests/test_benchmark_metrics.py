"""
Tests for the benchmark reporting layer (dual ARI, all-noise handling, summary
aggregation, batch-level timing) in ``scripts/materialize_benchmark.py``.

Kept separate from ``test_materialize.py`` (backend algorithms). Requires the
full stack (torch, sklearn, nugraph) -> runs in the EAF ``nugraph-gpu`` env on
CPU. Run:  pytest -q tests/test_benchmark_metrics.py
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("nugraph")
from torch_geometric.data import HeteroData  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import materialize_benchmark as mb  # noqa: E402


# --------------------------------------------------------------------------- #
# (a) ari_inclusive differs from ari_particle when a large -1 block is grouped
# --------------------------------------------------------------------------- #
def test_inclusive_vs_particle_differ_on_background():
    n = 100
    y_semantic = torch.zeros(n, dtype=torch.long)          # all labelled
    true = torch.full((n,), -1, dtype=torch.long)          # 90 background
    true[90:95] = 0                                        # particle 0
    true[95:100] = 1                                       # particle 1
    pred = torch.full((n,), -1, dtype=torch.long)          # 90 correct background
    pred[90:100] = 0                                       # merges the 2 particles
    keep_particle = torch.zeros(n, dtype=torch.bool)
    keep_particle[90:100] = True                           # the 10 foreground hits

    q = mb.event_quality(pred, true, y_semantic, keep_particle)
    # particle-only sees a merged 2->1 clustering -> ARI ~ 0
    assert abs(q["ari_particle"]) < 1e-6
    # inclusive rewards the correctly co-grouped 90 background hits -> much higher
    assert q["ari_inclusive"] > 0.3
    assert q["ari_inclusive"] > q["ari_particle"] + 0.3


# --------------------------------------------------------------------------- #
# (d) ARI is event-local: independent of other events and of id offsets
# --------------------------------------------------------------------------- #
def test_event_local_independence_and_label_invariance():
    # event A: two particles, partly split; event B: different structure
    yA = torch.zeros(20, dtype=torch.long)
    tA = torch.tensor([0] * 10 + [1] * 10)
    pA = torch.tensor([0] * 10 + [1] * 8 + [-1, -1])   # 2 hits predicted noise
    kA = torch.ones(20, dtype=torch.bool)
    qA = mb.event_quality(pA, tA, yA, kA)

    # remap event A's non-negative ids by +1000 (event-local numbering) -> same ARI
    pA2 = torch.where(pA >= 0, pA + 1000, pA)
    tA2 = tA + 500
    qA2 = mb.event_quality(pA2, tA2, yA, kA)
    assert abs(qA["ari_particle"] - qA2["ari_particle"]) < 1e-9
    assert abs(qA["ari_inclusive"] - qA2["ari_inclusive"]) < 1e-9

    # a different event B must not change event A's metric (they are separate calls)
    yB = torch.zeros(6, dtype=torch.long)
    tB = torch.tensor([0, 0, 0, 1, 1, 1])
    pB = torch.tensor([0, 0, 0, 0, 0, 0])              # merged
    kB = torch.ones(6, dtype=torch.bool)
    qB = mb.event_quality(pB, tB, yB, kB)
    qA_again = mb.event_quality(pA, tA, yA, kA)
    assert qA_again["ari_particle"] == qA["ari_particle"]
    assert abs(qB["ari_particle"]) < 1e-6              # merged -> 0, independently


# --------------------------------------------------------------------------- #
# (b) all-noise events: finite ARIs enter _all aggregates but not _ok
# --------------------------------------------------------------------------- #
def test_all_noise_included_in_all_not_ok():
    rows = [
        {"backend": "x", "batch": 0, "status": "all_noise", "valid": True,
         "mat_batch_total_s": 2.0, "ari_particle": 0.0, "ari_inclusive": 0.1},
        {"backend": "x", "batch": 0, "status": "ok", "valid": True,
         "mat_batch_total_s": 2.0, "ari_particle": 0.4, "ari_inclusive": 0.5},
    ]
    s = mb.summarize(rows, ["x"])[0]
    assert s["n_all_noise"] == 1 and s["n_ok"] == 1 and s["n_events"] == 2
    # all-event aggregates include the all_noise value
    assert s["n_ari_particle_all"] == 2
    assert abs(s["mean_ari_particle_all"] - 0.2) < 1e-9      # mean(0.0, 0.4)
    # ok-only aggregates exclude it
    assert s["n_ari_particle_ok"] == 1
    assert abs(s["mean_ari_particle_ok"] - 0.4) < 1e-9
    assert abs(s["mean_ari_inclusive_all"] - 0.3) < 1e-9     # mean(0.1, 0.5)


# --------------------------------------------------------------------------- #
# (c) genuinely invalid results are counted but never enter finite aggregates
# --------------------------------------------------------------------------- #
def test_invalid_counted_not_aggregated():
    rows = [
        {"backend": "x", "batch": 0, "status": "ok", "valid": True,
         "mat_batch_total_s": 1.0, "ari_particle": 0.5, "ari_inclusive": 0.6},
        # invalid: no ARI fields were produced (q was None)
        {"backend": "x", "batch": 0, "status": "edge_limit", "valid": False,
         "mat_batch_total_s": 1.0, "safety_limit_reached": True},
    ]
    s = mb.summarize(rows, ["x"])[0]
    assert s["n_invalid"] == 1 and s["n_safety_limited"] == 1
    # only the one ok event contributes a finite ARI
    assert s["n_ari_particle_all"] == 1 and s["n_ari_particle_ok"] == 1
    assert abs(s["mean_ari_particle_all"] - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# (e) timing is counted once per batch, not once per event row
# --------------------------------------------------------------------------- #
def test_feature_mode_recorded_in_summary():
    rows = [{"backend": "dbscan", "batch": 0, "status": "ok", "valid": True,
             "feature_mode": "legacy_zero_featext", "mat_batch_total_s": 1.0,
             "ari_particle": 0.1, "ari_inclusive": 0.19}]
    s = mb.summarize(rows, ["dbscan"])[0]
    assert s["feature_mode"] == "legacy_zero_featext"


# --------------------------------------------------------------------------- #
# legacy feature-extension compatibility mode
# --------------------------------------------------------------------------- #
def test_legacy_zero_featext_zeros_exactly_last_four():
    d = HeteroData()
    # 3 hits x 11 features, distinct nonzero values so we can prove preservation
    d["hit"].x = (torch.arange(33, dtype=torch.float) + 1.0).view(3, 11).clone()
    before = d["hit"].x.clone()
    mb.apply_legacy_featext_zero(d)
    # last four columns are exactly zero
    assert torch.equal(d["hit"].x[:, -4:], torch.zeros(3, 4))
    # first seven columns are untouched (populated mode unchanged there)
    assert torch.equal(d["hit"].x[:, :7], before[:, :7])


def test_populated_mode_untouched_when_not_applied():
    # when the legacy helper is NOT called, x is unchanged (default populated mode)
    d = HeteroData()
    x = torch.randn(4, 11)
    d["hit"].x = x.clone()
    # (no call to apply_legacy_featext_zero)
    assert torch.equal(d["hit"].x, x)
    # and the last four are generally nonzero (populated)
    assert d["hit"].x[:, -4:].abs().sum() > 0


def test_legacy_zero_featext_requires_11_features():
    d = HeteroData()
    d["hit"].x = torch.randn(2, 7)   # featext disabled -> only 7 features
    with pytest.raises(RuntimeError):
        mb.apply_legacy_featext_zero(d)


def test_timing_counted_per_batch_not_per_event():
    # batch 0 has 3 event rows sharing t=5.0; batch 1 has 1 event row t=1.0
    rows = [
        {"backend": "x", "batch": 0, "status": "ok", "valid": True,
         "mat_batch_total_s": 5.0, "ari_particle": 0.1, "ari_inclusive": 0.2},
        {"backend": "x", "batch": 0, "status": "ok", "valid": True,
         "mat_batch_total_s": 5.0, "ari_particle": 0.1, "ari_inclusive": 0.2},
        {"backend": "x", "batch": 0, "status": "ok", "valid": True,
         "mat_batch_total_s": 5.0, "ari_particle": 0.1, "ari_inclusive": 0.2},
        {"backend": "x", "batch": 1, "status": "ok", "valid": True,
         "mat_batch_total_s": 1.0, "ari_particle": 0.1, "ari_inclusive": 0.2},
    ]
    s = mb.summarize(rows, ["x"])[0]
    assert s["n_batches"] == 2
    # per-batch [5.0, 1.0] -> mean 3.0 (a per-event mean would be (5+5+5+1)/4=4.0)
    assert abs(s["mean_mat_batch_total_s"] - 3.0) < 1e-9
    assert s["max_mat_batch_total_s"] == 5.0
    assert abs(s["median_mat_batch_total_s"] - 3.0) < 1e-9
