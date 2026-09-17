"""
Unit tests for NuGraph3 instance materialization backends.

Require the full stack (torch, torch_geometric, torch_scatter, sklearn); run on
CPU (no GPU needed). GPU quality/timing is measured separately with
``scripts/dune_materialization/mat_bench.py``.

Run:  pytest -q tests/test_materialize.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_scatter")
pytest.importorskip("torch_geometric")
pytest.importorskip("sklearn")

from torch_geometric.data import HeteroData, Batch  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from nugraph.models.nugraph3 import materialize as M  # noqa: E402
from nugraph.models.nugraph3.materialize import (  # noqa: E402
    BACKENDS, MaterializeTimer, materialize_labels, labels_to_particle_graph,
    has_torch_cluster, preflight)

D = 8  # instance_features


def make_blobs(k=3, n_per=40, sep=10.0, std=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.zeros(k, D)
    centers[:, 0] = torch.arange(k, dtype=torch.float) * sep
    ox, true = [], []
    for c in range(k):
        ox.append(centers[c] + std * torch.randn(n_per, D, generator=g))
        true.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(ox), torch.full((k * n_per,), 0.9), torch.cat(true)


def same_partition(a, b):
    return adjusted_rand_score(a.numpy(), b.numpy()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Basic correctness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", ["seed", "greedy", "graph", "dbscan"])
def test_recovers_blobs(backend):
    ox, beta, true = make_blobs(k=3, n_per=40)
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    dbscan = DBSCAN(eps=1.0, min_samples=3)
    labels, stats = materialize_labels(
        backend, ox, beta, mask, None, 1, dbscan=dbscan,
        t_beta=0.1, t_d=1.0, min_cluster_size=1)
    assert (labels >= 0).all()
    assert int(labels.max()) + 1 == 3
    assert same_partition(labels, true)
    assert stats[0].status == "ok" and stats[0].n_foreground == ox.size(0)


@pytest.mark.parametrize("backend", ["seed", "greedy"])
def test_all_noise_by_beta(backend):
    ox, _, _ = make_blobs()
    beta = torch.zeros(ox.size(0))
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    labels, stats = materialize_labels(backend, ox, beta, mask, None, 1,
                                       t_beta=0.1, t_d=1.0)
    assert (labels == -1).all()
    assert stats[0].status == "all_noise"


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_empty_foreground(backend):
    ox, beta, _ = make_blobs()
    mask = torch.zeros(ox.size(0), dtype=torch.bool)
    dbscan = DBSCAN(eps=1.0, min_samples=3)
    labels, stats = materialize_labels(backend, ox, beta, mask, None, 1,
                                       dbscan=dbscan, t_beta=0.1, t_d=1.0)
    assert (labels == -1).all()
    assert stats[0].status == "empty" and stats[0].n_foreground == 0


def test_min_cluster_size_demotes_small():
    ox, beta, _ = make_blobs(k=2, n_per=40)
    tiny = torch.tensor([[100.0] + [0.0] * (D - 1)]).repeat(2, 1)
    ox = torch.cat([ox, tiny])
    beta = torch.cat([beta, torch.tensor([0.9, 0.9])])
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    labels, _ = materialize_labels("seed", ox, beta, mask, None, 1,
                                   t_beta=0.1, t_d=1.0, min_cluster_size=5)
    assert int(labels.max()) + 1 == 2
    assert (labels[-2:] == -1).all()


def test_unknown_backend_raises():
    ox, beta, _ = make_blobs()
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    with pytest.raises(ValueError):
        materialize_labels("nope", ox, beta, mask, None, 1)


# --------------------------------------------------------------------------- #
# Memory-safety / high-occupancy
# --------------------------------------------------------------------------- #
def test_seed_nms_is_chunked_and_bounded():
    # many candidate seeds, tiny memory budget -> must chunk, temp stays bounded
    ox, beta, true = make_blobs(k=3, n_per=100)  # 300 seeds
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    max_pairs = 5000
    labels, stats = materialize_labels("seed", ox, beta, mask, None, 1,
                                       t_beta=0.1, t_d=1.0, max_pairs=max_pairs)
    s = stats[0]
    assert s.n_candidate_seeds == 300
    assert 0 < s.chunk_query < 300           # actually chunked
    assert s.largest_temp_elems <= max_pairs  # bounded by the budget
    assert same_partition(labels, true)       # still correct


def test_seed_truncation_is_explicit_and_flagged():
    ox, beta, _ = make_blobs(k=3, n_per=100)  # 300 candidate seeds
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    labels, stats = materialize_labels("seed", ox, beta, mask, None, 1,
                                       t_beta=0.1, t_d=1.0, max_seeds=10)
    s = stats[0]
    assert s.status == "seed_truncated"
    assert s.safety_limit_reached is True
    assert s.n_candidate_seeds == 300
    assert (labels >= 0).any()               # still produced a clustering


def test_seed_assign_chunked_bounded():
    ox, beta, true = make_blobs(k=4, n_per=60)
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    labels, stats = materialize_labels("seed", ox, beta, mask, None, 1,
                                       t_beta=0.1, t_d=1.0, max_pairs=1000)
    assert stats[0].largest_temp_elems <= 1000
    assert same_partition(labels, true)


def test_graph_dense_fallback_bounded(monkeypatch):
    # force the no-torch_cluster path and check it is row-chunked (no n x n)
    monkeypatch.setattr(M, "_HAS_TORCH_CLUSTER", False)
    ox, beta, true = make_blobs(k=3, n_per=200)  # 600 hits
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    max_pairs = 2000
    labels, stats = materialize_labels("graph", ox, beta, mask, None, 1,
                                       t_d=1.0, max_pairs=max_pairs)
    s = stats[0]
    assert s.graph_path == "dense_chunked"
    assert s.largest_temp_elems <= max_pairs       # never built 600x600
    assert s.cc_converged is True
    assert same_partition(labels, true)


def test_graph_edge_limit_flagged(monkeypatch):
    monkeypatch.setattr(M, "_HAS_TORCH_CLUSTER", False)
    ox = torch.zeros(200, D)                    # all identical -> dense graph
    beta = torch.full((200,), 0.9)
    mask = torch.ones(200, dtype=torch.bool)
    labels, stats = materialize_labels("graph", ox, beta, mask, None, 1,
                                       t_d=1.0, max_edges=100)
    assert stats[0].status == "edge_limit"
    assert stats[0].safety_limit_reached is True


def test_greedy_cluster_cap_flagged():
    # many singletons (far apart, tiny radius) -> many clusters -> cap triggers
    ox = torch.arange(50).float().view(-1, 1).repeat(1, D) * 100.0
    beta = torch.full((50,), 0.9)
    mask = torch.ones(50, dtype=torch.bool)
    labels, stats = materialize_labels("greedy", ox, beta, mask, None, 1,
                                       t_beta=0.1, t_d=1.0, max_clusters=5)
    assert stats[0].status == "greedy_capped"
    assert stats[0].safety_limit_reached is True


def test_status_temp_never_exceeds_budget_across_backends():
    ox, beta, _ = make_blobs(k=3, n_per=100)
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    dbscan = DBSCAN(eps=1.0, min_samples=3)
    budget = 3000
    for backend in ["seed", "graph", "greedy", "dbscan"]:
        with_tc = has_torch_cluster()
        _, stats = materialize_labels(backend, ox, beta, mask, None, 1,
                                      dbscan=dbscan, t_d=1.0, max_pairs=budget)
        # graph fast path (torch_cluster) reports O(E) not a dense temp
        if backend == "graph" and with_tc:
            continue
        assert stats[0].largest_temp_elems <= max(budget, ox.size(0))


# --------------------------------------------------------------------------- #
# Batching / bookkeeping
# --------------------------------------------------------------------------- #
def _make_batch(per_event_ox):
    datas = []
    for ox in per_event_ox:
        d = HeteroData()
        d["hit"].ox = ox
        d["hit"].num_nodes = ox.size(0)
        datas.append(d)
    return Batch.from_data_list(datas)


def test_no_cross_event_and_particle_batch():
    ox1, b1, _ = make_blobs(k=2, n_per=30, seed=1)
    ox2, b2, _ = make_blobs(k=3, n_per=30, seed=2)
    batch = _make_batch([ox1, ox2])
    h = batch["hit"]
    ox = torch.cat([ox1, ox2])
    beta = torch.cat([b1, b2])
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    labels, stats = materialize_labels("seed", ox, beta, mask, h.ptr,
                                       batch.num_graphs, t_beta=0.1, t_d=1.0)
    assert len(stats) == 2
    lo, hi = int(h.ptr[0]), int(h.ptr[1])
    assert int(labels[lo:hi].max()) + 1 == 2
    lo, hi = int(h.ptr[1]), int(h.ptr[2])
    assert int(labels[lo:hi].max()) + 1 == 3
    labels_to_particle_graph(batch, labels)
    assert batch["particle"].x.size(0) == 5
    assert batch["particle"].batch.tolist() == [0, 0, 1, 1, 1]
    assert batch["particle"].ptr.tolist() == [0, 2, 5]
    ei = batch[("hit", "cluster", "particle")].edge_index
    assert ei.size(1) == int((labels >= 0).sum()) and int(ei[1].max()) < 5


def test_particle_nodes_single_graph():
    ox, beta, _ = make_blobs(k=3, n_per=20)
    d = HeteroData()
    d["hit"].ox = ox
    d["hit"].num_nodes = ox.size(0)
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    labels, _ = materialize_labels("greedy", ox, beta, mask, None, 1,
                                   t_beta=0.1, t_d=1.0)
    labels_to_particle_graph(d, labels)
    assert d["particle"].x.size(0) == 3
    ei = d[("hit", "cluster", "particle")].edge_index
    assert ei.size(1) == 60 and int(ei[1].max()) == 2


# --------------------------------------------------------------------------- #
# Preflight / timing
# --------------------------------------------------------------------------- #
def test_preflight_feasibility_tiers():
    rep = preflight(BACKENDS, max_hits=44000)
    assert rep["max_hits_checked"] == 44000
    assert "torch_cluster_installed" in rep and "feasibility_legend" in rep
    for b in BACKENDS:
        info = rep["backends"][b]
        # every backend reports the distinct tiers (not just "can process")
        assert "memory_safe" in info
        assert "offline_test_feasible" in info
        assert "train_step_feasible" in info
    # dbscan must NOT be advertised as train-step feasible, nor fully memory-safe
    assert rep["backends"]["dbscan"]["train_step_feasible"] == "no"
    assert rep["backends"]["dbscan"]["memory_safe"] == "gpu_only"
    # graph without torch_cluster is the emergency dense path, not train-feasible
    if not has_torch_cluster():
        g = rep["backends"]["graph"]
        assert g["dense_fallback_only"] is True
        assert g["train_step_feasible"] == "no"


def test_invalid_status_flags():
    # seed_truncated / greedy_capped / edge_limit / cc_not_converged are invalid
    from nugraph.models.nugraph3.materialize import MatStatus, INVALID_STATUSES
    for s in INVALID_STATUSES:
        assert MatStatus(method="x", status=s).valid is False
    for s in ("ok", "empty", "all_noise"):
        assert MatStatus(method="x", status=s).valid is True


def test_materialization_is_autograd_inert():
    # enabling materialization must not create autograd history on ox
    ox, beta, _ = make_blobs(k=3, n_per=30)
    ox = ox.clone().requires_grad_(True)
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    labels, _ = materialize_labels("seed", ox.detach(), beta, mask, None, 1,
                                   t_beta=0.1, t_d=1.0)
    assert labels.dtype == torch.long and not labels.requires_grad
    # an independent loss on ox still backprops normally; materialization added
    # nothing to the graph
    (ox.sum()).backward()
    assert ox.grad is not None


def test_timer_disabled_is_noop():
    t = MaterializeTimer(enabled=False)
    with t.section("x"):
        _ = torch.randn(10, 10)
    assert t.times == {}


def test_timer_enabled_records():
    t = MaterializeTimer(enabled=True, cuda=False)
    with t.section("x"):
        _ = torch.randn(100, 100) @ torch.randn(100, 100)
    assert "x" in t.times and t.times["x"] >= 0.0


def test_seed_t_assign_default_is_t_d():
    """t_assign=None reproduces the original single-radius seed labels exactly."""
    ox, beta, _ = make_blobs(k=3, n_per=40, sep=10.0, std=0.3)
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    a, _ = materialize_labels("seed", ox, beta, mask, None, 1, t_beta=0.1, t_d=0.5)
    b, _ = materialize_labels("seed", ox, beta, mask, None, 1, t_beta=0.1, t_d=0.5,
                              t_assign=None)
    assert torch.equal(a, b)


def test_seed_t_assign_decouples_nms_and_assignment():
    """Small NMS radius keeps one seed per blob; a larger t_assign then assigns
    the wide blobs completely instead of leaving most hits as noise."""
    ox, beta, true = make_blobs(k=3, n_per=60, sep=10.0, std=1.0)
    beta = torch.full_like(beta, 0.05)
    for c in range(3):  # one clear condensation point per blob
        beta[c * 60] = 0.9
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    tight, _ = materialize_labels("seed", ox, beta, mask, None, 1, t_beta=0.5, t_d=0.3)
    wide, st = materialize_labels("seed", ox, beta, mask, None, 1, t_beta=0.5, t_d=0.3,
                                  t_assign=float("inf"))
    assert (tight < 0).float().mean() > 0.5
    assert (wide < 0).sum() == 0
    assert st[0].n_surviving_seeds == 3
    assert adjusted_rand_score(true.numpy(), wide.numpy()) > 0.9


def test_seed_assign_chunk_not_capped_by_seed_count(monkeypatch):
    """With a single surviving seed the assignment must not run one hit per cdist."""
    ox, beta, _ = make_blobs(k=1, n_per=500, std=0.1)
    beta = torch.full_like(beta, 0.05)
    beta[0] = 0.9
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    calls = []
    real = torch.cdist
    monkeypatch.setattr(M.torch, "cdist", lambda a, b, *k, **kw: (calls.append(a.shape[0]), real(a, b, *k, **kw))[1])
    labels, st = materialize_labels("seed", ox, beta, mask, None, 1, t_beta=0.5, t_d=5.0)
    assert st[0].n_surviving_seeds == 1
    assert len(calls) == 2          # one NMS call + ONE assignment call
    assert (labels >= 0).all()


@pytest.mark.parametrize("k", [1, 3, 25])
def test_seed_assignment_independent_of_chunk_size(k):
    """With K <= 25 surviving seeds the assignment uses the exact per-pair cdist
    kernel, so labels must not change with max_pairs (the assignment chunk size).
    (K > 25 keeps the original matrix-multiplication kernel; its chunk invariance is
    checked on real events by the benchmark regression gate, not here.)"""
    ox, _, _ = make_blobs(k=k, n_per=40, sep=3.0, std=1.0, seed=1)
    ox = ox * 25.0  # large coordinates: where the mm formula loses precision
    beta = torch.full((ox.size(0),), 0.05)
    beta[torch.arange(k) * 40] = 0.9
    mask = torch.ones(ox.size(0), dtype=torch.bool)
    kw = dict(t_beta=0.5, t_d=1.0, t_assign=40.0)
    small, s1 = materialize_labels("seed", ox, beta, mask, None, 1, max_pairs=k, **kw)
    large, s2 = materialize_labels("seed", ox, beta, mask, None, 1, **kw)
    assert s1[0].n_surviving_seeds == s2[0].n_surviving_seeds == k
    assert torch.equal(small, large)
