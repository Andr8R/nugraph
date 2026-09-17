"""
NuGraph3 instance materialization backends (memory-safe).

Turns the per-hit object-condensation outputs (condensation score ``beta`` and
latent coordinate ``ox``) into discrete predicted particle instances, then into
``particle`` nodes (store ``N_IP``) and ``("hit","cluster","particle")`` edges
(store ``E_H_IP``).

Backends (all share one input/output contract for fair comparison):

* ``"dbscan"`` -- sklearn CPU DBSCAN (baseline / fallback);
* ``"seed"``   -- Candidate A: beta-seeded condensation-point assignment (GPU);
* ``"graph"``  -- Candidate B: radius graph + connected components (GPU);
* ``"greedy"`` -- Candidate C: sequential greedy condensation (GPU, oracle).

Label convention: each backend produces a per-hit ``labels`` LongTensor that is
**event-local** (``-1`` = background/noise, ``>=0`` = contiguous cluster id
within the event). ``labels_to_particle_graph`` builds the batched stores with a
correct ``particle.batch``.

MEMORY SAFETY (this dataset reaches >44k hits / >100k edges per event)
---------------------------------------------------------------------
No backend forms an unbounded dense pairwise matrix. Every pairwise operation is
**chunked** so the largest temporary never exceeds ``max_pairs`` elements
(configurable memory budget). Specifically:

* Candidate A's non-max suppression is chunked over query seeds (temp
  ``chunk x K_candidate``, not ``K_candidate x K_candidate``), and its assignment
  is chunked over hits (temp ``chunk x K_surviving``).
* Candidate B builds its radius graph either sparsely via ``torch_cluster`` (fast
  path) or via a **row-chunked** dense scan (safe path; O(n^2) compute but O(E)
  memory) -- it never materializes a full ``n x n`` matrix, and it no longer
  refuses large events; it reports the path taken and the edge count, with a
  ``max_edges`` safety cap.
* Candidate C uses one ``1 x n`` distance row per cluster (already bounded) plus
  a configurable ``max_clusters`` cap.

Every backend returns a per-event ``MatStatus`` (see below) recording the actual
counts, chunk sizes, largest temporary allocation, convergence, and any
safety-limit that was hit -- so failed / safety-limited events are visible and do
not silently corrupt benchmark averages.

Dependencies: ``torch``, ``torch_scatter`` (required). ``sklearn`` for
``"dbscan"``. ``torch_cluster`` is optional and only accelerates ``"graph"``.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from time import perf_counter
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import cumsum
from torch_scatter import scatter_min

from .types import Data, N_IP, E_H_IP

# optional GPU spatial-index dependency (only Candidate B fast path uses it)
try:
    from torch_cluster import radius_graph as _radius_graph
    _HAS_TORCH_CLUSTER = True
except Exception:  # pragma: no cover - depends on EAF env
    _radius_graph = None
    _HAS_TORCH_CLUSTER = False

BACKENDS = ("dbscan", "seed", "graph", "greedy")

# statuses that are legitimate materialization *outcomes* (a real clustering,
# possibly with zero clusters). Anything else is an invalid/failed result that
# must NOT be interpreted as a real prediction.
VALID_STATUSES = ("ok", "empty", "all_noise")
# safety-limited / failed statuses -- an invalid clustering, not a success.
INVALID_STATUSES = ("edge_limit", "cc_not_converged", "seed_truncated",
                    "greedy_capped", "error")

# ---- default resource limits (all overridable via materialize_labels) ------ #
DEFAULT_MAX_PAIRS = 4_000_000      # max elements in any pairwise temp (~16 MB fp32)
DEFAULT_MAX_SEEDS = 20_000         # cap on candidate seeds (A); explicit truncation
DEFAULT_MAX_EDGES = 8_000_000      # cap on radius-graph edges (B)
DEFAULT_CC_MAX_ITERS = 200         # connected-components iteration cap (B)
DEFAULT_MAX_CLUSTERS = 20_000      # greedy cluster cap (C)


# --------------------------------------------------------------------------- #
# Per-event status record
# --------------------------------------------------------------------------- #
@dataclass
class MatStatus:
    """Structured per-event materialization status (recorded for every event)."""
    method: str
    event: int = 0
    status: str = "ok"              # ok|empty|all_noise|cc_not_converged|
    #                                 edge_limit|seed_truncated|greedy_capped|error
    n_foreground: int = 0
    n_candidate_seeds: int = -1     # A
    n_surviving_seeds: int = -1     # A
    n_clusters: int = 0
    n_edges: int = -1               # B
    cc_iters: int = -1              # B
    cc_converged: Optional[bool] = None  # B
    chunk_query: int = -1
    largest_temp_elems: int = 0
    safety_limit_reached: bool = False
    graph_path: str = ""            # B: "torch_cluster" | "dense_chunked"
    note: str = ""

    @property
    def largest_temp_mb(self) -> float:
        return self.largest_temp_elems * 4 / 1e6  # fp32

    @property
    def valid(self) -> bool:
        """True only for a legitimate clustering outcome (see VALID_STATUSES).

        edge_limit / cc_not_converged / seed_truncated / greedy_capped / error
        are NOT valid clusterings and must not be treated as real predictions.
        """
        return self.status in VALID_STATUSES

    def as_row(self) -> dict:
        d = asdict(self)
        d["largest_temp_mb"] = self.largest_temp_mb
        d["valid"] = self.valid
        return d


# --------------------------------------------------------------------------- #
# Timing helper (unchanged: no-op with zero overhead when disabled)
# --------------------------------------------------------------------------- #
class MaterializeTimer:
    """Optional, CUDA-synchronized section timer. No-op when ``enabled`` False."""
    def __init__(self, enabled: bool = False, cuda: bool = True):
        self.enabled = enabled
        self.cuda = cuda and torch.cuda.is_available()
        self.times: dict[str, float] = {}

    def reset(self) -> None:
        self.times = {}

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if self.cuda:
            torch.cuda.synchronize()
        t0 = perf_counter()
        try:
            yield
        finally:
            if self.cuda:
                torch.cuda.synchronize()
            self.times[name] = self.times.get(name, 0.0) + (perf_counter() - t0)


# --------------------------------------------------------------------------- #
# Chunk-size helper
# --------------------------------------------------------------------------- #
def _chunk_for(n_key: int, max_pairs: int) -> int:
    """Largest query-chunk size whose (chunk x n_key) temp fits ``max_pairs``."""
    return max(1, min(n_key if n_key > 0 else 1, max_pairs // max(1, n_key)))


# --------------------------------------------------------------------------- #
# Candidate A: beta-seeded condensation-point assignment (chunked, bounded)
# --------------------------------------------------------------------------- #
def _seed_one(oxf: Tensor, bf: Tensor, t_beta: float, t_d: float,
              max_pairs: int, max_seeds: int,
              timer: MaterializeTimer, st: MatStatus,
              t_assign: Optional[float] = None) -> Tensor:
    """
    Assign foreground hits to beta-seeded condensation points, bounded memory.

    Inputs: ``oxf`` (n_fg, d), ``bf`` (n_fg,). Output: (n_fg,) ids, -1 = noise.

    Non-max suppression is chunked over query seeds so the largest temporary is
    ``chunk x K_candidate`` (<= ``max_pairs``), never ``K_candidate^2``. If the
    number of candidate seeds exceeds ``max_seeds`` the candidate set is
    explicitly truncated to the top-``max_seeds`` by beta (recorded via
    ``status="seed_truncated"`` / ``safety_limit_reached``). Assignment is chunked
    over hits (temp ``chunk x K_surviving``).

    ``t_d`` is the non-max-suppression radius. ``t_assign`` is the hit-assignment radius
    (``None`` = ``t_d``, the original single-radius behaviour; ``inf`` = assign every hit
    to its nearest condensation point). Decoupling them matters when particles are wider
    in latent space than the spacing of their condensation points.
    """
    device = oxf.device
    n = oxf.size(0)
    oxf = oxf.float()
    bf = bf.float()

    seed_idx = (bf > t_beta).nonzero(as_tuple=False).squeeze(1)
    st.n_candidate_seeds = int(seed_idx.numel())
    if seed_idx.numel() == 0:
        st.status = "all_noise"
        st.n_surviving_seeds = 0
        return torch.full((n,), -1, dtype=torch.long, device=device)

    # explicit safety cap on candidate seeds (bounds NMS compute, not memory)
    if max_seeds and seed_idx.numel() > max_seeds:
        keep = torch.topk(bf[seed_idx], max_seeds).indices
        seed_idx = seed_idx[keep]
        st.status = "seed_truncated"
        st.safety_limit_reached = True
        st.note = f"candidate seeds {st.n_candidate_seeds} capped to {max_seeds}"

    with timer.section("cluster/seed_nms"):
        S = oxf[seed_idx]                 # (Ks, d)
        bS = bf[seed_idx]                 # (Ks,)
        ks = S.size(0)
        idx = torch.arange(ks, device=device)
        chunk = _chunk_for(ks, max_pairs)
        st.chunk_query = chunk
        st.largest_temp_elems = max(st.largest_temp_elems, chunk * ks)
        dominated = torch.zeros(ks, dtype=torch.bool, device=device)
        for i in range(0, ks, chunk):
            q = slice(i, min(i + chunk, ks))
            qi = idx[q]
            dqs = torch.cdist(S[q], S)    # (c, Ks) bounded by max_pairs
            within = dqs <= t_d
            # seed j dominates query i if closer-or-equal & higher beta,
            # ties broken by lower index; exclude self (idx==qi)
            better = (bS[None, :] > bS[q][:, None]) | (
                (bS[None, :] == bS[q][:, None]) & (idx[None, :] < qi[:, None]))
            not_self = idx[None, :] != qi[:, None]
            dominated[q] = (within & better & not_self).any(dim=1)
        cp_local = seed_idx[~dominated]   # condensation points (indices into fg)
    st.n_surviving_seeds = int(cp_local.numel())

    ta = t_d if t_assign is None else t_assign
    with timer.section("cluster/seed_assign"):
        C = oxf[cp_local]                 # (K, d)
        k = C.size(0)
        # query = hits, key = K condensation points: bound the chunk by n, not by K
        # (_chunk_for(k, .) capped it at K, i.e. one hit per cdist call when K == 1)
        chunk = max(1, min(n, max_pairs // max(1, k)))
        # torch.cdist's default switches to the less precise matrix-multiplication
        # formula when either input has > 25 rows. The original chunk (= K rows) thus
        # used the exact per-pair kernel whenever K <= 25; keep exactly that choice so
        # labels do not depend on the chunk size (hits near t_assign would flip).
        mode = ("donot_use_mm_for_euclid_dist" if k <= 25
                else "use_mm_for_euclid_dist")
        st.largest_temp_elems = max(st.largest_temp_elems, chunk * k)
        labels = torch.full((n,), -1, dtype=torch.long, device=device)
        for i in range(0, n, chunk):
            sl = slice(i, min(i + chunk, n))
            dhc = torch.cdist(oxf[sl], C, compute_mode=mode)  # (c, K) <= max_pairs
            nn_d, nn_j = dhc.min(dim=1)
            labels[sl] = torch.where(nn_d <= ta, nn_j,
                                     torch.full_like(nn_j, -1))
    return labels


# --------------------------------------------------------------------------- #
# Candidate C: sequential greedy condensation (bounded memory + cluster cap)
# --------------------------------------------------------------------------- #
def _greedy_one(oxf: Tensor, bf: Tensor, t_beta: float, t_d: float,
                max_clusters: int, timer: MaterializeTimer,
                st: MatStatus) -> Tensor:
    """Greedy condensation. One (1 x n) distance row per cluster (bounded).

    A configurable ``max_clusters`` cap prevents an unbounded per-cluster loop on
    pathological events (recorded via ``status="greedy_capped"``).
    """
    device = oxf.device
    n = oxf.size(0)
    oxf = oxf.float()
    labels = torch.full((n,), -1, dtype=torch.long, device=device)
    assigned = torch.zeros(n, dtype=torch.bool, device=device)
    bwork = bf.float().clone()
    st.largest_temp_elems = max(st.largest_temp_elems, n)  # the 1 x n row
    cid = 0
    with timer.section("cluster/greedy_loop"):
        while True:
            if max_clusters and cid >= max_clusters:
                st.status = "greedy_capped"
                st.safety_limit_reached = True
                st.note = f"cluster count reached cap {max_clusters}"
                break
            bwork[assigned] = -1.0
            bmax, imax = bwork.max(dim=0)
            if bmax.item() <= t_beta:
                break
            d = torch.cdist(oxf[imax:imax + 1], oxf).squeeze(0)  # (n,)
            sel = (d <= t_d) & (~assigned)
            labels[sel] = cid
            assigned |= sel
            cid += 1
    return labels


# --------------------------------------------------------------------------- #
# Candidate B: radius graph + connected components (bounded memory)
# --------------------------------------------------------------------------- #
def _connected_components(n: int, edge_index: Tensor, max_iters: int):
    """
    Connected components via label propagation with pointer jumping.

    Returns ``(labels, iters, converged)``. Memory is O(n + E). Callers MUST
    treat ``converged=False`` as an invalid result (do not silently accept
    partial labels).
    """
    device = edge_index.device
    if edge_index.numel() == 0:
        return torch.arange(n, device=device), 0, True
    row = torch.cat([edge_index[0], edge_index[1]])
    col = torch.cat([edge_index[1], edge_index[0]])
    labels = torch.arange(n, device=device)
    converged = False
    used = 0
    for it in range(max_iters):
        used = it + 1
        m = scatter_min(labels[col], row, dim=0, dim_size=n)[0]
        new = torch.minimum(labels, m)
        new = new[new]                      # pointer jumping
        if torch.equal(new, labels):
            converged = True
            break
        labels = new
    labels = torch.unique(labels, return_inverse=True)[1]
    return labels, used, converged


def _radius_edges_dense_chunked(oxf: Tensor, t_d: float, max_pairs: int,
                                max_edges: int):
    """Row-chunked radius graph without ``torch_cluster``: never builds n x n.

    Returns ``(edge_index, largest_temp_elems, edge_limit_hit)``. Compute is
    O(n^2) but memory is bounded to ``chunk x n`` per step plus the O(E) output.
    """
    device = oxf.device
    n = oxf.size(0)
    chunk = _chunk_for(n, max_pairs)
    rows, cols = [], []
    n_edges = 0
    edge_limit_hit = False
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        d = torch.cdist(oxf[sl], oxf)          # (c, n) bounded by max_pairs
        c = d.size(0)
        ii = torch.arange(i, i + c, device=device)
        m = (d <= t_d)
        m[torch.arange(c, device=device), ii] = False  # drop self loops
        idx = m.nonzero(as_tuple=False)
        if idx.numel():
            rows.append(idx[:, 0] + i)
            cols.append(idx[:, 1])
            n_edges += idx.size(0)
        if n_edges > max_edges:
            edge_limit_hit = True
            break
    if rows:
        edge_index = torch.stack([torch.cat(rows), torch.cat(cols)])
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    return edge_index, chunk * n, edge_limit_hit


def _graph_one(oxf: Tensor, t_d: float, max_pairs: int, max_edges: int,
               cc_max_iters: int, timer: MaterializeTimer,
               st: MatStatus) -> Tensor:
    """Radius graph + connected components (memory-bounded).

    Uses ``torch_cluster.radius_graph`` when available (fast, sparse); otherwise a
    row-chunked dense scan (safe, O(n^2) compute). Records the path, edge count,
    CC iterations and convergence. Never materializes a full n x n matrix and
    never silently refuses an event.
    """
    device = oxf.device
    n = oxf.size(0)
    oxf = oxf.float()
    edge_limit_hit = False
    with timer.section("cluster/graph_build"):
        if _HAS_TORCH_CLUSTER:
            st.graph_path = "torch_cluster"
            edge_index = _radius_graph(oxf, r=t_d, loop=False,
                                       max_num_neighbors=n)
            st.n_edges = int(edge_index.size(1))
            if st.n_edges > max_edges:
                edge_limit_hit = True
        else:
            st.graph_path = "dense_chunked"
            edge_index, temp_elems, edge_limit_hit = _radius_edges_dense_chunked(
                oxf, t_d, max_pairs, max_edges)
            st.n_edges = int(edge_index.size(1))
            st.chunk_query = _chunk_for(n, max_pairs)
            st.largest_temp_elems = max(st.largest_temp_elems, temp_elems)

    if edge_limit_hit:
        st.status = "edge_limit"
        st.safety_limit_reached = True
        st.note = f"radius-graph edges exceeded max_edges={max_edges}"
        # explicit invalid result: every hit its own singleton
        return torch.arange(n, dtype=torch.long, device=device)

    with timer.section("cluster/graph_cc"):
        labels, iters, converged = _connected_components(n, edge_index,
                                                         cc_max_iters)
    st.cc_iters = iters
    st.cc_converged = converged
    if not converged:
        st.status = "cc_not_converged"
        st.safety_limit_reached = True
        st.note = f"connected components did not converge in {cc_max_iters} iters"
    return labels


# --------------------------------------------------------------------------- #
# DBSCAN baseline (CPU; bounded memory, records status)
# --------------------------------------------------------------------------- #
def _dbscan_one(oxf: Tensor, dbscan, timer: MaterializeTimer,
                st: MatStatus) -> Tensor:
    """sklearn DBSCAN on one event (CPU). Includes GPU->CPU->GPU transfers."""
    with timer.section("cluster/dbscan_to_cpu"):
        arr = oxf.detach().to(torch.float32).cpu().numpy()  # (n_fg, d), O(n)
    with timer.section("cluster/dbscan_fit"):
        lab = dbscan.fit_predict(arr)
    with timer.section("cluster/dbscan_to_gpu"):
        out = torch.as_tensor(lab, dtype=torch.long, device=oxf.device)
    st.largest_temp_elems = max(st.largest_temp_elems, oxf.numel())
    return out


# --------------------------------------------------------------------------- #
# Batched driver
# --------------------------------------------------------------------------- #
def materialize_labels(method: str, ox: Tensor, beta: Tensor, mask: Tensor,
                       ptr: Optional[Tensor], num_graphs: int, *,
                       dbscan=None, t_beta: float = 0.1, t_d: float = 0.3,
                       t_assign: Optional[float] = None,
                       min_cluster_size: int = 1,
                       max_pairs: int = DEFAULT_MAX_PAIRS,
                       max_seeds: int = DEFAULT_MAX_SEEDS,
                       max_edges: int = DEFAULT_MAX_EDGES,
                       cc_max_iters: int = DEFAULT_CC_MAX_ITERS,
                       max_clusters: int = DEFAULT_MAX_CLUSTERS,
                       timer: Optional[MaterializeTimer] = None):
    """
    Run the selected backend; return ``(labels, stats)``.

    ``labels``: (n_hit,) LongTensor of per-hit event-local cluster ids
    (-1 = background/noise). ``stats``: ``list[MatStatus]``, one per event, with
    counts, chunk sizes, largest temporary allocation, convergence and any
    safety-limit reached. Memory for every backend is bounded by ``max_pairs``
    (and O(E) for the graph backend, capped by ``max_edges``).
    """
    if timer is None:
        timer = MaterializeTimer(enabled=False)
    device = ox.device
    labels = torch.full((ox.size(0),), -1, dtype=torch.long, device=device)
    stats: list[MatStatus] = []

    bounds = ([(0, ox.size(0))] if ptr is None
              else [(int(ptr[g]), int(ptr[g + 1])) for g in range(num_graphs)])

    for ev, (lo, hi) in enumerate(bounds):
        st = MatStatus(method=method, event=ev)
        m = mask[lo:hi]
        fg = m.nonzero(as_tuple=False).squeeze(1)
        st.n_foreground = int(fg.numel())
        if fg.numel() == 0:
            st.status = "empty"
            stats.append(st)
            continue
        oxf = ox[lo:hi][fg]
        bf = beta[lo:hi][fg]

        if method == "dbscan":
            lab_fg = _dbscan_one(oxf, dbscan, timer, st)
        elif method == "seed":
            lab_fg = _seed_one(oxf, bf, t_beta, t_d, max_pairs, max_seeds,
                               timer, st, t_assign=t_assign)
        elif method == "greedy":
            lab_fg = _greedy_one(oxf, bf, t_beta, t_d, max_clusters, timer, st)
        elif method == "graph":
            lab_fg = _graph_one(oxf, t_d, max_pairs, max_edges, cc_max_iters,
                                timer, st)
        else:
            raise ValueError(f"Unknown backend '{method}'. Choose {BACKENDS}.")

        if min_cluster_size > 1 and lab_fg.numel():
            lab_fg = _apply_min_size(lab_fg, min_cluster_size)

        st.n_clusters = int(lab_fg.max().item()) + 1 if (
            lab_fg.numel() and lab_fg.max() >= 0) else 0
        if st.status == "ok" and st.n_clusters == 0:
            st.status = "all_noise"

        li = torch.full((hi - lo,), -1, dtype=torch.long, device=device)
        li[fg] = lab_fg
        labels[lo:hi] = li
        stats.append(st)

    return labels, stats


def _apply_min_size(lab: Tensor, min_cluster_size: int) -> Tensor:
    """Demote clusters smaller than ``min_cluster_size`` to noise and re-contig."""
    out = lab.clone()
    valid = out >= 0
    if not valid.any():
        return out
    ids, counts = torch.unique(out[valid], return_counts=True)
    small = ids[counts < min_cluster_size]
    if small.numel():
        out[torch.isin(out, small)] = -1
    valid = out >= 0
    if valid.any():
        _, inv = torch.unique(out[valid], return_inverse=True)
        out[valid] = inv
    return out


# --------------------------------------------------------------------------- #
# Shared assembly: event-local labels -> particle nodes + membership edges
# --------------------------------------------------------------------------- #
def labels_to_particle_graph(data: Data, labels: Tensor,
                             timer: Optional[MaterializeTimer] = None) -> None:
    """
    Build ``data[N_IP]`` (particle nodes) and ``data[E_H_IP]`` (hit->particle
    edges) from per-hit event-local ``labels``, with correct PyG batching.

    Memory is bounded: the only sizeable output is the edge index, of size
    ``(2, n_assigned) <= (2, n_hit)``. Fixes the previously empty
    ``particle.batch`` (each particle gets its event's graph id).
    """
    if timer is None:
        timer = MaterializeTimer(enabled=False)
    h = data["hit"]
    device = h.ox.device
    is_batch = (hasattr(h, "ptr") and h.ptr is not None
                and hasattr(data, "num_graphs"))

    with timer.section("assemble"):
        if is_batch:
            n_graphs = data.num_graphs
            ptr = h.ptr
            x_ip, e_h_ip, counts = [], [], []
            for g in range(n_graphs):
                lo, hi = int(ptr[g]), int(ptr[g + 1])
                le = labels[lo:hi]
                k = int(le.max().item()) + 1 if (le.numel() and le.max() >= 0) else 0
                counts.append(k)
                x_ip.append(torch.empty(k, 0, dtype=h.ox.dtype, device=device))
                sel = le >= 0
                e_h_ip.append(torch.stack(
                    (sel.nonzero(as_tuple=False).squeeze(1), le[sel])).long())

            counts_t = torch.tensor(counts, dtype=torch.long, device=device)
            data[N_IP].x = torch.cat(x_ip, dim=0)
            data[N_IP].batch = torch.repeat_interleave(
                torch.arange(n_graphs, device=device), counts_t)
            data[N_IP].ptr = cumsum(counts_t)
            # pylint: disable=protected-access
            data._slice_dict[N_IP] = {"x": data[N_IP].ptr}
            data._inc_dict[N_IP] = {
                "x": torch.zeros(n_graphs, dtype=torch.long, device=device)}

            e_inc = torch.stack((h.ptr[:-1], data[N_IP].ptr[:-1]), dim=1).unsqueeze(2)
            data[E_H_IP].edge_index = torch.cat(
                [e + inc for e, inc in zip(e_h_ip, e_inc)], dim=1)
            # pylint: disable=protected-access
            data._slice_dict[E_H_IP] = {
                "edge_index": cumsum(torch.tensor(
                    [e.size(1) for e in e_h_ip], dtype=torch.long, device=device))}
            data._inc_dict[E_H_IP] = {"edge_index": e_inc}
        else:
            k = int(labels.max().item()) + 1 if (labels.numel() and labels.max() >= 0) else 0
            data[N_IP].x = torch.empty(k, 0, dtype=h.ox.dtype, device=device)
            sel = labels >= 0
            data[E_H_IP].edge_index = torch.stack(
                (sel.nonzero(as_tuple=False).squeeze(1), labels[sel])).long()


# --------------------------------------------------------------------------- #
# Preflight / dry-run
# --------------------------------------------------------------------------- #
def has_torch_cluster() -> bool:
    """Whether ``torch_cluster`` is importable (Candidate B fast path)."""
    return _HAS_TORCH_CLUSTER


def preflight(methods=BACKENDS, max_hits: int = 44000, *,
              max_pairs: int = DEFAULT_MAX_PAIRS, max_seeds: int = DEFAULT_MAX_SEEDS,
              max_edges: int = DEFAULT_MAX_EDGES,
              cc_max_iters: int = DEFAULT_CC_MAX_ITERS,
              max_clusters: int = DEFAULT_MAX_CLUSTERS) -> dict:
    """
    Report, without running the model, whether each backend can safely process
    the largest known event, the execution path it will take, its size limits,
    and the configured chunking / safety limits.

    ``max_hits`` is the largest event size to check against (dataset ~44k).
    """
    report = {
        "torch_cluster_installed": _HAS_TORCH_CLUSTER,
        "cuda_available": torch.cuda.is_available(),
        "max_hits_checked": max_hits,
        "limits": {"max_pairs": max_pairs, "max_seeds": max_seeds,
                   "max_edges": max_edges, "cc_max_iters": cc_max_iters,
                   "max_clusters": max_clusters},
        "feasibility_legend": {
            "memory_safe": "will not OOM the GPU for the largest event",
            "offline_test_feasible": "expected to complete for an offline "
                                     "correctness test on selected events",
            "train_step_feasible": "expected to be cheap enough for EVERY "
                                   "training step -- 'measure' = decide from the "
                                   "EAF throughput runs, not asserted here",
            "efficient_path": "uses the fast torch_cluster path",
            "dense_fallback_only": "only the emergency O(n^2) dense path is "
                                   "available (correctness/testing, NOT train-time)",
        },
        "backends": {},
    }
    for method in methods:
        info: dict = {"memory_safe": True, "efficient_path": False,
                      "dense_fallback_only": False}
        if method == "dbscan":
            info["path"] = "cpu_sklearn"
            info["gpu_temp_mb"] = round(max_hits * 8 * 4 / 1e6, 3)  # (n,d) copy
            info["offline_test_feasible"] = True
            info["train_step_feasible"] = "no"
            # sklearn DBSCAN is NOT guaranteed memory-safe on CPU:
            info["memory_safe"] = "gpu_only"
            info["note"] = (
                "GPU impact limited to the (n,d) transfer/output; but sklearn "
                "DBSCAN CPU memory AND runtime can be large / worst-case "
                "quadratic depending on neighbourhood density and (eps, "
                "min_samples). Existing baseline, NOT a guaranteed safe "
                "high-occupancy solution.")
        elif method == "seed":
            chunk = _chunk_for(max_hits, max_pairs)
            info["path"] = "gpu_chunked"
            info["chunk_query"] = chunk
            info["largest_temp_mb"] = round(chunk * max_hits * 4 / 1e6, 3)
            info["seed_cap"] = max_seeds
            info["offline_test_feasible"] = True
            info["train_step_feasible"] = "measure"  # decide from EAF timing
            info["note"] = ("O(n*K) GPU, memory-bounded; candidate seeds capped "
                            "at max_seeds (explicit truncation -> seed_truncated). "
                            "Train-step feasibility to be confirmed by timing.")
        elif method == "greedy":
            info["path"] = "gpu_sequential"
            info["largest_temp_mb"] = round(max_hits * 4 / 1e6, 3)  # 1 x n row
            info["cluster_cap"] = max_clusters
            info["offline_test_feasible"] = True
            info["train_step_feasible"] = "no"
            info["note"] = ("bounded memory but a Python per-cluster loop with a "
                            "GPU->CPU sync each iteration -- oracle/offline only.")
        elif method == "graph":
            info["edge_cap"] = max_edges
            info["cc_max_iters"] = cc_max_iters
            info["offline_test_feasible"] = True
            if _HAS_TORCH_CLUSTER:
                info["path"] = "torch_cluster_sparse"
                info["efficient_path"] = True
                info["largest_temp_mb"] = "O(E)"
                info["train_step_feasible"] = "measure"  # decide from EAF timing
                info["note"] = ("sparse radius graph (fast). Train-step "
                                "feasibility to be confirmed by timing.")
            else:
                chunk = _chunk_for(max_hits, max_pairs)
                info["path"] = "dense_chunked"
                info["dense_fallback_only"] = True
                info["chunk_query"] = chunk
                info["largest_temp_mb"] = round(chunk * max_hits * 4 / 1e6, 3)
                info["train_step_feasible"] = "no"
                info["note"] = (
                    "EMERGENCY dense fallback: memory-safe (row-chunked) but "
                    "~n^2 distance evaluations (~44k^2 for the largest event) -- "
                    "for offline correctness/testing ONLY, NOT train-time. Install "
                    "torch_cluster for a train-feasible path.")
        report["backends"][method] = info
    return report
