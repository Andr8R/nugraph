#!/usr/bin/env python
"""mat_bench.py -- benchmark instance-materialization backends on a cached NuGraph3 latent.

Stages (every backend sees byte-identical inputs; CPU and GPU work can run in separate
processes):

  preflight  environment report: torch/CUDA, torch_cluster, sklearn, imported nugraph
  cache      ONE forward pass per validation batch with a trained checkpoint; stores per-event
             latent coords (ox), beta, predicted foreground, truth, NuGraph's OWN DBSCAN labels
             and its OWN per-batch ARI (the parity reference). Needs a clean checkout whose
             model class matches the checkpoint.
  bench      run one or more backends on the cache: per-event clustering-only time (CUDA-synced,
             warmed up), PyTorch peak memory, MatStatus, quality metrics; OOM-tolerant, resumable
  report     numpy only: parity gate (benchmark DBSCAN == NuGraph DBSCAN) + per-config table
  geometry   numpy only: truth-based latent geometry + oracle-seed particle-ARI ceiling
  beta/sizes candidate-seed abundance vs truth / events at risk for the graph backend

Backends come from nugraph.models.nugraph3.materialize (sha256 recorded in every output).
Metrics: ``ari_particle`` (primary; ARI over predicted-foreground, labelled hits that belong to a
true particle), ``ari_nugraph`` (legacy NuGraph validation metric: torchmetrics ARI over hits
with y_semantic >= 0, -1 kept on both sides), purity/completeness, multiplicities.
See README.md in this directory for the full procedure.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

NAN = float("nan")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_range(s):
    lo, _, hi = s.partition(":")
    return int(lo or 0), (int(hi) if hi else None)


# --------------------------------------------------------------------------- #
def load_materializers():
    """Return (materialize module, provenance dict)."""
    import nugraph.models.nugraph3.materialize as mat
    prov = {"materialize_py": mat.__file__, "materialize_sha256": sha256(mat.__file__),
            "torch_cluster": mat.has_torch_cluster(),
            "nugraph_pkg": sys.modules["nugraph"].__file__,
            "pynuml_pkg": getattr(sys.modules.get("pynuml"), "__file__", None)}
    return mat, prov


# vendored verbatim from the original research benchmark (scripts/materialize_benchmark.py
# on the instance-materialization branch) so the metric definitions are unchanged
def event_quality(pred, true, y_semantic, keep_particle):
    """Two per-event ARIs (all tensors are event-local slices for one event).

    Args (all shape (n_hit_in_event,)):
        pred: predicted event-local cluster id per hit (-1 = background/noise/
            unmaterialized). This equals the training ``x_i`` up to a label
            permutation, so the adjusted Rand index is identical.
        true: truth instance id per hit (``y_i``; -1 = hit has no truth particle).
        y_semantic: truth semantic label per hit (-1 = unlabelled).
        keep_particle: foreground (predicted signal, non-diffuse) & y_semantic>=0.

    Returns a dict with:
        ari_particle  -- ARI over ``keep_particle & true>=0`` (real-particle
                         separation; excludes the -1 no-instance population);
        ari_inclusive -- ARI over ``y_semantic>=0`` INCLUDING -1 on both sides,
                         reproducing the original NuGraph validation mask;
        purity/completeness/v_measure/n_pred/n_true/noise_frac -- particle-scope
                         (computed on the ``ari_particle`` population).
    Any metric whose mask is empty (<2 hits) is returned as NaN so the event is
    still recorded and enters aggregates only where a finite value exists. The
    two ARIs answer different questions; neither is "better".
    """
    from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure
    out = dict(ari_particle=NAN, ari_inclusive=NAN, purity=NAN,
               completeness=NAN, v_measure=NAN, n_pred=NAN, n_true=NAN,
               noise_frac=NAN)

    # particle-only metric: predicted foreground signal hits that have a truth
    # particle. Predicted noise (-1) among these is kept so over-noise is penalised.
    mp = keep_particle & (true >= 0)
    if int(mp.sum()) >= 2:
        p = pred[mp].cpu().numpy()
        t = true[mp].cpu().numpy()
        out["ari_particle"] = adjusted_rand_score(t, p)
        homo, comp, vm = homogeneity_completeness_v_measure(t, p)
        out["purity"], out["completeness"], out["v_measure"] = homo, comp, vm
        out["n_pred"] = int(len({int(x) for x in p if x >= 0}))
        out["n_true"] = int(len(set(int(x) for x in t)))
    if int(keep_particle.sum()):
        out["noise_frac"] = float((pred[keep_particle] < 0).float().mean())

    # NuGraph-compatible inclusive metric: all semantically-labelled hits, with
    # -1 kept as a label on both sides (large no-instance population included).
    mi = y_semantic >= 0
    if int(mi.sum()) >= 2:
        out["ari_inclusive"] = adjusted_rand_score(
            true[mi].cpu().numpy(), pred[mi].cpu().numpy())
    return out


def foreground_mask(h):
    """Identical to InstanceDecoder's DBSCAN input mask: predicted signal, non-diffuse (class 6)."""
    import torch
    mask = torch.ones_like(h.of, dtype=torch.bool)
    if hasattr(h, "x_filter"):
        mask = mask & (h.x_filter > 0.5)
    if hasattr(h, "x_semantic"):
        mask = mask & (h.x_semantic.argmax(dim=1) != 6)
    return mask


# --------------------------------------------------------------------------- #
def cmd_preflight(a):
    import torch
    import sklearn
    rep = {"python": sys.version.split()[0], "torch": torch.__version__,
           "cuda": torch.cuda.is_available(), "sklearn": sklearn.__version__,
           "conda_env": os.environ.get("CONDA_DEFAULT_ENV")}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        rep.update(gpu=torch.cuda.get_device_name(0),
                   gpu_free_gb=round(free / 1e9, 2), gpu_total_gb=round(total / 1e9, 2))
    try:
        import nugraph
        rep["nugraph_pkg"] = nugraph.__file__
        _, prov = load_materializers()
        rep.update(prov)
    except Exception as e:  # report, don't hide
        rep["materializer_error"] = repr(e)
    print(json.dumps(rep, indent=2))


# --------------------------------------------------------------------------- #
def cmd_cache(a):
    import subprocess
    import torch
    import nugraph as ng

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(ng.__file__).resolve().parents[2]  # <repo>/nugraph/nugraph/__init__.py
    git = lambda *x: subprocess.run(["git", "-C", str(repo), *x],
                                    capture_output=True, text=True).stdout
    diff = git("diff", "HEAD")
    commit = git("rev-parse", "HEAD").strip()
    if diff and not a.allow_dirty:
        raise SystemExit(f"{repo} has uncommitted changes; commit them or pass --allow-dirty")

    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)
    model = ng.models.NuGraph3.load_from_checkpoint(a.checkpoint, map_location=dev).eval()
    hp = dict(model.hparams)
    nudata = ng.data.H5DataModule(a.data_path, batch_size=a.batch_size,
                                  model=ng.models.NuGraph3, shuffle="balance",
                                  num_workers=0, featext=False)
    lo, hi = parse_range(a.batches)
    manifest = {"checkpoint": a.checkpoint, "checkpoint_sha256": sha256(a.checkpoint),
                "repo": str(repo), "git_commit": commit, "tree_clean": not diff,
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "layernorm_blocks": sum(hasattr(m, "norm") for m in model.core_net.children()),
                "data_path": a.data_path, "split": "validation", "batch_size": a.batch_size,
                "batches": a.batches, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(dev),
                "hparams": {k: hp[k] for k in ("in_features", "coord_features",
                            "instance_features", "semantic_classes", "use_checkpointing")},
                "foreground": "x_semantic.argmax != 6 (no filter head)",
                "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    bfile = open(out / "batches.csv", "w", newline="")
    bw = csv.writer(bfile)
    bw.writerow(["batch", "n_graphs", "n_hits", "nugraph_ari", "forward_s"])
    shard, shard_id, t_all = [], 0, time.perf_counter()
    torch.cuda.reset_peak_memory_stats(dev)
    for bi, batch in enumerate(nudata.val_dataloader()):
        if bi < lo:
            continue
        if hi is not None and bi >= hi:
            break
        data = batch.to(dev)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            _, metrics = model(data, stage="test")   # also runs NuGraph's own DBSCAN
        torch.cuda.synchronize(dev)
        t_fwd = time.perf_counter() - t0
        h = data["hit"]
        mask = foreground_mask(h).cpu()
        ox = h.ox.detach().float().cpu()
        beta = h.of.detach().float().cpu()
        y_i = data.y_i().to(torch.int32).cpu()
        x_ng = data.x_i().to(torch.int32).cpu()
        ysem = h.y_semantic.to(torch.int8).cpu()
        ptr = h.ptr.cpu().tolist()
        for g in range(data.num_graphs):
            s = slice(ptr[g], ptr[g + 1])
            shard.append({"batch": bi, "event": g, "ox": ox[s].clone(),
                          "beta": beta[s].clone(), "mask": mask[s].clone(),
                          "y_i": y_i[s].clone(), "y_sem": ysem[s].clone(),
                          "x_nugraph": x_ng[s].clone()})
        bw.writerow([bi, data.num_graphs, h.num_nodes,
                     float(metrics["instance/adjusted-rand-test"]), f"{t_fwd:.6f}"])
        if len(shard) >= a.shard_events:
            torch.save(shard, out / f"shard_{shard_id:04d}.pt")
            shard, shard_id = [], shard_id + 1
            bfile.flush()
        if bi % 25 == 0:
            print(f"  cache batch {bi}  fwd {t_fwd:.2f}s  "
                  f"peak {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB", flush=True)
    if shard:
        torch.save(shard, out / f"shard_{shard_id:04d}.pt")
    bfile.close()
    manifest.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    wall_s=round(time.perf_counter() - t_all, 1),
                    forward_peak_alloc_gb=round(torch.cuda.max_memory_allocated(dev) / 1e9, 3))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
def iter_cache(cache, lo, hi):
    import torch
    for f in sorted(Path(cache).glob("shard_*.pt")):
        for ev in torch.load(f, weights_only=True):
            if ev["batch"] >= lo and (hi is None or ev["batch"] < hi):
                yield ev


def is_cuda_oom(exc):
    """CUDA OOM, including the MIG case where PyTorch's OOM path trips an NVML assert."""
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc)
    return isinstance(exc, RuntimeError) and ("NVML_SUCCESS == r" in msg
                                              or "out of memory" in msg.lower())


def radius_buffer_gb(n_fg):
    """torch_cluster CUDA radius(max_num_neighbors=n) preallocates 2 x n*n int64 (16 n^2 B)."""
    return 16.0 * n_fg * n_fg / 1e9


def cgroup_cpu_max():
    """Effective CPU quota (cgroup v2 'quota period'); os.cpu_count() ignores it."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if q == "max" else round(int(q) / int(p), 2)
    except Exception:
        return None


def write_rows(path, rows):
    keys = sorted({k for r in rows for k in r})
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def tag_for(backend, a):
    if backend == "dbscan":
        return f"dbscan_eps{a.dbscan_eps:g}_min{a.dbscan_min_samples}"
    t = f"{backend}_tb{a.t_beta:g}_td{a.t_d:g}"
    if backend == "seed" and a.t_assign is not None:
        t += f"_ta{a.t_assign:g}"
    return t + (f"_mcs{a.min_cluster_size}" if a.min_cluster_size > 1 else "")


def cmd_bench(a):
    import torch
    import sklearn
    from sklearn.cluster import DBSCAN
    from torchmetrics.functional.clustering import adjusted_rand_score as tm_ari

    mat, prov = load_materializers()
    cuda = a.device != "cpu" and torch.cuda.is_available()
    dev = torch.device(f"cuda:{a.device}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(dev)
    sync = (lambda: torch.cuda.synchronize(dev)) if cuda else (lambda: None)
    dbscan = DBSCAN(eps=a.dbscan_eps, min_samples=a.dbscan_min_samples)
    timer = mat.MaterializeTimer(enabled=a.profile, cuda=cuda)
    lo, hi = parse_range(a.batches)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    kw = dict(dbscan=dbscan, t_beta=a.t_beta, t_d=a.t_d, t_assign=a.t_assign,
              min_cluster_size=a.min_cluster_size, max_pairs=a.max_pairs,
              max_seeds=a.max_seeds, max_edges=a.max_edges,
              cc_max_iters=a.cc_max_iters, max_clusters=a.max_clusters, timer=timer)

    def run(backend, ev):
        ox = ev["ox"].to(dev)
        beta = ev["beta"].to(dev)
        mask = ev["mask"].to(dev)
        if cuda:
            sync()
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
        timer.reset()
        t0 = time.perf_counter()
        labels, stats = mat.materialize_labels(backend, ox, beta, mask, None, 1, **kw)
        sync()
        dt = time.perf_counter() - t0
        peak = (torch.cuda.max_memory_allocated(dev) - base) / 1e6 if cuda else NAN
        return labels.cpu(), stats[0], dt, peak

    for backend in a.backends:
        tag = tag_for(backend, a)
        csv_path, json_path = outdir / f"{tag}.csv", outdir / f"{tag}.json"
        rows, done, prior_wall = [], set(), 0.0
        if a.resume and csv_path.exists():
            rows = list(csv.DictReader(open(csv_path)))
            done = {(int(r["batch"]), int(r["event"])) for r in rows}
            if json_path.exists():
                prior_wall = json.loads(json_path.read_text()).get("wall_s", 0.0)
            print(f"  {tag}: resuming, {len(done)} events already recorded", flush=True)
        elif csv_path.exists() and not a.resume:
            raise SystemExit(f"{csv_path} exists; use --resume to continue it or remove it")

        n_warm = 0
        for ev in iter_cache(a.cache, lo, hi):  # untimed warm-up on the first events
            if n_warm >= a.warmup:
                break
            if radius_buffer_gb(int(ev["mask"].sum())) > 1.0:
                continue  # keep warm-up away from the large events
            run(backend, ev)
            n_warm += 1

        state = {"complete": False, "n_oom": 0}

        def save(wall):
            write_rows(csv_path, rows)
            meta = {"tag": tag, "backend": backend, "cache": a.cache, "batches": a.batches,
                    "n_events": len(rows), "complete": state["complete"],
                    "n_oom": sum(1 for r in rows if str(r.get("oom")) == "True"),
                    "warmup_events": n_warm, "wall_s": round(prior_wall + wall, 2),
                    "resumed": bool(done), "device": str(dev),
                    "gpu": torch.cuda.get_device_name(dev) if cuda else None,
                    "torch": torch.__version__, "sklearn": sklearn.__version__,
                    "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                    "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                    "cpu_threads": torch.get_num_threads(), "nproc": os.cpu_count(),
                    "cgroup_cpu_max": cgroup_cpu_max(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "profile_inner_syncs": a.profile,
                    "params": {k: getattr(a, k) for k in ("t_beta", "t_d", "t_assign", "min_cluster_size",
                               "dbscan_eps", "dbscan_min_samples", "max_pairs", "max_seeds",
                               "max_edges", "cc_max_iters", "max_clusters")}, **prov}
            json_path.write_text(json.dumps(meta, indent=2))

        t_start = time.perf_counter()
        n_new = 0
        try:
            for ev in iter_cache(a.cache, lo, hi):
                key = (int(ev["batch"]), int(ev["event"]))
                if key in done:
                    continue
                status_note, oom = "", False
                n_fg = int(ev["mask"].sum())
                try:
                    labels, st, dt, peak = run(backend, ev)
                except Exception as e:  # noqa: BLE001 -- only CUDA OOM is tolerated
                    if not (cuda and is_cuda_oom(e)):
                        raise
                    oom = True
                    torch.cuda.empty_cache()
                    try:  # the context must still be usable, else stop (resume later)
                        torch.ones(1024, device=dev).sum().item()
                        sync()
                    except Exception as e2:
                        raise SystemExit(f"CUDA context unusable after OOM at event {key}: "
                                         f"{e2!r}. Partial results saved; rerun with --resume.")
                    st = mat.MatStatus(method=backend, status="error", n_foreground=n_fg,
                                       safety_limit_reached=True)
                    labels, dt, peak = None, NAN, NAN
                    status_note = (f"CUDA OOM ({'NVML-masked' if 'NVML' in str(e) else 'OOM'}); "
                                   f"radius buffer est {radius_buffer_gb(n_fg):.1f} GB")
                    print(f"  {tag}: OOM at batch {key[0]} event {key[1]} "
                          f"(n_fg={n_fg}, est {radius_buffer_gb(n_fg):.1f} GB) -> recorded, continuing",
                          flush=True)
                y_i = ev["y_i"].long()
                ysem = ev["y_sem"].long()
                mask = ev["mask"]
                row = st.as_row()  # first: its "event" is the call-local index (always 0)
                row.update({"tag": tag, "backend": backend, "batch": ev["batch"],
                            "event": ev["event"], "n_hits": int(y_i.numel()),
                            "t_cluster_s": dt, "peak_alloc_mb": peak, "oom": oom,
                            "radius_buffer_est_gb": radius_buffer_gb(n_fg),
                            "n_true_all": int(torch.unique(y_i[y_i >= 0]).numel())})
                if status_note:
                    row["note"] = status_note
                if a.profile:
                    row.update({f"t_{k.replace('/', '_')}_s": v for k, v in timer.times.items()})
                if cuda:  # reference cost of the D2H/H2D hop DBSCAN needs in the model
                    sync()
                    t0 = time.perf_counter()
                    arr = ev["ox"].to(dev)[mask.to(dev)].cpu()
                    torch.zeros(arr.shape[0], dtype=torch.long).to(dev)
                    sync()
                    row["t_transfer_ref_s"] = time.perf_counter() - t0
                if labels is not None and (labels >= 0).any():  # untimed size diagnostics
                    sizes = torch.bincount(labels[labels >= 0])
                    sizes = sizes[sizes > 0]
                    fg = max(int(mask.sum()), 1)
                    for k in (2, 5, 15):
                        row[f"n_clusters_ge{k}"] = int((sizes >= k).sum())
                        row[f"frac_fg_in_clusters_lt{k}"] = float(sizes[sizes < k].sum()) / fg
                m = ysem >= 0
                if labels is not None and st.valid:
                    if int(m.sum()):
                        row["ari_nugraph"] = float(tm_ari(labels[m], y_i[m]))
                    row.update(event_quality(labels, y_i, ysem, mask & m))
                if backend == "dbscan" and labels is not None:  # label-level parity
                    xng = ev["x_nugraph"].long()
                    row["parity_ari_vs_nugraph_labels"] = float(tm_ari(labels, xng))
                    row["parity_n_clusters_nugraph"] = int(torch.unique(xng[xng >= 0]).numel())
                rows.append(row)
                n_new += 1
                if n_new % 500 == 1:
                    print(f"  {tag}: new event {n_new}  (batch {key[0]})  t={dt:.4f}s  "
                          f"status={st.status}", flush=True)
                if n_new % 250 == 0:
                    save(time.perf_counter() - t_start)
            state["complete"] = True
        finally:
            save(time.perf_counter() - t_start)
            print(f"{'wrote' if state['complete'] else 'PARTIAL'} {csv_path}  "
                  f"({len(rows)} events, {n_new} new)", flush=True)


# --------------------------------------------------------------------------- #
def cmd_report(a):
    import numpy as np

    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return NAN

    def q(x, p):
        x = np.asarray([v for v in x if v == v], dtype=float)
        return float(np.percentile(x, p)) if x.size else NAN

    def mean(x):
        x = np.asarray([v for v in x if v == v], dtype=float)
        return float(x.mean()) if x.size else NAN

    bench = Path(a.bench)
    tables = {}
    for p in sorted(bench.glob("*.csv")):
        if p.name.startswith("report"):
            continue
        rows = list(csv.DictReader(open(p)))
        if a.batches:
            lo, hi = parse_range(a.batches)
            rows = [r for r in rows if int(r["batch"]) >= lo and (hi is None or int(r["batch"]) < hi)]
        tables[p.stem] = rows
    base_tag = a.baseline
    if base_tag not in tables:
        raise SystemExit(f"baseline {base_tag}.csv not found in {bench}")
    base = {(r["batch"], r["event"]): r for r in tables[base_tag]}

    lines = ["# Materialization benchmark report", ""]
    # ---------------- parity gate ----------------
    ref = {r["batch"]: r for r in csv.DictReader(open(Path(a.cache) / "batches.csv"))}
    lines[0] += f" (batches {a.batches})" if a.batches else " (all cached batches)"
    per_batch = {}
    for r in tables[base_tag]:
        per_batch.setdefault(r["batch"], []).append(f(r.get("ari_nugraph")))
    diffs = [abs(mean(v) - f(ref[b]["nugraph_ari"])) for b, v in per_batch.items() if b in ref]
    n_nan = sum(1 for d in diffs if d != d)
    diffs = [d for d in diffs if d == d] or [NAN]
    lab = [f(r.get("parity_ari_vs_nugraph_labels")) for r in tables[base_tag]]
    lab = [v for v in lab if v == v] or [NAN]
    ev_mean = mean([f(r.get("ari_nugraph")) for r in tables[base_tag]])
    ref_mean = (sum(f(ref[b]["nugraph_ari"]) * int(ref[b]["n_graphs"]) for b in per_batch)
                / sum(int(ref[b]["n_graphs"]) for b in per_batch))
    ok_batch = max(diffs) < a.tol
    ok_label = min(lab) > 1 - 1e-6
    lines += ["## Parity gate (benchmark DBSCAN vs NuGraph's own DBSCAN, same forward)",
              f"- batches compared: {len(diffs)}; max |batch-mean ARI diff| = {max(diffs):.2e} "
              f"(tol {a.tol:g}; {n_nan} NaN batches) -> {'PASS' if ok_batch else 'FAIL'}",
              f"- label-level: min ARI(bench labels, NuGraph labels) = {min(lab):.8f} "
              f"-> {'PASS' if ok_label else 'FAIL'}",
              f"- event-mean ari_nugraph = {ev_mean:.6f}; NuGraph weighted batch mean = "
              f"{ref_mean:.6f}" + (f"; validate_ckpt reference = {a.ref:.6f} "
              f"(|diff| {abs(ev_mean - a.ref):.1e})" if a.ref else ""), ""]
    if not (ok_batch and ok_label):
        lines += ["**PARITY FAILED -- do not trust the comparison below.**", ""]

    # ---------------- per-configuration table ----------------
    out_rows = []
    for tag, rows in tables.items():
        n = len(rows)
        valid = [r for r in rows if r.get("valid") == "True"]
        inval = [r for r in rows if r.get("valid") != "True"]
        a_valid = [f(r.get("ari_nugraph")) for r in valid]
        b_matched = [f(base[(r["batch"], r["event"])].get("ari_nugraph")) for r in valid]
        composite = [f(r.get("ari_nugraph")) if r.get("valid") == "True"
                     else f(base[(r["batch"], r["event"])].get("ari_nugraph")) for r in rows]
        t_ev = [f(r["t_cluster_s"]) for r in rows]
        timed = [r for r in rows if f(r["t_cluster_s"]) == f(r["t_cluster_s"])]
        by_batch = {}
        for r in timed:
            by_batch[r["batch"]] = by_batch.get(r["batch"], 0.0) + f(r["t_cluster_s"])
        t_tot = sum(v for v in t_ev if v == v)
        bt = lambda rs: sum(f(base[(r["batch"], r["event"])]["t_cluster_s"]) for r in rs)
        t_base = bt(timed)          # raw: only events this configuration actually timed
        t_base_all = bt(rows)       # with fallback: every event
        t_fb = t_tot + bt(inval)    # invalid (edge_limit / OOM) events re-run by the baseline
        npred = [f(r.get("n_pred")) for r in valid]
        ntrue = [f(r.get("n_true")) for r in valid]
        ratio = [p / t for p, t in zip(npred, ntrue) if t == t and t > 0 and p == p]
        status = {}
        for r in rows:
            status[r["status"]] = status.get(r["status"], 0) + 1
        o = {"tag": tag, "n_events": n, "n_valid": len(valid),
             "invalid_rate": len(inval) / n if n else NAN, "status": status,
             "ari_nugraph_mean_valid": mean(a_valid),
             "ari_nugraph_median_valid": q(a_valid, 50),
             "ari_nugraph_p10_valid": q(a_valid, 10), "ari_nugraph_p90_valid": q(a_valid, 90),
             "dbscan_ari_same_events": mean(b_matched),
             "ari_nugraph_mean_with_dbscan_fallback": mean(composite),
             "ari_particle_mean_valid": mean([f(r.get("ari_particle")) for r in valid]),
             "purity_median": q([f(r.get("purity")) for r in valid], 50),
             "completeness_median": q([f(r.get("completeness")) for r in valid], 50),
             "n_true_median": q(ntrue, 50), "n_pred_median": q(npred, 50),
             "n_pred_p90": q(npred, 90), "n_clusters_median": q([f(r["n_clusters"]) for r in valid], 50),
             "pred_over_true_median": q(ratio, 50),
             "frac_over_seg": mean([float(x > 1.5) for x in ratio]),
             "frac_under_seg": mean([float(x < 1 / 1.5) for x in ratio]),
             "noise_frac_median": q([f(r.get("noise_frac")) for r in valid], 50),
             "t_event_mean_ms": 1e3 * mean(t_ev), "t_event_median_ms": 1e3 * q(t_ev, 50),
             "t_event_p95_ms": 1e3 * q(t_ev, 95),
             "t_batch_median_ms": 1e3 * q(list(by_batch.values()), 50),
             "t_batch_p95_ms": 1e3 * q(list(by_batch.values()), 95),
             "t_total_s": t_tot, "speedup_total_vs_dbscan": t_base / t_tot if t_tot else NAN,
             "t_total_with_fallback_s": t_fb,
             "speedup_with_fallback": t_base_all / t_fb if t_fb else NAN,
             "n_oom": sum(1 for r in rows if r.get("oom") == "True"),
             "n_untimed": len(rows) - len(timed),
             "peak_alloc_mb_max": max([f(r["peak_alloc_mb"]) for r in rows if f(r["peak_alloc_mb"]) == f(r["peak_alloc_mb"])], default=NAN),
             "peak_alloc_mb_p95": q([f(r["peak_alloc_mb"]) for r in rows], 95),
             "seed_candidates_median": q([f(r["n_candidate_seeds"]) for r in rows if f(r["n_candidate_seeds"]) >= 0], 50),
             "seed_surviving_median": q([f(r["n_surviving_seeds"]) for r in rows if f(r["n_surviving_seeds"]) >= 0], 50),
             "graph_edges_median": q([f(r["n_edges"]) for r in rows if f(r["n_edges"]) >= 0], 50),
             "graph_edges_max": max([f(r["n_edges"]) for r in rows], default=NAN),
             "graph_paths": sorted({r.get("graph_path") for r in rows if r.get("graph_path")}),
             "transfer_ref_median_ms": 1e3 * q([f(r.get("t_transfer_ref_s")) for r in rows], 50),
             "n_foreground_median_valid": q([f(r["n_foreground"]) for r in valid], 50),
             "n_foreground_median_invalid": q([f(r["n_foreground"]) for r in inval], 50),
             "n_clusters_ge15_median": q([f(r.get("n_clusters_ge15")) for r in valid], 50),
             "frac_fg_in_clusters_lt15_median": q([f(r.get("frac_fg_in_clusters_lt15")) for r in valid], 50),
             "t_base_same_events_s": t_base}
        out_rows.append(o)

    cols = [("tag", "{}"), ("n_valid", "{}"), ("invalid_rate", "{:.3f}"),
            ("ari_nugraph_mean_valid", "{:.4f}"), ("dbscan_ari_same_events", "{:.4f}"),
            ("ari_nugraph_mean_with_dbscan_fallback", "{:.4f}"),
            ("ari_particle_mean_valid", "{:.4f}"), ("purity_median", "{:.3f}"),
            ("completeness_median", "{:.3f}"), ("noise_frac_median", "{:.3f}"),
            ("n_true_median", "{:.0f}"),
            ("n_pred_median", "{:.0f}"), ("pred_over_true_median", "{:.2f}"),
            ("frac_over_seg", "{:.2f}"), ("frac_under_seg", "{:.2f}"),
            ("t_event_median_ms", "{:.2f}"), ("t_event_p95_ms", "{:.2f}"),
            ("t_batch_median_ms", "{:.1f}"), ("speedup_total_vs_dbscan", "{:.1f}"),
            ("speedup_with_fallback", "{:.1f}"), ("peak_alloc_mb_max", "{:.0f}")]
    lines += ["## Per configuration (identical cached events)", "",
              "| " + " | ".join(c for c, _ in cols) + " |",
              "|" + "---|" * len(cols)]
    for o in out_rows:
        lines.append("| " + " | ".join(fmt.format(o[c]) if o[c] == o[c] else "nan"
                                       for c, fmt in cols) + " |")
    lines += ["", "## Health / caps", ""]
    for o in out_rows:
        lines.append(f"- **{o['tag']}**: status={o['status']}; OOM={o['n_oom']} "
                     f"(untimed {o['n_untimed']}); seed cand/surv median="
                     f"{o['seed_candidates_median']:.0f}/{o['seed_surviving_median']:.0f}; "
                     f"graph edges median/max={o['graph_edges_median']:.0f}/"
                     f"{o['graph_edges_max']:.0f} path={o['graph_paths']}; "
                     f"transfer-ref median={o['transfer_ref_median_ms']:.2f} ms; "
                     f"n_foreground median valid/invalid={o['n_foreground_median_valid']:.0f}/"
                     f"{o['n_foreground_median_invalid']:.0f}; clusters>=15 median="
                     f"{o['n_clusters_ge15_median']:.0f}; fg frac in clusters<15 median="
                     f"{o['frac_fg_in_clusters_lt15_median']:.3f}; totals: {o['t_total_s']:.1f} s vs "
                     f"baseline {o['t_base_same_events_s']:.1f} s")
    lines += ["", "Definitions: ARI over valid events only; `dbscan_ari_same_events` = baseline on "
              "exactly those events; `with_dbscan_fallback` = invalid events scored/timed with "
              "the baseline. over/under-seg: n_pred/n_true >1.5 or <1/1.5 (particle-scope counts). "
              "Raw speedup = baseline / configuration clustering-only time over the events the "
              "configuration timed (OOM events have no time); speedup with fallback = baseline over "
              "ALL events / (configuration time + baseline time of its invalid events)."]
    text = "\n".join(lines)
    (bench / "report.md").write_text(text)
    with open(bench / "report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(text)


# --------------------------------------------------------------------------- #
def cmd_sizes(a):
    """Events ordered as the bench visits them, with the graph radius-buffer estimate."""
    lo, hi = parse_range(a.batches)
    evs = []
    for i, ev in enumerate(iter_cache(a.cache, lo, hi)):
        n_fg = int(ev["mask"].sum())
        evs.append((i, int(ev["batch"]), int(ev["event"]), int(ev["mask"].numel()), n_fg,
                    radius_buffer_gb(n_fg)))
    big = [e for e in evs if e[5] > a.gb]
    print(f"{len(evs)} events; {len(big)} with estimated radius buffer > {a.gb} GB "
          f"(16*n_fg^2; observed peak = 1.15-2.2x this)")
    print(f"{'order':>6} {'batch':>6} {'event':>6} {'n_hits':>7} {'n_fg':>7} {'buf_GB':>7}")
    for e in big[:a.show]:
        print(f"{e[0]:>6} {e[1]:>6} {e[2]:>6} {e[3]:>7} {e[4]:>7} {e[5]:>7.1f}")
    top = sorted(evs, key=lambda e: -e[4])[:5]
    print("largest:", ", ".join(f"b{e[1]}e{e[2]} n_fg={e[4]} ({e[5]:.1f} GB)" for e in top))


def ari_np(a, b):
    """Adjusted Rand index (numpy; same definition and special cases as sklearn)."""
    import numpy as np
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    n = a.size
    if n < 2:
        return float("nan")
    cont = np.zeros((a.max() + 1, b.max() + 1), dtype=np.float64)
    np.add.at(cont, (a, b), 1.0)
    c2 = lambda x: x * (x - 1.0) / 2.0
    s_ij, s_a, s_b = c2(cont).sum(), c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum()
    expected = s_a * s_b / c2(float(n))
    maximum = (s_a + s_b) / 2.0
    if maximum == expected:
        return 1.0
    return float((s_ij - expected) / (maximum - expected))


def _nearest(xa, xb, chunk=4096):
    """For each row of xa: (index, distance) of the nearest row of xb (numpy, chunked)."""
    import numpy as np
    idx = np.empty(len(xa), dtype=np.int64)
    dist = np.empty(len(xa), dtype=np.float64)
    for i in range(0, len(xa), chunk):
        d = np.sqrt(((xa[i:i + chunk, None, :] - xb[None, :, :]) ** 2).sum(-1))
        idx[i:i + chunk] = d.argmin(1)
        dist[i:i + chunk] = d.min(1)
    return idx, dist


TD_GRID = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf")]
TB_GRID = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]


def geometry_event(ox, beta, fg, y_i, ysem, min_hits=5):
    """Truth-based latent geometry of one event on the ari_particle population.

    Condensation point (CP) of a true particle = its foreground hit with the highest beta.
    Returns per-particle rows (n_hits, beta_cp, r50, r90, nearest-other-CP distance),
    the beta values of non-CP population hits, and the oracle-seed particle ARI for each
    t_d in TD_GRID (every population hit -> nearest TRUE CP if within t_d, else noise).
    """
    import numpy as np
    pop = np.nonzero(fg & (ysem >= 0) & (y_i >= 0))[0]
    if pop.size < 2:
        return None
    parts = y_i[pop]
    uniq = np.unique(parts)
    cps = np.array([pop[parts == p][np.argmax(beta[pop[parts == p]])] for p in uniq])
    C = ox[cps]
    if len(cps) > 1:
        dcc = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dcc, np.inf)
        nn_cp = dcc.min(1)
    else:
        nn_cp = np.full(1, np.inf)
    rows = []
    for k, p in enumerate(uniq):
        h = pop[parts == p]
        if h.size < min_hits:
            continue
        d = np.sqrt(((ox[h] - ox[cps[k]]) ** 2).sum(-1))
        rows.append((h.size, float(beta[cps[k]]), float(np.median(d)),
                     float(np.percentile(d, 90)), float(nn_cp[k])))
    noncp = np.setdiff1d(pop, cps)
    near, dmin = _nearest(ox[pop], C)
    oracle = {}
    for td in TD_GRID:
        lab = np.where(dmin <= td, uniq[near], -1)
        oracle[td] = ari_np(parts, lab)
    return {"particles": rows, "beta_noncp": beta[noncp], "oracle": oracle,
            "n_true": int(uniq.size)}


def cmd_geometry(a):
    import numpy as np
    lo, hi = parse_range(a.batches)
    parts, bnon, orc, checked = [], [], {td: [] for td in TD_GRID}, 0
    try:
        from sklearn.metrics import adjusted_rand_score as sk_ari
    except Exception:
        sk_ari = None
    for ev in iter_cache(a.cache, lo, hi):
        ox = ev["ox"].numpy().astype(np.float64)
        g = geometry_event(ox, ev["beta"].numpy().astype(np.float64), ev["mask"].numpy(),
                           ev["y_i"].numpy().astype(np.int64),
                           ev["y_sem"].numpy().astype(np.int64), a.min_hits)
        if g is None:
            continue
        if sk_ari is not None and checked < 20:  # validate ari_np against sklearn
            y = ev["y_i"].numpy().astype(np.int64)
            lab = np.random.default_rng(checked).integers(-1, 5, y.size)
            assert abs(ari_np(y, lab) - sk_ari(y, lab)) < 1e-9, "ari_np != sklearn"
            checked += 1
        parts += g["particles"]
        if len(bnon) < 2_000_000:
            bnon.append(g["beta_noncp"])
        for td, v in g["oracle"].items():
            orc[td].append(v)
    P = np.asarray(parts, dtype=float)
    bn = np.concatenate(bnon)
    q = lambda x, p: float(np.percentile(x[np.isfinite(x)], p))
    print(f"particles with >= {a.min_hits} population hits: {len(P)}  "
          f"(ari_np validated vs sklearn on {checked} events)")
    print(f"  hits/particle           p50 {q(P[:,0],50):8.0f}  p90 {q(P[:,0],90):8.0f}")
    print(f"  r50 (hit->own CP)       p10 {q(P[:,2],10):8.3f}  p50 {q(P[:,2],50):8.3f}  p90 {q(P[:,2],90):8.3f}")
    print(f"  r90 (hit->own CP)       p10 {q(P[:,3],10):8.3f}  p50 {q(P[:,3],50):8.3f}  p90 {q(P[:,3],90):8.3f}")
    print(f"  nearest other true CP   p10 {q(P[:,4],10):8.3f}  p50 {q(P[:,4],50):8.3f}  p90 {q(P[:,4],90):8.3f}")
    sep = P[:, 3] < P[:, 4] / 2
    print(f"  separable (r90 < half nearest-CP distance): {sep.mean():.3f} of particles")
    print(f"  beta at true CP         p10 {q(P[:,1],10):8.3f}  p50 {q(P[:,1],50):8.3f}  p90 {q(P[:,1],90):8.3f}")
    print(f"  beta of other pop hits  p50 {q(bn,50):8.3f}  p90 {q(bn,90):8.3f}  p99 {q(bn,99):8.3f}")
    print("  t_beta: frac true CPs above | frac other pop hits above")
    for t in TB_GRID:
        print(f"    {t:>4g}: {(P[:,1] > t).mean():.3f} | {(bn > t).mean():.4f}")
    print("  oracle-seed particle ARI (true CPs, assign within t_d):")
    for td in TD_GRID:
        v = np.asarray(orc[td], dtype=float)
        print(f"    t_d {td:>5g}: mean {np.nanmean(v):.4f}  median {np.nanmedian(v):.4f}")


BETA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]


def cmd_beta(a):
    """Candidate-seed abundance vs truth: foreground hits with beta > t, per event."""
    import numpy as np
    import torch
    lo, hi = parse_range(a.batches)
    per_t = {t: [] for t in BETA_GRID}
    ntrue, betas_true_max = [], []
    for ev in iter_cache(a.cache, lo, hi):
        b = ev["beta"][ev["mask"]]
        y = ev["y_i"].long()
        nt = int(torch.unique(y[y >= 0]).numel())
        if nt == 0 or b.numel() == 0:
            continue
        ntrue.append(nt)
        for t in BETA_GRID:
            per_t[t].append(int((b > t).sum()))
    ntrue = np.asarray(ntrue, dtype=float)
    print(f"events with truth + foreground: {len(ntrue)}; n_true_all median {np.median(ntrue):.0f}")
    print(f"{'t_beta':>7} {'cand_median':>12} {'cand_p90':>9} {'median cand/n_true':>19} {'frac events cand<n_true':>24}")
    rows = []
    for t in BETA_GRID:
        c = np.asarray(per_t[t], dtype=float)
        r = dict(t_beta=t, cand_median=float(np.median(c)), cand_p90=float(np.percentile(c, 90)),
                 ratio_median=float(np.median(c / ntrue)), frac_below_truth=float((c < ntrue).mean()))
        rows.append(r)
        print(f"{t:>7g} {r['cand_median']:>12.0f} {r['cand_p90']:>9.0f} "
              f"{r['ratio_median']:>19.2f} {r['frac_below_truth']:>24.2f}")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("preflight")
    s.set_defaults(fn=cmd_preflight)

    s = sub.add_parser("cache")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--data-path", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", type=int, default=0)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--batches", default="0:", help="validation batch range lo:hi")
    s.add_argument("--shard-events", type=int, default=400)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=cmd_cache)

    s = sub.add_parser("bench")
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--backends", nargs="+", required=True,
                   choices=("dbscan", "seed", "graph", "greedy"))
    s.add_argument("--device", default="0", help="GPU index or 'cpu'")
    s.add_argument("--batches", default="0:")
    s.add_argument("--warmup", type=int, default=8, help="untimed warm-up events")
    s.add_argument("--resume", action="store_true",
                   help="continue <tag>.csv in --out: skip events already recorded")
    s.add_argument("--profile", action="store_true",
                   help="per-section timers (adds inner CUDA syncs; not for headline timing)")
    # defaults == the original research benchmark's defaults (Stage A 'no retuning')
    s.add_argument("--t-beta", type=float, default=0.1)
    s.add_argument("--t-d", type=float, default=0.3)
    s.add_argument("--t-assign", type=float, default=None,
                   help="seed hit-assignment radius (default: = --t-d, original behaviour; 'inf' ok)")
    s.add_argument("--min-cluster-size", type=int, default=1)
    s.add_argument("--dbscan-eps", type=float, default=0.3)
    s.add_argument("--dbscan-min-samples", type=int, default=15)
    s.add_argument("--max-pairs", type=int, default=4_000_000)
    s.add_argument("--max-seeds", type=int, default=20_000)
    s.add_argument("--max-edges", type=int, default=8_000_000)
    s.add_argument("--cc-max-iters", type=int, default=200)
    s.add_argument("--max-clusters", type=int, default=20_000)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("report")
    s.add_argument("--bench", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--baseline", default="dbscan_eps0.3_min15")
    s.add_argument("--ref", type=float, default=None, help="validate_ckpt ARI to compare")
    s.add_argument("--tol", type=float, default=1e-5)
    s.add_argument("--batches", default=None, help="restrict every table to batches lo:hi")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("sizes")
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:")
    s.add_argument("--gb", type=float, default=8.0)
    s.add_argument("--show", type=int, default=40)
    s.set_defaults(fn=cmd_sizes)

    s = sub.add_parser("geometry")
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:")
    s.add_argument("--min-hits", type=int, default=5)
    s.set_defaults(fn=cmd_geometry)

    s = sub.add_parser("beta")
    s.add_argument("--cache", required=True)
    s.add_argument("--batches", default="0:")
    s.add_argument("--out", default=None, help="optional CSV path")
    s.set_defaults(fn=cmd_beta)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
