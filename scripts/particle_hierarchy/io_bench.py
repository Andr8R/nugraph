#!/usr/bin/env python
"""io_bench.py -- where does NuGraph's input pipeline spend its time? (read-only diagnostic)

Stage-1b training runs at ~4.6 s/step of which only ~2.4 s is the model; the epoch-29 baseline did
0.79 s/step *in total* on the same 148 GB file with num_workers=0, already in its first epoch (so no
page-cache warm-up explains it). This measures the pipeline directly, separating the candidates:

  stages   per sample: raw HDF5 read (cold), the same read warm, NuGraphData.load conversion,
           NuGraph3.transform, and collation of batches -- in dataset order and in BalanceSampler
           order, so locality is visible. Bytes/s is reported from the actual record sizes.
  loader   end-to-end batches/s through a real DataLoader with `--workers 0 2 4 8`, using a
           lazy-open dataset (the HDF5 handle is opened inside each worker, which is what makes
           num_workers > 0 possible at all here). Semantics are unchanged: same order, same batching.
           Each worker count reads a DISJOINT block of samples by default, because a second pass over
           the same samples measures the page cache (warm reads are ~10x faster here), not the loader;
           --shared-samples restores the old, cache-contaminated behaviour on purpose.
  env      versions, cgroup CPU/memory limits, MemAvailable, h5py chunk cache, file size.

Every mode prints JSON too (--json). Nothing is written to the dataset; no training state is touched.

  io_bench.py env --data-path DATA.h5
  io_bench.py stages --data-path DATA.h5 --split train --n 192 --offset 0
  io_bench.py loader --data-path DATA.h5 --split train --n 384 --workers 0 4 --offset 200000
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_int(path):
    try:
        v = Path(path).read_text().split()[0]
        return None if v in ("max", "") else int(v)
    except Exception:
        return None


def meminfo(key="MemAvailable"):
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1]) * 1024
    except Exception:
        return None


def env_report(data_path=None):
    import h5py
    import numpy
    import torch
    import torch_geometric
    info = {
        "python": sys.version.split()[0], "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version, "numpy": numpy.__version__,
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "cgroup_memory_max_gb": (read_int("/sys/fs/cgroup/memory.max") or 0) / 1e9 or None,
        "cgroup_memory_current_gb": (read_int("/sys/fs/cgroup/memory.current") or 0) / 1e9 or None,
        "mem_available_gb": (meminfo() or 0) / 1e9 or None,
        "start_method": __import__("multiprocessing").get_start_method(allow_none=True),
    }
    if data_path:
        p = Path(data_path)
        info["file_gb"] = p.stat().st_size / 1e9
        with h5py.File(p) as f:
            info["h5_driver"] = f.driver
            info["h5_chunk_cache"] = f.id.get_access_plist().get_cache()
            info["n_train"] = len(f["samples/train"])
            info["n_test"] = len(f["samples/test"])
    return info


def sample_names(data_path, split, n, offset, order):
    """The sample keys to read, either in dataset order or in BalanceSampler order."""
    import h5py
    import numpy as np
    with h5py.File(data_path) as f:
        names = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in f[f"samples/{split}"].asstr()[()]]
        sizes = f["datasize/train"][()] if split == "train" and "datasize/train" in f else None
    if order == "sampler":
        if sizes is None:
            raise SystemExit("sampler order needs datasize/train (only for --split train)")
        from nugraph.data.BalanceSampler import BalanceSampler
        np.random.seed(0)
        idx = list(BalanceSampler(list(sizes), 32, 0.1).__iter__())
    else:
        idx = list(range(len(names)))
    idx = idx[offset:offset + n]
    return [names[i] for i in idx], len(names)


def cmd_stages(a):
    import h5py
    import nugraph as ng
    from pynuml.data import NuGraphData
    from torch_geometric.data import Batch

    names, total = sample_names(a.data_path, a.split, a.n, a.offset, a.order)
    transform = ng.models.NuGraph3.transform(planes=("u", "v", "y"))
    t = {k: [] for k in ("read_cold", "read_warm", "convert", "transform", "collate")}
    nbytes, samples = [], []
    with h5py.File(a.data_path) as f:
        for name in names:
            dset = f[f"/dataset/{name}"]
            t0 = time.perf_counter()
            raw = dset[()]                                  # cold: the actual record read
            t1 = time.perf_counter()
            raw2 = dset[()]                                 # warm: same bytes, page cache
            t2 = time.perf_counter()
            data = NuGraphData.load(dset)                   # read (warm) + tensor conversion
            t3 = time.perf_counter()
            data = transform(data)
            t4 = time.perf_counter()
            nbytes.append(int(raw.nbytes))
            t["read_cold"].append(t1 - t0)
            t["read_warm"].append(t2 - t1)
            t["convert"].append((t3 - t2) - (t2 - t1))      # conversion only
            t["transform"].append(t4 - t3)
            samples.append(data)
            del raw, raw2
    for i in range(0, len(samples) - 31, 32):
        t0 = time.perf_counter()
        Batch.from_data_list(samples[i:i + 32])
        t["collate"].append(time.perf_counter() - t0)
    q = lambda v: {"mean_ms": 1e3 * statistics.fmean(v), "median_ms": 1e3 * statistics.median(v),
                   "p90_ms": 1e3 * sorted(v)[int(0.9 * (len(v) - 1))], "n": len(v)}
    per_sample = {k: q(v) for k, v in t.items() if v}
    mb = statistics.fmean(nbytes) / 1e6
    cold = statistics.fmean(t["read_cold"])
    res = {"mode": "stages", "split": a.split, "order": a.order, "offset": a.offset,
           "n_samples": len(names), "split_size": total, "record_mb_mean": mb,
           "per_sample": per_sample,
           "cold_read_MBps": mb / cold if cold else None,
           "warm_read_MBps": mb / statistics.fmean(t["read_warm"]),
           "per_step_estimate_s": 32 * sum(statistics.fmean(t[k]) for k in
                                           ("read_cold", "convert", "transform"))
           + statistics.fmean(t["collate"]) if t["collate"] else None,
           "env": env_report(a.data_path)}
    print(json.dumps(res, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2, default=str))
    print(f"\nper sample: cold read {per_sample['read_cold']['median_ms']:.1f} ms "
          f"({res['cold_read_MBps']:.0f} MB/s) | warm read {per_sample['read_warm']['median_ms']:.2f} ms "
          f"| convert {per_sample['convert']['median_ms']:.2f} | transform "
          f"{per_sample['transform']['median_ms']:.2f} | record {mb:.2f} MB")
    print(f"=> implied loader cost for a 32-sample step: {res['per_step_estimate_s']:.2f} s "
          f"(Stage-1b observed 1.4-2.8 s; baseline implied <= 0.3 s)")


class LazyDataset:
    """NuGraphDataset that opens the HDF5 file lazily, so it can be pickled to DataLoader workers.

    Identical samples and order; only the handle's lifetime changes. This is the mechanism that makes
    num_workers > 0 usable on EAF (the current dataset holds an open h5py.File, which cannot be pickled).
    """

    def __init__(self, data_path, split, transform):
        import h5py
        self.data_path, self.split, self.transform = str(data_path), split, transform
        with h5py.File(self.data_path) as f:
            self.names = [s for s in f[f"samples/{split}"].asstr()[()]]
        self._file = None

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_file"] = None                                   # never pickle the handle
        return s

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        import h5py
        from pynuml.data import NuGraphData
        if self._file is None:
            self._file = h5py.File(self.data_path)          # one handle per worker process
        data = NuGraphData.load(self._file[f"/dataset/{self.names[i]}"])
        return self.transform(data) if self.transform else data


def cmd_loader(a):
    import nugraph as ng
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    nvar = len(a.workers)
    per = a.n if a.shared_samples else a.n // nvar
    if not a.shared_samples and per < 32:
        raise SystemExit(f"--n {a.n} over {nvar} worker counts gives {per} samples each; use --n >= {32*nvar}")
    transform = ng.models.NuGraph3.transform(planes=("u", "v", "y"))
    ds = LazyDataset(a.data_path, a.split, transform)
    lookup = {n: i for i, n in enumerate(ds.names)}
    out = {"mode": "loader", "split": a.split, "order": a.order, "offset": a.offset,
           "samples_per_variant": per, "shared_samples": a.shared_samples,
           "batch_size": 32, "variants": {}, "env": env_report(a.data_path)}
    for vi, w in enumerate(a.workers):
        off = a.offset if a.shared_samples else a.offset + vi * per
        names, total = sample_names(a.data_path, a.split, per, off, a.order)
        idx = [lookup[n] for n in names]                    # disjoint block per variant unless shared
        kw = dict(batch_size=32, num_workers=w, pin_memory=True, shuffle=False, drop_last=True)
        if w:
            kw.update(persistent_workers=True, prefetch_factor=a.prefetch)
        loader = DataLoader(Subset(ds, idx), **kw)
        it, times, t0 = iter(loader), [], time.perf_counter()
        first = None
        for batch in it:
            assert batch["hit"].num_nodes > 0               # touch the batch
            t1 = time.perf_counter()
            times.append(t1 - t0)
            first = first if first is not None else times[0]
            t0 = t1
        del it, loader
        steady = times[1:] or times
        out["variants"][f"workers={w}"] = {
            "offset": off, "cold": not a.shared_samples,
            "batches": len(times), "first_batch_s": first,
            "mean_s": statistics.fmean(steady), "median_s": statistics.median(steady),
            "total_s": sum(times)}
        print(f"  workers={w}: {statistics.median(steady):.3f} s/batch median "
              f"({statistics.fmean(steady):.3f} mean, first {first:.2f}, {len(times)} batches)", flush=True)
    if not a.shared_samples:
        print("  (each worker count read a disjoint, cold block of samples)")
    base = out["variants"].get("workers=0", {}).get("median_s")
    if base:
        for k, v in out["variants"].items():
            v["speedup_vs_workers0"] = base / v["median_s"] if v["median_s"] else None
    print(json.dumps(out, indent=2, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("env")
    s.add_argument("--data-path", default=None)
    s.add_argument("--json", default=None)
    s.set_defaults(fn=lambda a: (print(json.dumps(env_report(a.data_path), indent=2, default=str)),
                                 Path(a.json).write_text(json.dumps(env_report(a.data_path), indent=2,
                                                                    default=str)) if a.json else None))
    for name, fn in (("stages", cmd_stages), ("loader", cmd_loader)):
        s = sub.add_parser(name)
        s.add_argument("--data-path", required=True)
        s.add_argument("--split", default="train", choices=("train", "validation", "test"))
        s.add_argument("--n", type=int, default=192, help="samples to read (cold: use a fresh --offset)")
        s.add_argument("--offset", type=int, default=0,
                       help="skip this many samples first; use disjoint offsets to keep reads cold")
        s.add_argument("--order", default="sampler", choices=("sampler", "dataset"),
                       help="BalanceSampler order (as training) or on-disk order (locality test)")
        s.add_argument("--json", default=None)
        if name == "loader":
            s.add_argument("--workers", type=int, nargs="+", default=[0, 4])
            s.add_argument("--prefetch", type=int, default=4)
            s.add_argument("--shared-samples", action="store_true",
                           help="all worker counts read the SAME samples (warm after the first pass; "
                                "only for measuring cached throughput)")
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
