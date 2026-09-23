"""Tests for the input-pipeline diagnostic (scripts/particle_hierarchy/io_bench.py).

The interesting parts are the sample ordering (must match training) and the lazy-open dataset, which is
the mechanism that would let DataLoader workers be used at all. Needs h5py (EAF), so these skip on GPVM.
"""
import importlib.util
import json
import sys
import pickle
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")

ROOT = Path(__file__).resolve().parents[1]
# Load io_bench once per session and share it. It must be in sys.modules for pickle to resolve the
# classes defined in it (DataLoader workers pickle the dataset), and registering two *different*
# module objects under the same name breaks pickle's identity check, so reuse whatever is there.
if "io_bench" in sys.modules:
    ib = sys.modules["io_bench"]
else:
    _spec = importlib.util.spec_from_file_location("io_bench", ROOT / "scripts/particle_hierarchy/io_bench.py")
    ib = importlib.util.module_from_spec(_spec)
    sys.modules["io_bench"] = ib
    _spec.loader.exec_module(ib)



class _FakeLoader:
    """Yields objects that quack like a PyG batch (`batch["hit"].num_nodes`)."""

    def __init__(self, *a, batches=2, **k):
        self.batches = batches

    def __iter__(self):
        hit = type("store", (), {"num_nodes": 7})()
        return iter([{"hit": hit} for _ in range(self.batches)])


def tiny_file(path, n=40, hits=5):
    """A miniature file in the real layout: one scalar compound dataset per event."""
    with h5py.File(path, "w") as f:
        names = [f"evt{i:03d}" for i in range(n)]
        for i, name in enumerate(names):
            dt = np.dtype([("hit/x", np.float32, (hits, 3)), ("hit/y_semantic", np.int8, (hits,)),
                           ("evt/y", np.int64)])
            rec = (np.full((hits, 3), float(i), np.float32), np.zeros(hits, np.int8), i)
            f.create_dataset(f"/dataset/{name}", shape=(), dtype=dt, data=rec)
        f.create_dataset("samples/train", data=np.array(names, dtype="S16"))
        f.create_dataset("samples/test", data=np.array(names[:4], dtype="S16"))
        f.create_dataset("datasize/train", data=np.arange(1, n + 1))
    return path


def test_dataset_order_is_the_on_disk_order(tmp_path):
    p = tiny_file(tmp_path / "t.h5")
    names, total = ib.sample_names(p, "train", n=5, offset=2, order="dataset")
    assert total == 40 and names == [f"evt{i:03d}" for i in (2, 3, 4, 5, 6)]


def test_sampler_order_matches_the_training_sampler(tmp_path):
    p = tiny_file(tmp_path / "t.h5")
    pytest.importorskip("nugraph")
    from nugraph.data.BalanceSampler import BalanceSampler
    names, _ = ib.sample_names(p, "train", n=8, offset=0, order="sampler")
    np.random.seed(0)
    want = list(BalanceSampler(list(np.arange(1, 41)), 32, 0.1).__iter__())[:8]
    assert names == [f"evt{i:03d}" for i in want]
    assert names != [f"evt{i:03d}" for i in range(8)]          # and it is not the on-disk order


def test_sampler_order_needs_the_datasize_table(tmp_path):
    p = tiny_file(tmp_path / "t.h5")
    with pytest.raises(SystemExit):
        ib.sample_names(p, "test", n=2, offset=0, order="sampler")


def test_lazy_dataset_pickles_and_reopens_per_process(tmp_path):
    p = tiny_file(tmp_path / "t.h5")
    ds = ib.LazyDataset(p, "train", transform=None)
    assert len(ds) == 40
    first = ds[3]                                              # opens the handle in this process
    assert ds._file is not None
    assert pickle.loads(pickle.dumps(ds))._file is None        # the handle is never pickled
    revived = pickle.loads(pickle.dumps(ds))                   # as a worker would receive it
    again = revived[3]
    assert revived._file is not None                           # reopened lazily on first access
    assert np.allclose(first["hit"].x.numpy(), again["hit"].x.numpy())
    assert float(again["hit"].x[0, 0]) == 3.0                  # the right event


def test_lazy_dataset_applies_the_transform(tmp_path):
    p = tiny_file(tmp_path / "t.h5")
    seen = []

    def transform(d):
        seen.append(d)
        return "transformed"

    ds = ib.LazyDataset(p, "train", transform=transform)
    assert ds[0] == "transformed" and len(seen) == 1


def test_env_report_records_limits_and_versions(tmp_path):
    p = tiny_file(tmp_path / "t.h5")
    info = ib.env_report(p)
    for k in ("python", "h5py", "numpy", "torch", "cpu_affinity", "file_gb", "n_train"):
        assert k in info, k
    assert info["n_train"] == 40 and info["file_gb"] > 0


def test_loader_variants_use_disjoint_blocks_by_default(tmp_path, monkeypatch):
    """Each worker count must read its own cold samples; a shared block measures the page cache."""
    p = tiny_file(tmp_path / "t.h5", n=200)
    seen = []

    def fake_names(data_path, split, n, offset, order):
        seen.append((n, offset))
        return [f"evt{i:03d}" for i in range(offset, offset + n)], 200
    monkeypatch.setattr(ib, "sample_names", fake_names)
    monkeypatch.setattr(ib, "env_report", lambda *a, **k: {})

    monkeypatch.setitem(sys.modules, "torch_geometric.loader",
                        type("m", (), {"DataLoader": _FakeLoader}))
    a = type("A", (), {"data_path": str(p), "split": "train", "n": 192, "offset": 0,
                       "order": "dataset", "workers": [0, 4, 8], "prefetch": 4,
                       "shared_samples": False, "json": None})()
    ib.cmd_loader(a)
    assert seen == [(64, 0), (64, 64), (64, 128)]              # disjoint, 192/3 each
    seen.clear()
    a.shared_samples = True
    ib.cmd_loader(a)
    assert seen == [(192, 0)] * 3                              # same block, warm after the first


def test_loader_reports_a_variant_that_yields_no_batches(tmp_path, monkeypatch, capsys):
    """A loader that produces nothing must be recorded, not crash in the summary statistics."""
    p = tiny_file(tmp_path / "t.h5", n=200)
    monkeypatch.setattr(ib, "sample_names", lambda *a, **k: ([f"evt{i:03d}" for i in range(64)], 200))
    monkeypatch.setattr(ib, "env_report", lambda *a, **k: {})
    monkeypatch.setitem(sys.modules, "torch_geometric.loader",
                        type("m", (), {"DataLoader": lambda *a, **k: _FakeLoader(batches=0)}))
    out = tmp_path / "r.json"
    a = type("A", (), {"data_path": str(p), "split": "train", "n": 64, "offset": 0,
                       "order": "dataset", "workers": [0], "prefetch": 2,
                       "shared_samples": False, "json": str(out)})()
    ib.cmd_loader(a)                                            # must not raise
    rec = json.loads(out.read_text())["variants"]["workers=0"]
    assert rec["batches"] == 0 and rec["mean_s"] is None and rec["median_s"] is None
    assert "no batches produced" in capsys.readouterr().out


def test_loader_refuses_too_few_samples_per_variant(tmp_path):
    p = tiny_file(tmp_path / "t.h5")
    a = type("A", (), {"data_path": str(p), "split": "train", "n": 60, "offset": 0,
                       "order": "dataset", "workers": [0, 4, 8], "prefetch": 4,
                       "shared_samples": False, "json": None})()
    with pytest.raises(SystemExit):
        ib.cmd_loader(a)
