"""NuGraphDataset with a lazily opened HDF5 handle: picklable, and DataLoader workers change nothing.

The handle used to be opened in __init__, which made the dataset unpicklable and therefore forced
num_workers=0 on EAF. These tests pin the two properties that matter: the handle is never pickled (so
workers can be used at all), and batches are identical whether they are read by the main process or by
workers. Needs h5py + pynuml, so they skip on GPVM.
"""
import importlib.util
import sys
import pickle
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
pytest.importorskip("pynuml")
ng = pytest.importorskip("nugraph")

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


def tiny_file(path, n=64, hits=7):
    with h5py.File(path, "w") as f:
        names = [f"evt{i:03d}" for i in range(n)]
        for i, name in enumerate(names):
            dt = np.dtype([("hit/x", np.float32, (hits, 3)), ("hit/y_semantic", np.int8, (hits,)),
                           ("evt/y", np.int64)])
            f.create_dataset(f"/dataset/{name}", shape=(), dtype=dt,
                             data=(np.full((hits, 3), float(i), np.float32),
                                   np.zeros(hits, np.int8), i))
        f.create_dataset("samples/train", data=np.array(names, dtype="S16"))
    return [n for n in names]


def dataset(tmp_path, transform=None):
    from nugraph.data import NuGraphDataset
    names = tiny_file(tmp_path / "t.h5")
    return NuGraphDataset(str(tmp_path / "t.h5"), names, transform), names


def test_handle_opens_lazily_and_is_never_pickled(tmp_path):
    ds, names = dataset(tmp_path)
    assert ds._file is None                                    # nothing open before first access
    a = ds[5]
    assert ds._file is not None
    revived = pickle.loads(pickle.dumps(ds))                   # what a worker receives
    assert revived._file is None
    b = revived[5]
    assert revived._file is not None                           # reopened in the new process
    assert torch.equal(a["hit"].x, b["hit"].x) and float(b["hit"].x[0, 0]) == 5.0


def test_samples_and_length_are_unchanged(tmp_path):
    ds, names = dataset(tmp_path)
    assert len(ds) == len(names) == 64
    assert float(ds[0]["hit"].x[0, 0]) == 0.0 and float(ds[63]["hit"].x[0, 0]) == 63.0


@pytest.mark.parametrize("workers", [1, 2])
def test_dataloader_workers_yield_identical_batches(tmp_path, workers):
    """The semantics gate: same sampler order, same batches, whoever reads them."""
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    ds, _ = dataset(tmp_path)
    order = list(range(60, 0, -1))                              # a fixed, non-trivial order
    sub = Subset(ds, order)

    def batches(nw):
        kw = dict(batch_size=4, shuffle=False, drop_last=True, num_workers=nw)
        if nw:
            kw.update(persistent_workers=True, prefetch_factor=2)
        return [b for b in DataLoader(sub, **kw)]

    ref, got = batches(0), batches(workers)
    assert len(ref) == len(got) == 15
    for a, b in zip(ref, got):
        assert a["hit"].num_nodes == b["hit"].num_nodes
        assert torch.equal(a["hit"].x, b["hit"].x)              # identical bytes, identical order
        assert torch.equal(a["evt"].y, b["evt"].y)


def test_io_bench_lazy_dataset_matches_the_real_one(tmp_path):
    """io_bench's standalone wrapper must read the same tensors as NuGraphDataset."""
    ds, names = dataset(tmp_path)
    lazy = ib.LazyDataset(tmp_path / "t.h5", "train", transform=None)
    assert lazy.names == names
    for i in (0, 7, 63):
        assert torch.equal(ds[i]["hit"].x, lazy[i]["hit"].x)
