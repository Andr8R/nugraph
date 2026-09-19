"""Unit tests for Stage-1b fine-tuning (scripts/particle_hierarchy/finetune.py). CPU only.

Run:  pytest -q tests/test_finetune.py
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("torch_scatter")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft = _load("finetune", ROOT / "scripts/particle_hierarchy/finetune.py")
sp = ft.sp
mat = pytest.importorskip("nugraph.models.nugraph3.materialize")
DEV = torch.device("cpu")


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(12345)


def make_event(k=3, n=40, sep=6.0, std=0.2, n_bkg=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    cen = torch.zeros(k, 8)
    cen[:, 0] = torch.arange(k, dtype=torch.float) * sep
    y = torch.arange(k).repeat_interleave(n)
    ox = cen[y] + std * torch.randn(k * n, 8, generator=g)
    beta = 0.1 * torch.rand(k * n, generator=g)
    beta[torch.arange(k) * n] = 0.9
    beta[5] = 0.8                                           # a duplicate seed on particle 0
    ox[5] = ox[0] + torch.nn.functional.one_hot(torch.tensor(1), 8).float()
    hit = torch.nn.functional.one_hot(y, k).float() + 0.1 * torch.randn(k * n, k, generator=g)
    ox = torch.cat([ox, 20 + torch.randn(n_bkg, 8, generator=g)])
    beta = torch.cat([beta, 0.05 * torch.rand(n_bkg, generator=g)])
    y = torch.cat([y, torch.full((n_bkg,), -1)])
    hit = torch.cat([hit, torch.zeros(n_bkg, k)])
    return ox, beta, hit, y


def batch_tensors(n_events=2):
    parts = [make_event(seed=s) for s in range(n_events)]
    ox = torch.cat([p[0] for p in parts]).requires_grad_()
    beta = torch.cat([p[1] for p in parts]).requires_grad_()
    hx = torch.cat([p[2] for p in parts]).requires_grad_()
    y = torch.cat([p[3] for p in parts])
    sizes = [p[0].size(0) for p in parts]
    ptr = torch.tensor([0] + list(np.cumsum(sizes)))
    fg = torch.ones(y.numel(), dtype=torch.bool)
    return ox, beta, hx, fg, y, torch.zeros(y.numel(), dtype=torch.long), ptr


def soft_loss(soft, tensors, grad_to_trunk, checkpoint=True):
    ox, beta, hx, fg, y, ysem, ptr = tensors
    return ft.soft_loss_from_hits(soft, ox, beta, hx, fg, y, ysem, ptr, grad_to_trunk,
                                  sp.s0.make_rule("fixed1:0.5"), mat, 3, 0.3, float("inf"), DEV,
                                  checkpoint=checkpoint)


def new_soft():
    soft = sp.build_model(3, 8, 16, 2)
    soft.overlap = "sparse"
    return soft


# --------------------------------------------------------------------------- #
def test_control_arm_gives_trunk_no_gradient_treatment_does():
    soft = new_soft()
    tc = batch_tensors()
    terms_c, n_c = soft_loss(soft, tc, grad_to_trunk=False)
    terms_c.sum().backward()
    assert n_c == 2
    assert tc[0].grad is None and tc[1].grad is None and tc[2].grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in soft.parameters())

    soft.zero_grad()
    tt = batch_tensors()
    terms_t, n_t = soft_loss(soft, tt, grad_to_trunk=True)
    terms_t.sum().backward()
    assert n_t == 2
    assert torch.allclose(terms_t.detach(), terms_c.detach(), atol=1e-6)   # same values
    for x in tt[:3]:
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_soft_checkpointing_does_not_change_loss_or_gradients():
    soft = new_soft()
    ta, tb = batch_tensors(), batch_tensors()
    la, _ = soft_loss(soft, ta, True, checkpoint=True)
    la.sum().backward()
    ga = [p.grad.clone() for p in soft.parameters()]
    soft.zero_grad()
    lb, _ = soft_loss(soft, tb, True, checkpoint=False)
    lb.sum().backward()
    assert torch.allclose(la.detach(), lb.detach(), atol=1e-6)
    for g1, p in zip(ga, soft.parameters()):
        assert torch.allclose(g1, p.grad, atol=1e-5)
    assert torch.allclose(ta[0].grad, tb[0].grad, atol=1e-5)


def test_seeded_sampler_order_is_reproducible_and_leaves_global_rng_alone():
    sizes = np.arange(1, 201)
    s1 = ft.seeded_balance_sampler(sizes, 8, 0.1, seed=1)
    s2 = ft.seeded_balance_sampler(sizes, 8, 0.1, seed=1)
    np.random.seed(0)
    e0 = list(iter(s1))
    np.random.rand(17)                                     # other RNG use in between
    e1 = list(iter(s1))
    np.random.seed(99)
    assert list(iter(s2)) == e0 and list(iter(s2)) == e1
    assert e0 != e1                                        # epochs differ
    assert list(iter(ft.seeded_balance_sampler(sizes, 8, 0.1, seed=2))) != e0
    np.random.seed(5)
    ref = np.random.rand()
    np.random.seed(5)
    list(iter(ft.seeded_balance_sampler(sizes, 8, 0.1, seed=3)))
    assert np.random.rand() == ref                         # global state restored


def test_sparse_overlap_equals_dense_values_and_gradients():
    n, K, m = 60, 9, 4
    D = torch.rand(n, K)
    nb = torch.topk(D, m, dim=1, largest=False).indices
    w = torch.rand(n, m).softmax(dim=1).requires_grad_()
    for scope in ("local", "all"):
        pairs = sp.candidate_pairs(nb, K, scope, DEV)
        full = torch.zeros(n, K).scatter(1, nb, w)
        dense = (full[:, pairs[:, 0]] * full[:, pairs[:, 1]]).sum(0)
        sparse = sp.sparse_overlap(w, nb, pairs, K)
        assert torch.allclose(dense, sparse, atol=1e-6)
        gd = torch.autograd.grad(dense.sum(), w, retain_graph=True)[0]
        gs = torch.autograd.grad(sparse.sum(), w)[0]
        assert torch.allclose(gd, gs, atol=1e-6)


def test_sparse_and_dense_overlap_give_the_same_labels():
    ox, beta, hit, y = make_event(sep=1.5, std=0.5)
    ev = {"batch": 0, "event": 0, "ox": ox, "beta": beta,
          "mask": torch.ones(y.numel(), dtype=torch.bool), "y_i": y.to(torch.int32),
          "y_sem": torch.zeros(y.numel(), dtype=torch.int8), "hit_x_fg": hit.half()}
    d = sp.prepare(ev, mat, sp.s0.make_rule("fixed1:0.5"), 3, 0.3, float("inf"), DEV)
    model = sp.build_model(3, 8, 16, 2)
    a, _ = sp.predict_labels(model, d, **ft.DECODE)
    model.overlap = "sparse"
    b, _ = sp.predict_labels(model, d, **ft.DECODE)
    assert torch.equal(a, b)


def test_primary_lookup_equals_dict_lookup():
    g = torch.Generator().manual_seed(3)
    y = torch.randint(-1, 40, (500,), generator=g)
    primary = {3: 0, 17: 5, 39: 2, 8: 7}
    ref = torch.tensor([primary.get(int(t), -1) for t in y.tolist()])
    assert torch.equal(sp.primary_lookup(primary, y, DEV), ref)
    assert torch.equal(sp.primary_lookup({}, y, DEV), torch.full((500,), -1))


def test_trunk_checkpoint_loads_with_unchanged_nugraph(tmp_path):
    pl = pytest.importorskip("pytorch_lightning")
    ng = pytest.importorskip("nugraph")
    net = ng.models.NuGraph3(in_features=7, hit_features=16, nexus_features=8,
                             interaction_features=8, beta_features=8, coord_features=8,
                             semantic_classes=("a", "b", "diffuse"), event_classes=("x", "y"),
                             event_head=True, semantic_head=True, filter_head=False,
                             instance_head=True)
    base = {"state_dict": net.state_dict(), "hyper_parameters": dict(net.hparams),
            "pytorch-lightning_version": pl.__version__}
    with torch.no_grad():
        for p_ in net.parameters():
            p_.add_(0.5)                                   # "fine-tuned" weights
    path = tmp_path / "trunk.ckpt"
    ft.save_trunk(net, base, path, {"epoch": 0, "global_step": 3, "arm": "T"})
    back = ng.models.NuGraph3.load_from_checkpoint(path, map_location="cpu")
    for k, v in net.state_dict().items():
        assert torch.equal(back.state_dict()[k], v)


def test_dev_identity_accepts_matching_and_refuses_other_events(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    sizes = [5, 7, 3, 9, 4, 6, 8, 2, 5, 5, 6, 7, 1, 2, 3, 4]      # 2 batches of 8
    torch.save([{"batch": i // 8, "event": i % 8, "mask": torch.ones(n, dtype=torch.bool)}
                for i, n in enumerate(sizes)], cache / "shard_0000.pt")
    ds = [{"hit": SimpleNamespace(num_nodes=n)} for n in sizes]
    assert ft.dev_identity(ds, 0, 2, cache) == list(range(16))
    ds[9] = {"hit": SimpleNamespace(num_nodes=99)}
    with pytest.raises(SystemExit):
        ft.dev_identity(ds, 0, 2, cache)
