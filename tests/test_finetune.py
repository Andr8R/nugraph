"""Unit tests for Stage-1b fine-tuning (scripts/particle_hierarchy/finetune.py). CPU only.

Run:  pytest -q tests/test_finetune.py
"""
import importlib.util
from pathlib import Path

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


# ---- profiling instrumentation (diagnostic only) -------------------------- #
def test_profiling_does_not_change_the_loss_or_the_gradients():
    """The timers must be pure observation: same numbers, same gradients, with and without them."""
    soft = new_soft()
    ta, tb = batch_tensors(), batch_tensors()
    la, na = soft_loss(soft, ta, True)
    la.sum().backward()
    ga = [p.grad.clone() for p in soft.parameters()]
    soft.zero_grad()

    timers, syncs = {}, []
    ox, beta, hx, fg, y, ysem, ptr = tb
    lb, nb = ft.soft_loss_from_hits(soft, ox, beta, hx, fg, y, ysem, ptr, True,
                                    sp.s0.make_rule("fixed1:0.5"), mat, 3, 0.3, float("inf"), DEV,
                                    timers=timers, sync=lambda: syncs.append(1))
    lb.sum().backward()
    assert na == nb == 2
    assert torch.equal(la.detach(), lb.detach())                      # bitwise, not just close
    for g1, p in zip(ga, soft.parameters()):
        assert torch.equal(g1, p.grad)
    for k in ("ev_inputs", "prepare", "swap_grad_tensors", "soft_fwd", "soft_other"):
        assert timers[k] >= 0.0, k
    assert len(syncs) == 2 * 4 + 1                                    # 4 stages x 2 events + close


def test_tick_is_inert_without_timers_and_accumulates_with_them():
    assert ft.tick(None, None, "x", 12.5) == 12.5                     # production path: no-op
    timers, calls = {}, []
    t = ft.tick(timers, lambda: calls.append(1), "x", 0.0)
    assert t > 0 and timers["x"] > 0 and calls == [1]
    before = timers["x"]
    ft.tick(timers, None, "x", t)
    assert timers["x"] > before                                       # same key accumulates


def test_profile_row_accounts_for_every_second_of_the_step():
    row = ft.profile_row(7, 1, {"nugraph_fwd": 0.5, "backward": 1.0}, 2.0,
                         {"n_events": 32, "n_soft_events": 30})
    assert row["global_step"] == 7 and row["epoch"] == 1 and row["n_events"] == 32
    for k in ft.PROFILE_STAGES:
        assert f"t_{k}_s" in row                                      # a stable column set
    assert row["t_prepare_s"] == 0.0
    assert row["t_unattributed_s"] == pytest.approx(0.5)              # 2.0 - (0.5 + 1.0)


def test_split_backward_is_exact_when_the_graphs_are_disjoint():
    """Arm C's claim: separate backwards accumulate the same gradients as one backward on the sum.

    The soft layer sees detached copies in arm C, so it shares no node with the trunk graph; a stand-in
    trunk loss on its own parameters reproduces that structure.
    """
    soft = new_soft()
    trunk = torch.nn.Linear(4, 1)
    x = torch.ones(2, 4)

    def grads(split):
        soft.zero_grad()
        trunk.zero_grad()
        terms, _ = soft_loss(soft, batch_tensors(), grad_to_trunk=False)
        loss_ng = trunk(x).square().mean()
        if split:
            loss_ng.backward()
            (1.0 * terms.sum()).backward()
        else:
            (loss_ng + 1.0 * terms.sum()).backward()
        return [p.grad.clone() for p in list(trunk.parameters()) + list(soft.parameters())]

    for g1, g2 in zip(grads(False), grads(True)):
        assert torch.equal(g1, g2)


def test_split_backward_is_refused_outside_arm_c():
    a = type("A", (), {"profile_split_backward": True, "arm": "T", "profile_steps": 50})()
    with pytest.raises(SystemExit, match="requires --arm C"):
        ft.validate_profile_args(a)
    a.arm, a.profile_steps = "C", 0
    with pytest.raises(SystemExit, match="profile-steps"):
        ft.validate_profile_args(a)
    a.profile_steps = 50
    assert ft.validate_profile_args(a) is None                        # arm C + profiling is allowed
    ft.validate_profile_args(type("A", (), {"profile_split_backward": False, "arm": "T",
                                            "profile_steps": 0})())   # off: never refuses


def test_profile_steps_is_not_part_of_the_resume_identity(tmp_path):
    """Profiling changes who is watching, not what is computed, so it must not block a resume."""
    a = type("A", (), {k: v for k, v in dict(
        arm="T", seed=1, lam=1.0, lr=1e-4, head_lr=1e-3, epochs=5, batch_size=32,
        balance_frac=0.1, limit_train_batches=None, dev_batches="0:2", no_soft_checkpoint=False,
        resume_every=100, data_path=str(tmp_path / "d.h5"), num_workers=4, prefetch_factor=2,
        profile_steps=100, profile_split_backward=False).items()})()
    cfg = ft.run_config(a, "sha", "hsha", {"git_commit": "c", "git_dirty": False})
    for k in ("profile_steps", "profile_split_backward", "num_workers", "prefetch_factor"):
        assert k not in cfg, k
    a.profile_steps = 0
    assert ft.run_config(a, "sha", "hsha", {"git_commit": "c", "git_dirty": False}) == cfg


def test_seeded_sampler_order_is_a_pure_function_of_seed_and_epoch():
    sizes = np.arange(1, 201)
    s1 = ft.seeded_balance_sampler(sizes, 8, 0.1, seed=1)
    s2 = ft.seeded_balance_sampler(sizes, 8, 0.1, seed=1)
    np.random.seed(0)
    e0, e1 = s1.epoch_order(0), s1.epoch_order(1)
    np.random.rand(17)                                     # other RNG use in between
    assert s2.epoch_order(1) == e1 and s2.epoch_order(0) == e0
    assert e0 != e1                                        # epochs differ
    assert ft.seeded_balance_sampler(sizes, 8, 0.1, seed=2).epoch_order(0) != e0
    np.random.seed(5)
    ref = np.random.rand()
    np.random.seed(5)
    s1.epoch_order(3)
    list(iter(s1))
    assert np.random.rand() == ref                         # global state restored
    s1.epoch, s1.skip = 1, 24
    assert list(iter(s1)) == e1[24:]                       # resumed mid-epoch


# --------------------------------------------------------------------------- #
# exact resume (generic loop, CPU, bitwise)
# --------------------------------------------------------------------------- #
class ToyLoop(ft.ResumableLoop):
    """Tiny model trained through the same ResumableLoop; consumes torch and numpy RNG in
    step() and end_epoch() so that RNG restoration is exercised."""

    def __init__(self, out, init_seed=3, epochs=3, resume_every=3, limit=None, lr=0.05):
        out.mkdir(parents=True, exist_ok=True)
        super().__init__(out, epochs, {"toy": 1, "epochs": epochs, "lr": lr}, resume_every, limit)
        torch.manual_seed(init_seed)
        self.model = torch.nn.Linear(3, 1)
        g = torch.Generator().manual_seed(0)
        self.X, self.Y = torch.randn(40, 3, generator=g), torch.randn(40, 1, generator=g)
        self.bs = 4
        self.sampler = ft.seeded_balance_sampler(np.arange(1, 41), self.bs, 0.1, seed=7)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def modules(self):
        return {"model": self.model}

    def batches(self, epoch, skip):
        order = self.sampler.epoch_order(epoch)
        for k in range(skip, len(order) // self.bs):
            ids = order[k * self.bs:(k + 1) * self.bs]
            yield ids, torch.tensor(ids)

    def step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        noise = 0.01 * torch.randn(len(batch), 1)
        loss = ((self.model(self.X[batch]) + noise - self.Y[batch]) ** 2).mean()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach()), "np": float(np.random.rand())}

    def end_epoch(self, epoch):
        self.hist.append({"epoch": epoch, "loss": self.tr["loss"] / self.tr["n"],
                          "n": self.tr["n"], "r": float(torch.rand(1))})


def _steps(path):
    import csv
    with open(path / "steps.csv") as f:
        return [{k: v for k, v in r.items() if k != "step_s"} for r in csv.DictReader(f)]


def _uninterrupted(out):
    np.random.seed(1)
    loop = ToyLoop(out)
    loop.run()
    return loop


def _check_same(ref, other, ref_dir, other_dir):
    for (k, p), (_, q) in zip(ref.model.state_dict().items(), other.model.state_dict().items()):
        assert torch.equal(p, q), k
    assert other.hist == ref.hist
    assert _steps(other_dir) == _steps(ref_dir)
    assert other.st == ref.st


@pytest.mark.parametrize("crash", [
    {"crash_at_step": 7},          # mid-epoch, between two resume saves (steps.csv truncated)
    {"crash_at_step": 12},         # exactly at a resume save
    {"crash_after_epoch": 0},      # after an epoch's validation, before its resume save
    {"crash_after_epoch": 2},      # after the last epoch, before the final save
])
def test_interrupted_and_resumed_run_equals_uninterrupted(tmp_path, crash):
    ref = _uninterrupted(tmp_path / "ref")
    assert ref.st["epoch"] == 3 and ref.st["global_step"] > 20
    np.random.seed(1)
    first = ToyLoop(tmp_path / "run")
    with pytest.raises(ft.SimulatedCrash):
        first.run(**crash)
    np.random.seed(12345)                                  # a fresh process: other init, RNG
    torch.manual_seed(999)
    resumed = ToyLoop(tmp_path / "run", init_seed=77)
    resumed.load_resume()
    resumed.run()
    _check_same(ref, resumed, tmp_path / "ref", tmp_path / "run")


def test_resume_twice_and_after_completion(tmp_path):
    ref = _uninterrupted(tmp_path / "ref")
    np.random.seed(1)
    with pytest.raises(ft.SimulatedCrash):
        ToyLoop(tmp_path / "run").run(crash_at_step=5)
    second = ToyLoop(tmp_path / "run", init_seed=8)
    second.load_resume()
    with pytest.raises(ft.SimulatedCrash):
        second.run(crash_at_step=17)
    third = ToyLoop(tmp_path / "run", init_seed=9)
    third.load_resume()
    third.run()
    _check_same(ref, third, tmp_path / "ref", tmp_path / "run")
    done = ToyLoop(tmp_path / "run", init_seed=10)          # resuming a finished run: no-op
    done.load_resume()
    done.run()
    _check_same(ref, done, tmp_path / "ref", tmp_path / "run")


def test_resume_refuses_a_changed_configuration(tmp_path):
    np.random.seed(1)
    with pytest.raises(ft.SimulatedCrash):
        ToyLoop(tmp_path / "run").run(crash_at_step=4)
    with pytest.raises(SystemExit):
        ToyLoop(tmp_path / "run", lr=0.01).load_resume()


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
    base = {"state_dict": {k: v.clone() for k, v in net.state_dict().items()},
            "hyper_parameters": dict(net.hparams), "pytorch-lightning_version": pl.__version__}
    # "fine-tuned" state. InputNorm keeps running statistics as non-trainable Parameters,
    # including an integer `count`, so only floating-point tensors get the float offset.
    counts = [k for k, v in net.named_parameters() if not v.is_floating_point()]
    assert counts, "expected InputNorm's integer running count among the parameters"
    with torch.no_grad():
        for p_ in net.parameters():
            if p_.is_floating_point():
                p_.add_(0.5)
            else:
                p_.fill_(123)
    path = tmp_path / "trunk.ckpt"
    ft.save_trunk(net, base, path, {"epoch": 0, "global_step": 3, "arm": "T"})
    back = ng.models.NuGraph3.load_from_checkpoint(path, map_location="cpu")
    state, loaded = net.state_dict(), back.state_dict()
    assert set(loaded) == set(state)
    for k, v in state.items():
        assert loaded[k].dtype == v.dtype and torch.equal(loaded[k], v), k
    assert all(int(loaded[k].flatten()[0]) == 123 for k in counts)
    assert any(not torch.equal(loaded[k], base["state_dict"][k]) for k in state)   # not the base


class _HeteroDataset:
    """Like NuGraphDataset: every __getitem__ builds a fresh HeteroData, so a node store taken
    from it outlives its (weakly referenced) parent unless the sample is kept."""

    def __init__(self, sizes):
        self.sizes = list(sizes)

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, i):
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["hit"].x = torch.zeros(self.sizes[i], 3)
        data["sp"].x = torch.zeros(2, 3)
        return data


def _dev_cache(tmp_path, sizes):
    cache = tmp_path / "cache"
    cache.mkdir()
    torch.save([{"batch": i // 8, "event": i % 8, "mask": torch.ones(n, dtype=torch.bool)}
                for i, n in enumerate(sizes)], cache / "shard_0000.pt")
    return cache


SIZES = [5, 7, 3, 9, 4, 6, 8, 2, 5, 5, 6, 7, 1, 2, 3, 4]            # 2 batches of 8


def test_n_hits_reads_a_temporary_pyg_sample():
    pytest.importorskip("torch_geometric")
    ds = _HeteroDataset(SIZES)
    assert [ft.n_hits(ds[i]) for i in range(len(ds))] == SIZES


def test_dev_identity_accepts_matching_and_refuses_other_events(tmp_path):
    pytest.importorskip("torch_geometric")
    cache = _dev_cache(tmp_path, SIZES)
    assert ft.dev_identity(_HeteroDataset(SIZES), 0, 2, cache) == list(range(16))
    assert ft.dev_identity(_HeteroDataset(SIZES), 1, 2, cache) == list(range(8, 16))
    wrong = list(SIZES)
    wrong[9] = 99
    with pytest.raises(SystemExit):
        ft.dev_identity(_HeteroDataset(wrong), 0, 2, cache)
