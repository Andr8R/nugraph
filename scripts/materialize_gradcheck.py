#!/usr/bin/env python
"""
Gradient/loss equivalence check for train-time materialization.

Verifies empirically (not just by the "detached tensor" argument) that enabling
train-time materialization does NOT change the training loss or any parameter
gradient. Runs the SAME model state on the SAME (deterministic, first) batch
twice -- once with ``materialize_train`` OFF and once ON -- and reports the
maximum absolute and relative differences in the loss and in every parameter's
gradient after ``backward``.

Must use the real training configuration (heads + featext + in-feats 11).
Requires a GPU + checkpoint + dataset -> run on EAF.

Example (EAF):
  python scripts/materialize_gradcheck.py --device 0 --checkpoint <CKPT> \
      --data-path $DATA --event --semantic --filter --instance --particle-loss \
      --featext --in-feats 11 --no-lr-scheduler --num-workers 0 --batch-size 8 \
      --materializer seed
"""
import argparse

import torch

import nugraph as ng


def configure():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--device', type=int, default=None)
    p.add_argument('--split', type=str, default='validation',
                   choices=('train', 'validation', 'test'))
    p = ng.data.H5DataModule.add_data_args(p)
    p = ng.models.NuGraph3.add_model_args(p)
    return p.parse_args()


def run_once(model, batch, mat_on):
    """One deterministic forward+backward; return (loss, {param: grad})."""
    torch.manual_seed(0)
    model.instance_decoder.materialize_train = mat_on
    model.zero_grad(set_to_none=True)
    loss, _ = model(batch, "train")
    loss.backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def main():
    args = configure()
    nudata = ng.data.H5DataModule(
        args.data_path, batch_size=args.batch_size, model=ng.models.NuGraph3,
        shuffle='random', num_workers=args.num_workers, featext=args.featext)
    model = ng.models.NuGraph3.load_from_checkpoint(args.checkpoint)
    accelerator, _ = ng.util.configure_device(args.device)
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")
    model = model.to(device).train()

    if not hasattr(model, "instance_decoder"):
        raise RuntimeError("checkpoint has no instance head; pass --instance "
                           "and use an instance-trained checkpoint")
    # freeze running feature-norm statistics so both runs are identical
    try:
        model.encoder.input_norm.update = False
    except AttributeError:
        pass

    loader = {"train": nudata.train_dataloader,
              "validation": nudata.val_dataloader,
              "test": nudata.test_dataloader}[args.split]

    # the val/test loaders are unshuffled -> the first batch is deterministic;
    # re-create the iterator for each run so both see the identical, unmutated batch
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    batch_off = next(iter(loader())).to(device)
    loss_off, g_off = run_once(model, batch_off, mat_on=False)

    model.load_state_dict(state)  # restore exact state
    batch_on = next(iter(loader())).to(device)
    loss_on, g_on = run_once(model, batch_on, mat_on=True)

    # compare
    dloss = abs(loss_on - loss_off)
    rloss = dloss / (abs(loss_off) + 1e-12)
    max_abs = 0.0
    max_rel = 0.0
    worst = None
    keys = sorted(set(g_off) & set(g_on))
    only_off = sorted(set(g_off) - set(g_on))
    only_on = sorted(set(g_on) - set(g_off))
    for k in keys:
        a = (g_on[k] - g_off[k]).abs().max().item()
        r = a / (g_off[k].abs().max().item() + 1e-12)
        if a > max_abs:
            max_abs, worst = a, k
        max_rel = max(max_rel, r)

    print("=== materialization gradient/loss equivalence ===")
    print(f"materializer         : {args.materializer}")
    print(f"loss (off)           : {loss_off:.8f}")
    print(f"loss (on)            : {loss_on:.8f}")
    print(f"loss abs / rel diff  : {dloss:.3e} / {rloss:.3e}")
    print(f"grads compared       : {len(keys)} params")
    print(f"max abs grad diff    : {max_abs:.3e}  (param: {worst})")
    print(f"max rel grad diff    : {max_rel:.3e}")
    if only_off or only_on:
        print(f"WARNING: param-grad key mismatch off-only={only_off} "
              f"on-only={only_on}")
    ok = dloss < 1e-5 and max_abs < 1e-5
    print(f"EQUIVALENT within tol (1e-5): {ok}")
    print("(expected: equal within floating-point tolerance; materialization is "
          "detached and does not feed the loss)")


if __name__ == "__main__":
    main()
