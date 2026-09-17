#!/usr/bin/env python
"""Validate-only launcher: run NuGraph's own validation loop on one checkpoint.

Reuses train_with_checkpoints.py's argument parsing (same data/model flags), loads the
model from --ckpt (saved hparams win, as on resume), and calls Trainer.validate with
no logger. Prints every logged metric and writes them as JSON to --out.

Model-neutral: nothing in nugraph/ is modified. For a code-patched variant, run it with
that variant's patch still applied so the checkpoint loads into the matching class.
"""
import json
import sys

import pytorch_lightning as pl
import nugraph as ng

from train_with_checkpoints import configure, Data


def main():
    argv = sys.argv[1:]
    ckpt = out = None
    rest = []
    it = iter(argv)
    for a in it:
        if a == "--ckpt":
            ckpt = next(it)
        elif a == "--out":
            out = next(it)
        else:
            rest.append(a)
    if ckpt is None:
        raise SystemExit("usage: nugraph_validate.py --ckpt <file> [--out metrics.json] <train.py data flags>")
    sys.argv = [sys.argv[0]] + rest
    args, Model = configure()

    nudata = Data(args.data_path, batch_size=args.batch_size, model=Model,
                  shuffle=args.shuffle, balance_frac=args.balance_frac,
                  num_workers=args.num_workers, featext=args.featext)
    model = Model.load_from_checkpoint(ckpt)

    accelerator, devices = ng.util.configure_device(args.device)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=False,
                         limit_val_batches=args.limit_val_batches,
                         enable_checkpointing=False)
    metrics = trainer.validate(model, datamodule=nudata, verbose=False)[0]
    metrics = {k: float(v) for k, v in sorted(metrics.items())}

    print(f"checkpoint        : {ckpt}")
    print(f"batch_size        : {args.batch_size}")
    print(f"limit_val_batches : {args.limit_val_batches or 'full'}")
    for k, v in metrics.items():
        print(f"  {k:32s} {v:.6f}")
    if out:
        with open(out, "w") as f:
            json.dump({"checkpoint": ckpt, "batch_size": args.batch_size,
                       "limit_val_batches": args.limit_val_batches,
                       "metrics": metrics}, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
