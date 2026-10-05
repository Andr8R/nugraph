"""
Realistic end-to-end training-throughput callback (THROUGHPUT mode).

This PyTorch Lightning callback estimates the *real* cost of a training step by
timing one **continuous window** of batches with a wall clock -- it deliberately
does NOT synchronize CUDA around individual operations or batches, and it does
NOT register forward hooks. Per-operation, CUDA-synchronized timing is the
separate PROFILE mode (the instance decoder's own ``MaterializeTimer`` in
``models/nugraph3/materialize.py``, enabled by ``--materialize-timing``).

Window protocol (matches the agreed throughput contract):

1. run ``warmup`` batches normally (no sync, excluded from the measurement);
2. at the start of the measured window: ``reset_peak_memory_stats()`` then a
   single ``torch.cuda.synchronize()``, then start a wall-clock timer;
3. run the window (``window`` batches, or to end-of-epoch if ``window==0``) with
   **no** per-batch / per-op synchronization;
4. at the end of the window: a single ``torch.cuda.synchronize()``, stop the
   wall clock, and read ``max_memory_allocated()`` once;
5. batches/second and the projected time for ``epoch_batches`` (default 11280)
   are computed from the **full window** (``n_batches / elapsed``).

Run three times per backend (baseline / materialize+no-metric / materialize+
metric) to separate clustering vs metric vs total overhead. When the callback is
not added there is no overhead.
"""
import json
from time import perf_counter

import torch
from pytorch_lightning.callbacks import Callback


class ThroughputTimer(Callback):
    """Measure real training throughput over one continuous, un-synchronized window."""

    def __init__(self, epoch_batches: int = 11280, warmup: int = 10,
                 window: int = 0, out_path: str = None):
        """
        Args:
            epoch_batches: batches in a full training epoch (DUNE default 11280),
                used only to project epoch time from the measured throughput.
            warmup: batches to run before the measured window starts (excluded).
            window: number of batches to measure; 0 = measure from the end of
                warmup until the end of the epoch. When >0 the run is stopped
                right after the window (set ``--limit_train_batches`` >= warmup +
                window, or leave it unset and rely on this early stop).
            out_path: optional stem; writes ``<stem>.json`` (and a one-row
                ``<stem>.csv``) with the window summary.
        """
        super().__init__()
        self.epoch_batches = epoch_batches
        self.warmup = warmup
        self.window = window
        self.out_path = out_path
        self._cuda = torch.cuda.is_available()
        self._seen = 0          # completed train batches (across warmup+window)
        self._measuring = False
        self._done = False
        self._t0 = None
        self._n = 0             # batches measured inside the window
        self._elapsed = None
        self._peak_mb = float("nan")

    # -- window control (NO per-batch / per-op sync) ----------------------- #
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # begin the measured window exactly once, after `warmup` batches
        if self._done or self._measuring:
            return
        if self._seen >= self.warmup:
            if self._cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()   # single sync BEFORE the window
            self._t0 = perf_counter()
            self._measuring = True
            self._n = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._seen += 1
        if self._measuring and not self._done:
            self._n += 1
            if self.window and self._n >= self.window:
                self._close(trainer)
                trainer.should_stop = True  # end cleanly right after the window

    def on_train_epoch_end(self, trainer, pl_module):
        # close an open (window==0) measurement at epoch end
        if self._measuring and not self._done:
            self._close(trainer)

    def on_fit_end(self, trainer, pl_module):
        if self._measuring and not self._done:
            self._close(trainer)
        if self._done:
            self._report()

    # -- close the window with a single sync ------------------------------- #
    def _close(self, trainer):
        if self._done or not self._measuring:
            return
        if self._cuda:
            torch.cuda.synchronize()       # single sync AFTER the window
            self._peak_mb = torch.cuda.max_memory_allocated() / 1e6
        self._elapsed = perf_counter() - self._t0
        self._measuring = False
        self._done = True

    # -- summary ----------------------------------------------------------- #
    def _report(self):
        if not self._elapsed or self._n == 0:
            print("\n=== ThroughputTimer: no batches measured (increase "
                  "limit_train_batches beyond warmup) ===\n")
            return
        bps = self._n / self._elapsed
        proj_epoch_h = (self.epoch_batches / bps) / 3600.0 if bps > 0 else float("nan")
        summary = {
            "warmup_batches": self.warmup,
            "window_batches": self._n,
            "window_elapsed_s": self._elapsed,
            "batches_per_s": bps,
            "epoch_batches": self.epoch_batches,
            "projected_train_epoch_hours": proj_epoch_h,
            "peak_mem_mb": self._peak_mb,
            "cuda": self._cuda,
        }
        print("\n=== ThroughputTimer (continuous window) ===")
        for k, val in summary.items():
            print(f"  {k}: {val}")
        print("  (reference: baseline ~1.27 batch/s, ~2h28m train/epoch)\n")
        if self.out_path:
            import csv
            with open(f"{self.out_path}.json", "w") as f:
                json.dump(summary, f, indent=2)
            with open(f"{self.out_path}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary.keys()))
                w.writeheader()
                w.writerow(summary)
            print(f"  wrote {self.out_path}.json and {self.out_path}.csv")
