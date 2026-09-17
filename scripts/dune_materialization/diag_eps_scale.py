#!/usr/bin/env python
"""Scale-aware instance-ARI diagnostic for OC checkpoints (DBSCAN eps sweep + adaptive eps).

For each checkpoint, on the same fixed validation events, reports: fixed eps=0.3 ARI (the
NuGraph validation number), the best ARI over an absolute eps sweep, the best ARI after
per-dimension normalisation, the best ARI with a per-event adaptive radius
eps = rho * median(k-th nearest-neighbour distance), and latent-scale statistics.
Used to show that the unnormalised DUNE runs' ARI "decay" was partly a fixed-eps artefact
and that the LayerNorm run fixes the representation itself.

  python diag_eps_scale.py --ckpts <epoch=00...ckpt> <epoch=01...ckpt> ... \
      --data-path <dune.gnn.h5> --device 0 --limit-batches 40 --out traj.txt

Forward passes only; the model class must match the checkpoints (e.g. LayerNorm checkpoints
need the LayerNorm commit). Slow: tens of minutes per checkpoint (many DBSCAN fits).
"""
import argparse
from itertools import islice
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import nugraph as ng

EPS_SWEEP = [0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
             0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
RHO_SWEEP = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]  # eps = rho * per-event k-NN density radius
# (extended past 3.0: the first run's optimum sat at the old ceiling rho=3.0, unbracketed)
MIN_SAMPLES = 15  # match the decoder's DBSCAN(eps=0.3, min_samples=15)


def foreground_mask(h):
    """Replicate the decoder's eval foreground selection (no cross-repo import)."""
    m = torch.ones_like(h.of, dtype=torch.bool)
    if hasattr(h, "x_filter"):
        m = m & (h.x_filter > 0.5)
    if hasattr(h, "x_semantic"):
        m = m & (h.x_semantic.argmax(dim=1) != 6)  # exclude diffuse (index 6, DUNE ordering)
    return m


def collect(ckpt, dm, n_batches, device):
    """Forward a FRESH pass over the first n_batches val batches; return per-event
    (ox, y_semantic, y_i, foreground) plus a per-batch fingerprint.

    The model forward mutates the batch in place (encoder overwrites hit.x 7->128-D,
    decoders add ox/of/x_semantic/particle nodes, etc.), so we must NOT reuse batch
    objects across checkpoints. We re-iterate the val dataloader (num_workers=0,
    shuffle=False -> deterministic, identical events) so each checkpoint gets brand-new,
    unmodified batches. The fingerprint lets the caller verify the two passes matched.
    """
    model = ng.models.NuGraph3.load_from_checkpoint(ckpt).eval().to(device)
    events, fingerprint = [], []
    with torch.no_grad():
        for b in islice(dm.val_dataloader(), n_batches):
            fingerprint.append(int(b["hit"].num_nodes))  # captured BEFORE forward mutates it
            b = b.to(device)
            model(b, "test")  # populates h.ox / h.of / h.x_semantic
            for l in b.to_data_list():
                h = l["hit"]
                events.append((h.ox.detach().float().cpu().numpy(),
                               h.y_semantic.detach().cpu().numpy(),
                               l.y_i().detach().cpu().numpy(),
                               foreground_mask(h).cpu().numpy()))
    return events, fingerprint


def mean_ari(events, eps, mins=MIN_SAMPLES, mean=None, std=None):
    """Per-event ARI (mask y_semantic>=0, pred=-1 off-foreground/noise), event-weighted."""
    aris = []
    for ox, ysem, yi, fg in events:
        x = ox if mean is None else (ox - mean) / std
        pred = np.full(ox.shape[0], -1, dtype=np.int64)
        fgi = np.nonzero(fg)[0]
        if fgi.size:
            pred[fgi] = DBSCAN(eps=eps, min_samples=mins).fit_predict(x[fgi])
        m = ysem >= 0 if ysem.ndim == 1 else ysem.argmax(1) >= 0
        if m.sum() >= 2:
            aris.append(adjusted_rand_score(yi[m], pred[m]))
    return float(np.mean(aris)) if aris else float("nan"), len(aris)


def adaptive_ari(events, rho, k=MIN_SAMPLES):
    """Per-event scale-adaptive DBSCAN: eps = rho * median(k-th NN distance) of that
    event's foreground coords. Translation/anisotropy-robust (uses real point distances),
    event-local, one dimensionless knob rho. Returns (mean ARI, median per-event scale s)."""
    from sklearn.neighbors import NearestNeighbors
    aris, svals = [], []
    for ox, ysem, yi, fg in events:
        pred = np.full(ox.shape[0], -1, dtype=np.int64)
        fgi = np.nonzero(fg)[0]
        if fgi.size > k:
            xf = ox[fgi]
            d, _ = NearestNeighbors(n_neighbors=k + 1).fit(xf).kneighbors(xf)
            s = float(np.median(d[:, k]))  # k-th NN distance = local density radius DBSCAN keys on
            svals.append(s)
            if s > 0:
                pred[fgi] = DBSCAN(eps=rho * s, min_samples=k).fit_predict(xf)
        m = ysem >= 0
        if m.sum() >= 2:
            aris.append(adjusted_rand_score(yi[m], pred[m]))
    return (float(np.mean(aris)) if aris else float("nan"),
            float(np.median(svals)) if svals else float("nan"))


def scale_stats(events):
    """Latent-scale summary over foreground coords of the subset. Robust to a
    checkpoint whose foreground is empty/degenerate for this subset (e.g. a trough
    epoch where the semantic head predicts ~all-diffuse) — returns NaN stats instead
    of crashing, so the trajectory continues and that checkpoint still gets its ARIs."""
    dim = int(events[0][0].shape[1]) if events else 1
    fg_coords = [ox[np.nonzero(fg)[0]] for ox, _, _, fg in events if fg.any()]
    if not fg_coords:
        return np.zeros((0, dim)), np.full(dim, np.nan), float("nan"), float("nan")
    fgc = np.concatenate(fg_coords)
    perdim_std = fgc.std(axis=0)
    rms = float(np.sqrt((fgc ** 2).sum(axis=1).mean()))
    med_nn = float("nan")
    if fgc.shape[0] >= 2:  # median nearest-neighbour distance on a random sample
        rng = np.random.default_rng(0)
        s = fgc[rng.choice(fgc.shape[0], size=min(2000, fgc.shape[0]), replace=False)]
        d = np.sqrt(((s[:, None, :] - s[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        med_nn = float(np.median(d.min(axis=1)))
    return fgc, perdim_std, rms, med_nn


def report(name, events, out):
    fgc, perdim_std, rms, med_nn = scale_stats(events)
    fixed, n = mean_ari(events, 0.3)
    sweep = [(eps, mean_ari(events, eps)[0]) for eps in EPS_SWEEP]
    best_eps, best = max(sweep, key=lambda t: (t[1] if t[1] == t[1] else -9))
    if fgc.shape[0]:  # per-dim z-score only meaningful with foreground coords
        mean, std = fgc.mean(axis=0), fgc.std(axis=0) + 1e-9
        norm_sweep = [(eps, mean_ari(events, eps, mean=mean, std=std)[0]) for eps in EPS_SWEEP]
    else:
        norm_sweep = [(eps, float("nan")) for eps in EPS_SWEEP]
    nbest_eps, nbest = max(norm_sweep, key=lambda t: (t[1] if t[1] == t[1] else -9))
    adapt = [(r, *adaptive_ari(events, r)) for r in RHO_SWEEP]  # (rho, ari, median_s)
    abest_rho, abest, s_med = max(adapt, key=lambda t: (t[1] if t[1] == t[1] else -9))
    lines = [
        f"=== {name}  ({n} events) ===",
        f"  fixed eps=0.3 ARI            : {fixed:.4f}",
        f"  best ARI (raw eps sweep)     : {best:.4f}  @ eps={best_eps}",
        f"  best ARI (per-dim z-score)   : {nbest:.4f}  @ eps={nbest_eps}",
        f"  best ARI (adaptive eps=ρ·kNN): {abest:.4f}  @ ρ={abest_rho}  (median per-event scale s={s_med:.4f})",
        f"  latent scale: global RMS={rms:.3f}  median-NN={med_nn:.4f}  "
        f"per-dim std mean={perdim_std.mean():.3f} (min {perdim_std.min():.3f}, max {perdim_std.max():.3f})",
        f"  raw sweep     : " + " ".join(f"{e}:{a:.3f}" for e, a in sweep),
        f"  norm sweep    : " + " ".join(f"{e}:{a:.3f}" for e, a in norm_sweep),
        f"  adaptive sweep: " + " ".join(f"{r}:{a:.3f}" for r, a, _ in adapt),
    ]
    print("\n".join(lines))
    out.write("\n".join(lines) + "\n\n")
    return dict(fixed=fixed, best=best, best_eps=best_eps, nbest=nbest,
                abest=abest, abest_rho=abest_rho, s_med=s_med, rms=rms, med_nn=med_nn)


def main():
    import os
    p = argparse.ArgumentParser()
    p.add_argument("--early-ckpt")
    p.add_argument("--late-ckpt")
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="Trajectory mode: N checkpoints evaluated on the same fixed events "
                        "(alternative to --early-ckpt/--late-ckpt).")
    p.add_argument("--data-path", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="logs/materialization_eval/diag_eps_scale.txt")
    args = p.parse_args()

    if args.ckpts:
        ckpts = list(args.ckpts); labels = [os.path.basename(c) for c in ckpts]
    elif args.early_ckpt and args.late_ckpt:
        ckpts = [args.early_ckpt, args.late_ckpt]; labels = ["EARLY (peak ARI)", "LATE (decayed)"]
    else:
        raise SystemExit("give --ckpts <list>  OR  both --early-ckpt and --late-ckpt")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dm = ng.data.H5DataModule(args.data_path, batch_size=args.batch_size,
                              model=ng.models.NuGraph3, num_workers=0, featext=False)
    print(f"nugraph={ng.__file__}\nfixed subset: first {args.limit_batches} val batches "
          f"(bs={args.batch_size}, shuffle=False) from {args.data_path}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as out:
        out.write(f"data={args.data_path} subset=first {args.limit_batches} val batches "
                  f"(bs{args.batch_size})\n\n")
        res, ref_fp = [], None
        for lab, ck in zip(labels, ckpts):
            events, fp = collect(ck, dm, args.limit_batches, device)  # fresh pass per ckpt
            if ref_fp is None:
                ref_fp = fp
                print(f"event-identity anchor: {len(events)} events, {len(fp)} batches\n")
            elif fp != ref_fp:
                raise SystemExit(f"event mismatch for {lab} (non-deterministic val order?)")
            res.append((lab, report(lab, events, out)))
        # trajectory / comparison table
        hdr = (f"\n{'checkpoint':40s} {'fixed0.3':>9} {'best-raw':>12} "
               f"{'adaptive':>14} {'scale s':>9} {'RMS':>9}")
        print(hdr); out.write(hdr + "\n")
        for lab, r in res:
            line = (f"{lab:40.40s} {r['fixed']:>9.4f} {r['best']:>7.4f}@{r['best_eps']:<4} "
                    f"{r['abest']:>7.4f}@ρ{r['abest_rho']:<4} {r['s_med']:>9.4f} {r['rms']:>9.1f}")
            print(line); out.write(line + "\n")
        if len(res) == 2:  # early/late verdict
            e, l = res[0][1], res[1][1]
            rho_match = "SAME" if e["abest_rho"] == l["abest_rho"] else "DIFFERENT"
            v = ("METRIC/SCALE ARTIFACT" if l["best"] >= 0.9 * e["best"] or l["nbest"] >= 0.9 * e["nbest"]
                 else "GENUINE DEGRADATION" if l["best"] < 0.6 * e["best"] else "MIXTURE")
            s = f"\nAUTO-VERDICT: {v}  |  adaptive optimal ρ {rho_match} across the two"
            print(s); out.write(s + "\n")
        note = ("\nNote: if the best adaptive ρ sits at the sweep MAX (8.0), rerun with a higher "
                "RHO_SWEEP ceiling to bracket the true optimum before trusting the adaptive ARI.")
        print(note); out.write(note + "\n")


if __name__ == "__main__":
    main()
