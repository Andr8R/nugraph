#!/usr/bin/env python
"""
READ-ONLY diagnostic: reconcile the standalone benchmark instance ARI with the
native NuGraph3 InstanceDecoder validation ARI and the epoch-19 TensorBoard value.

It does NOT modify the checkpoint, HDF5, model parameters, benchmark results,
training logs, or any source file. It only prints diagnostics and writes two CSVs
under ``--out-prefix``.

It answers, empirically, the g/h/i decision:
  A  native decoder ARI (from the normal eval forward's E_H_IP)
  B  manual decoder-compatible ARI on the SAME native E_H_IP
     (adjusted_rand_score(x_i[mask], y_i[mask]), mask = y_semantic>=0)
  C  standalone benchmark ari_inclusive (fresh DBSCAN materialization)
  D  standalone benchmark ari_particle
plus partition-equality of native x_i vs standalone labels, a full config dump,
a cumulative validation trend of B, and the TensorBoard epoch-19 scalar.

Only the original/default DBSCAN path is used. No seed/graph/greedy, no sweeps.

Usage (EAF):
  python scripts/diagnose_instance_ari.py --device 0 --checkpoint $CKPT \
    --data-path $DATA --event --semantic --filter --instance --particle-loss \
    --featext --in-feats 11 --no-lr-scheduler --num-workers 0 --batch-size 8 \
    --out-prefix logs/materialization_eval/diag_epoch19 \
    --tb-run <run_dir_containing_events.out.tfevents> --limit-batches 0
"""
import argparse
import csv
import os
import statistics
from itertools import islice

import torch
from sklearn.metrics import adjusted_rand_score

import nugraph as ng
from nugraph.models.nugraph3.materialize import materialize_labels
from nugraph.models.nugraph3.types import E_H_IP

# reuse the benchmark's exact metric + foreground definitions (read-only import)
import importlib.util
_bench_path = os.path.join(os.path.dirname(__file__), "materialize_benchmark.py")
_spec = importlib.util.spec_from_file_location("materialize_benchmark", _bench_path)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)


def configure():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--split", default="validation",
                   choices=("train", "validation", "test"))
    p.add_argument("--limit-batches", type=int, default=0,
                   help="cap trend batches (0 = full validation split)")
    p.add_argument("--out-prefix", default="diag_instance_ari")
    p.add_argument("--tb-run", default=None,
                   help="directory containing events.out.tfevents for TB read")
    p = ng.data.H5DataModule.add_data_args(p)
    p = ng.models.NuGraph3.add_model_args(p)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def per_event_ari_from_native(data):
    """B: per-event decoder-compatible ARI over the native E_H_IP.

    Returns a list of dicts (one per event) with the ARI (mask y_semantic>=0) and
    diagnostic counts, using the decoder-built predicted edges already on ``data``.
    """
    out = []
    for l in data.to_data_list():
        ys = l["hit"].y_semantic
        m = ys >= 0
        xi, yi = l.x_i(), l.y_i()
        ari = float(adjusted_rand_score(xi[m].cpu().numpy(),
                                        yi[m].cpu().numpy())) if int(m.sum()) >= 2 \
            else float("nan")
        n_pred = int(len({int(x) for x in xi[m].tolist() if x >= 0}))
        n_true = int(len({int(x) for x in yi[m].tolist() if x >= 0}))
        status = "all_noise" if n_pred == 0 else "ok"
        out.append(dict(
            ari=ari, status=status, n_hits=int(ys.numel()),
            n_yi_neg1=int((yi == -1).sum()), n_yi_pos=int((yi >= 0).sum()),
            n_pred=n_pred, n_true=n_true))
    return out


# --------------------------------------------------------------------------- #
# 6. configuration dump
# --------------------------------------------------------------------------- #
def dump_config(model, ckpt_path, data0):
    raw = torch.load(ckpt_path, map_location="cpu")
    hp = raw.get("hyper_parameters", {})
    sd = raw.get("state_dict", {})
    msd = model.state_dict()
    missing = [k for k in msd if k not in sd]
    unexpected = [k for k in sd if k not in msd]
    dec = getattr(model, "instance_decoder", None)
    inorm = getattr(getattr(model, "encoder", None), "input_norm", None)
    print("\n===== 6. CONFIGURATION DUMP =====")
    print(f"checkpoint            : {ckpt_path}")
    print(f"ckpt epoch            : {raw.get('epoch')}")
    print(f"ckpt global_step      : {raw.get('global_step')}")
    print(f"model.training        : {model.training}")
    for k in ("in_features", "hit_features", "nexus_features",
              "interaction_features", "instance_features", "num_iters",
              "event_head", "semantic_head", "filter_head", "instance_head",
              "vertex_head", "spacepoint_head", "particle_loss"):
        if k in hp:
            print(f"hparam {k:22s}: {hp[k]}")
    if dec is not None:
        print(f"decoder.materializer  : {getattr(dec, 'materializer', '?')}")
        print(f"decoder.dbscan eps/min: {dec.dbscan.eps} / {dec.dbscan.min_samples}")
        print(f"decoder.mat_train     : {getattr(dec, 'materialize_train', '?')}")
    if inorm is not None:
        rm = getattr(inorm, "mean", None)
        rs = getattr(inorm, "std", None)
        print(f"input_norm.update     : {getattr(inorm, 'update', '?')}")
        if isinstance(rm, torch.Tensor):
            print(f"input_norm.mean[:4]   : {rm.flatten()[:4].tolist()}")
        if isinstance(rs, torch.Tensor):
            print(f"input_norm.std[:4]    : {rs.flatten()[:4].tolist()}")
    print(f"param dtype           : {next(model.parameters()).dtype}")
    print(f"input hit.x dtype     : {data0['hit'].x.dtype}")
    print(f"autocast enabled(cuda): {torch.is_autocast_enabled()}")
    print(f"missing keys ({len(missing)}) : {missing[:8]}")
    print(f"unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    if not missing and not unexpected:
        print("=> checkpoint loaded cleanly; no silent default substitution")


# --------------------------------------------------------------------------- #
# 4/5. first-batch A/B/C/D + partition equality
# --------------------------------------------------------------------------- #
def first_batch_abcd(model, data, dbscan):
    print("\n===== 4/5. FIRST-BATCH A/B/C/D =====")
    # single forward -> populates ox/of/x_filter/x_semantic and native E_H_IP
    with torch.no_grad():
        _, metrics = model(data, stage="val")
    A_mean = float(metrics.get("instance/adjusted-rand-val", float("nan")))
    h = data["hit"]

    # native predicted per-hit labels from the decoder's E_H_IP
    x_i_native = data.x_i().clone()
    y_i = data.y_i()

    # B: manual decoder-compatible per-event ARI on the native E_H_IP
    B = per_event_ari_from_native(data)

    # C/D: standalone benchmark path -- a FRESH DBSCAN materialization on the
    # same cached ox/beta and the same foreground mask
    ox, beta = h.ox.detach(), h.of.detach()
    fg = mb.foreground_mask(h)
    ng_ = data.num_graphs
    labels_std, stats = materialize_labels(
        "dbscan", ox, beta, fg, h.ptr, ng_, dbscan=dbscan)

    # mask equality (decoder foreground vs benchmark foreground; both identical code)
    dec_fg = model.instance_decoder._foreground_mask(h)  # noqa: SLF001 (read-only)
    fg_equal = bool(torch.equal(dec_fg, fg))
    yi_equal = bool(torch.equal(y_i, data.y_i()))  # benchmark truth == native y_i

    header = ("ev n_hits n_fg  n_yi-1 n_yi>=0 nPredA nPredC     A(mean)      B"
              "         C         D  statA  statC  partition_ARI  raw_eq")
    print(header)
    rows_first = []
    for g in range(ng_):
        gm = h.batch == g
        ys_g = h.y_semantic[gm]
        keep_particle_g = fg[gm] & (ys_g >= 0)
        q = mb.event_quality(labels_std[gm], y_i[gm], ys_g, keep_particle_g)
        # partition ARI: native x_i vs standalone labels over all event hits
        xi_g = x_i_native[gm]
        lb_g = labels_std[gm]
        part = float(adjusted_rand_score(xi_g.cpu().numpy(), lb_g.cpu().numpy()))
        raw_eq = float((xi_g == lb_g).float().mean())
        b = B[g]
        print(f"{g:2d} {b['n_hits']:6d} {int((fg[gm]).sum()):4d} "
              f"{b['n_yi_neg1']:6d} {b['n_yi_pos']:7d} {b['n_pred']:6d} "
              f"{q['n_pred'] if q['n_pred']==q['n_pred'] else -1!s:>6} "
              f"{A_mean:10.4f} {b['ari']:9.4f} {q['ari_inclusive']:9.4f} "
              f"{q['ari_particle']:9.4f} {b['status']:>6} {stats[g].status:>6} "
              f"{part:13.4f} {raw_eq:6.3f}")
        rows_first.append(dict(event=g, **b, ari_inclusive=q["ari_inclusive"],
                               ari_particle=q["ari_particle"],
                               standalone_status=stats[g].status,
                               partition_ari=part, raw_eq=raw_eq))
    print(f"\nA (decoder batch mean)      : {A_mean:.5f}")
    Bvals = [r['ari'] for r in B if r['ari'] == r['ari']]
    print(f"mean(B) over events         : {statistics.mean(Bvals) if Bvals else float('nan'):.5f}")
    print(f"foreground mask equal (dec vs bench): {fg_equal}")
    print(f"benchmark truth == native y_i       : {yi_equal}")
    print(f"semantic mask: both (y_semantic>=0)  : identical by construction")
    part_all = [r['partition_ari'] for r in rows_first]
    print(f"partition ARI (native x_i vs standalone labels): "
          f"min={min(part_all):.4f} mean={statistics.mean(part_all):.4f}")
    return rows_first


# --------------------------------------------------------------------------- #
# 7. cumulative validation trend of B
# --------------------------------------------------------------------------- #
def trend(model, loader, out_prefix, limit):
    print("\n===== 7. CUMULATIVE VALIDATION TREND (native decoder metric B) =====")
    checkpoints = [1, 2, 10, 50, 100]
    per_event_rows = []
    cum_rows = []
    batch_means = []
    n_batches = 0
    it = loader if limit == 0 else islice(loader, limit)
    device = next(model.parameters()).device

    def snapshot():
        aris = [r["ari"] for r in per_event_rows if r["ari"] == r["ari"]]
        ff = [r["foreground_fraction"] for r in per_event_rows]
        an = sum(1 for r in per_event_rows if r["status"] == "all_noise")
        return dict(
            n_batches=n_batches, n_events=len(per_event_rows),
            mean_event_ari=statistics.mean(aris) if aris else float("nan"),
            median_event_ari=statistics.median(aris) if aris else float("nan"),
            mean_of_batch_means=statistics.mean(batch_means) if batch_means else float("nan"),
            all_noise_fraction=an / len(per_event_rows) if per_event_rows else float("nan"),
            mean_fg_fraction=statistics.mean(ff) if ff else float("nan"),
            median_fg_fraction=statistics.median(ff) if ff else float("nan"),
            mean_n_pred=statistics.mean([r["n_pred"] for r in per_event_rows]) if per_event_rows else float("nan"),
            mean_n_true=statistics.mean([r["n_true"] for r in per_event_rows]) if per_event_rows else float("nan"))

    for bi, batch in enumerate(it):
        data = batch.to(device)
        with torch.no_grad():
            model(data, stage="val")  # native forward builds E_H_IP
        h = data["hit"]
        fg = mb.foreground_mask(h)
        B = per_event_ari_from_native(data)
        bmeans = []
        for g, b in enumerate(B):
            gm = h.batch == g
            nfg = int(fg[gm].sum())
            row = dict(batch=bi, event=g, ari=b["ari"], status=b["status"],
                       n_hits=b["n_hits"], n_foreground=nfg,
                       foreground_fraction=nfg / b["n_hits"] if b["n_hits"] else 0.0,
                       n_pred=b["n_pred"], n_true=b["n_true"])
            per_event_rows.append(row)
            if b["ari"] == b["ari"]:
                bmeans.append(b["ari"])
        if bmeans:
            batch_means.append(statistics.mean(bmeans))
        n_batches = bi + 1
        if n_batches in checkpoints:
            cum_rows.append(snapshot())
    cum_rows.append(snapshot())  # full (or limit)

    # save CSVs
    with open(f"{out_prefix}_per_event.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_event_rows[0].keys()))
        w.writeheader()
        w.writerows(per_event_rows)
    with open(f"{out_prefix}_cumulative.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cum_rows[0].keys()))
        w.writeheader()
        w.writerows(cum_rows)

    hdr = ("nB   nEv   mean_ari  median_ari  mean_batchmeans  allnoise  "
           "mean_fgfrac  mean_nPred  mean_nTrue")
    print(hdr)
    for c in cum_rows:
        print(f"{c['n_batches']:<4d} {c['n_events']:<5d} "
              f"{c['mean_event_ari']:9.4f} {c['median_event_ari']:10.4f} "
              f"{c['mean_of_batch_means']:15.4f} {c['all_noise_fraction']:8.3f} "
              f"{c['mean_fg_fraction']:11.3f} {c['mean_n_pred']:10.2f} "
              f"{c['mean_n_true']:10.2f}")
    print(f"\nwrote {out_prefix}_per_event.csv and {out_prefix}_cumulative.csv")
    return cum_rows[-1]


# --------------------------------------------------------------------------- #
# 8. TensorBoard scalar
# --------------------------------------------------------------------------- #
def read_tb(tb_run):
    print("\n===== 8. TENSORBOARD instance/adjusted-rand-val =====")
    if not tb_run:
        print("(--tb-run not given; skipping)")
        return
    try:
        from tensorboard.backend.event_processing.event_accumulator import \
            EventAccumulator
        import glob
        efs = sorted(glob.glob(os.path.join(tb_run, "**", "events.out.tfevents*"),
                               recursive=True))
        ea = EventAccumulator(os.path.dirname(efs[-1]))
        ea.Reload()
        tag = "instance/adjusted-rand-val"
        if tag not in ea.Tags().get("scalars", []):
            print(f"tag {tag} not found; available:",
                  [t for t in ea.Tags()["scalars"] if "instance" in t])
            return
        sc = ea.Scalars(tag)
        print(f"n logged points       : {len(sc)}")
        print(f"first                 : step={sc[0].step} value={sc[0].value:.5f}")
        print(f"last (epoch 19)       : step={sc[-1].step} value={sc[-1].value:.5f}")
        print("Lightning reduction: val metric logged via log_dict(batch_size="
              "num_graphs); with _rand_score = per-graph mean, the epoch scalar is"
              " the event-weighted mean == arithmetic per-event mean.")
    except Exception as e:  # pragma: no cover
        print("TB read failed:", e)


def main():
    args = configure()
    nudata = ng.data.H5DataModule(
        args.data_path, batch_size=args.batch_size, model=ng.models.NuGraph3,
        shuffle="random", num_workers=args.num_workers, featext=args.featext)
    model = ng.models.NuGraph3.load_from_checkpoint(args.checkpoint)
    accelerator, _ = ng.util.configure_device(args.device)
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")
    model = model.to(device).eval()
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)

    loader = {"train": nudata.train_dataloader, "validation": nudata.val_dataloader,
              "test": nudata.test_dataloader}[args.split]()

    first = next(iter(loader))
    data0 = first.to(device)
    dump_config(model, args.checkpoint, data0)
    first_batch_abcd(model, data0, dbscan)

    # fresh loader for the trend (deterministic order for val)
    loader2 = {"train": nudata.train_dataloader, "validation": nudata.val_dataloader,
               "test": nudata.test_dataloader}[args.split]()
    full = trend(model, loader2, args.out_prefix, args.limit_batches)
    read_tb(args.tb_run)

    print("\n===== 9. DECISION INPUTS =====")
    print(f"full-val arithmetic per-event mean B : {full['mean_event_ari']:.5f}")
    print(f"full-val mean of batch means         : {full['mean_of_batch_means']:.5f}")
    print("Compare these to the TB epoch-19 value above to classify g/h/i.")


if __name__ == "__main__":
    main()
