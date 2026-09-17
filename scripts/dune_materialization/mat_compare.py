#!/usr/bin/env python
"""mat_compare.py -- group-ready comparison of mat_bench outputs (numpy only; runs on GPVM too).

  python mat_compare.py --bench DIR --canonical TAG --tuned TAG [--repeat DIR] [--out FILE]

Per configuration (all CSVs in --bench, restricted to the events of the canonical run):
quality with PAIRED differences (+- standard error) vs canonical and tuned DBSCAN, multiplicity,
failures, unassigned hits, runtime (per-event median/p95, total, speedup vs both DBSCANs; invalid
events re-timed with canonical DBSCAN for the 'with fallback' number), peak memory, and runtime
by event size. --repeat compares per-event times of identically-tagged CSVs in a second directory.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

BINS = [0, 500, 1000, 2000, 5000, 10000, 20000, 60000]


def load(path):
    return {(r["batch"], r["event"]): r for r in csv.DictReader(open(path))}


def col(rows, keys, c):
    out = []
    for k in keys:
        v = rows[k].get(c)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def paired(a, b):
    d = a - b
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan, np.nan
    return d.mean(), d.std(ddof=1) / np.sqrt(d.size)


def fmt(v, f="{:.3f}"):
    return "nan" if v is None or not np.isfinite(v) else f.format(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--canonical", default="dbscan_eps0.3_min15")
    p.add_argument("--tuned", required=True)
    p.add_argument("--repeat", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    bench = Path(a.bench)
    tabs = {q.stem: load(q) for q in sorted(bench.glob("*.csv")) if not q.stem.startswith("report")}
    can, tun = tabs[a.canonical], tabs[a.tuned]
    keys = sorted(can, key=lambda k: (int(k[0]), int(k[1])))
    missing = {t: len([k for k in keys if k not in r]) for t, r in tabs.items()}
    if any(missing.values()):
        raise SystemExit(f"configs missing canonical events: {missing}")
    L = []
    L.append(f"# Materialization comparison — {len(keys)} events, batches "
             f"{min(int(k[0]) for k in keys)}–{max(int(k[0]) for k in keys)}")
    L.append("")
    ari_c, ari_t = col(can, keys, "ari_particle"), col(tun, keys, "ari_particle")
    t_c, t_t = col(can, keys, "t_cluster_s"), col(tun, keys, "t_cluster_s")
    fg = col(can, keys, "n_foreground")

    L += ["## Quality and multiplicity", "",
          "| config | valid | edge_limit | OOM | all_noise | particle ARI (±SE) | Δ vs canonical (±SE) | "
          "Δ vs tuned (±SE) | legacy ARI | purity med | compl med | n_true med | n_pred med | "
          "pred/true med | over-seg | under-seg | unassigned mean / med |",
          "|" + "---|" * 17]
    T = ["## Runtime (clustering only) and memory", "",
         "| config | ms/event median | ms/event p95 | total s (timed) | speedup vs canonical | "
         "speedup vs tuned | total s with canonical fallback | speedup w/ fallback vs canonical | "
         "peak MB p99 | peak MB max | threads | cgroup CPUs |",
         "|" + "---|" * 12]
    for tag, r in tabs.items():
        st = [r[k]["status"] for k in keys]
        valid = np.array([r[k]["valid"] == "True" for k in keys])
        oom = sum(r[k].get("oom") == "True" for k in keys)
        pa = col(r, keys, "ari_particle")
        pa_v = np.where(valid, pa, np.nan)
        m = np.nanmean(pa_v)
        se = np.nanstd(pa_v, ddof=1) / np.sqrt(np.isfinite(pa_v).sum())
        dc, sc = paired(pa_v, ari_c)
        dt, sd = paired(pa_v, ari_t)
        npred, ntrue = col(r, keys, "n_pred"), col(r, keys, "n_true")
        ok = valid & np.isfinite(npred) & np.isfinite(ntrue) & (ntrue > 0)
        ratio = npred[ok] / ntrue[ok]
        nf = np.where(valid, col(r, keys, "noise_frac"), np.nan)
        L.append(
            f"| {tag} | {valid.sum()} | {st.count('edge_limit')} | {oom} | {st.count('all_noise')} | "
            f"{fmt(m, '{:.4f}')} ± {fmt(se, '{:.4f}')} | {fmt(dc, '{:+.4f}')} ± {fmt(sc, '{:.4f}')} | "
            f"{fmt(dt, '{:+.4f}')} ± {fmt(sd, '{:.4f}')} | "
            f"{fmt(np.nanmean(np.where(valid, col(r, keys, 'ari_nugraph'), np.nan)), '{:.4f}')} | "
            f"{fmt(np.nanmedian(np.where(valid, col(r, keys, 'purity'), np.nan)))} | "
            f"{fmt(np.nanmedian(np.where(valid, col(r, keys, 'completeness'), np.nan)))} | "
            f"{fmt(np.nanmedian(ntrue[ok]), '{:.0f}')} | {fmt(np.nanmedian(npred[ok]), '{:.0f}')} | "
            f"{fmt(np.median(ratio) if ratio.size else np.nan, '{:.2f}')} | "
            f"{fmt((ratio > 1.5).mean() if ratio.size else np.nan, '{:.2f}')} | "
            f"{fmt((ratio < 1 / 1.5).mean() if ratio.size else np.nan, '{:.2f}')} | "
            f"{fmt(np.nanmean(nf))} / {fmt(np.nanmedian(nf))} |")
        tt = col(r, keys, "t_cluster_s")
        timed = np.isfinite(tt)
        tot = tt[timed].sum()
        fb = tot + t_c[~valid].sum()
        pk = col(r, keys, "peak_alloc_mb")
        meta = Path(bench / f"{tag}.json")
        thr = cg = "?"
        if meta.exists():
            import json
            d = json.loads(meta.read_text())
            thr, cg = d.get("cpu_threads", "?"), d.get("cgroup_cpu_max", "?")
        T.append(
            f"| {tag} | {fmt(1e3 * np.median(tt[timed]), '{:.2f}')} | "
            f"{fmt(1e3 * np.percentile(tt[timed], 95), '{:.1f}')} | {tot:.1f} | "
            f"{fmt(t_c[timed].sum() / tot if tot else np.nan, '{:.1f}')}× | "
            f"{fmt(t_t[timed].sum() / tot if tot else np.nan, '{:.1f}')}× | {fb:.1f} | "
            f"{fmt(t_c.sum() / fb if fb else np.nan, '{:.1f}')}× | "
            f"{fmt(np.nanpercentile(pk, 99) if np.isfinite(pk).any() else np.nan, '{:.0f}')} | "
            f"{fmt(np.nanmax(pk) if np.isfinite(pk).any() else np.nan, '{:.0f}')} | {thr} | {cg} |")
    L += [""] + T + [""]

    L += ["## Median clustering time by event size (ms; foreground hits)", "",
          "| n_fg | events | " + " | ".join(tabs) + " |", "|" + "---|" * (len(tabs) + 2)]
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (fg >= lo) & (fg < hi)
        if not sel.any():
            continue
        cells = []
        for tag, r in tabs.items():
            tt = col(r, keys, "t_cluster_s")[sel]
            vv = np.array([r[k]["valid"] == "True" for k in keys])[sel]
            txt = fmt(1e3 * np.nanmedian(tt), "{:.1f}") if np.isfinite(tt).any() else "nan"
            if (~vv).any():
                txt += f" ({(~vv).mean():.0%} inv)"
            cells.append(txt)
        L.append(f"| {lo}–{hi} | {sel.sum()} | " + " | ".join(cells) + " |")
    L.append("")

    if a.repeat:
        L += ["## Timing repeatability (same config, separate solo run)", "",
              "| config | events | median ms run1 / run2 | per-event ratio median (p10–p90) | total ratio |",
              "|---|---|---|---|---|"]
        for q in sorted(Path(a.repeat).glob("*.csv")):
            if q.stem not in tabs:
                continue
            r2 = load(q)
            ks = [k for k in keys if k in r2]
            x, y = col(tabs[q.stem], ks, "t_cluster_s"), col(r2, ks, "t_cluster_s")
            ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
            rr = x[ok] / y[ok]
            L.append(f"| {q.stem} | {ok.sum()} | {1e3*np.median(x[ok]):.2f} / {1e3*np.median(y[ok]):.2f} | "
                     f"{np.median(rr):.3f} ({np.percentile(rr,10):.3f}–{np.percentile(rr,90):.3f}) | "
                     f"{x[ok].sum()/y[ok].sum():.3f} |")
        L.append("")
    L += ["Particle ARI = ARI over predicted-foreground, labelled hits that belong to a true particle "
          "(invalid events excluded; paired Δ uses the same events). Over/under-seg: n_pred/n_true "
          ">1.5 / <1/1.5. Speedups = ratio of summed clustering-only time over the events the config "
          "timed; 'with fallback' re-times invalid events with canonical DBSCAN."]
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
