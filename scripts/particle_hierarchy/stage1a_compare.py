#!/usr/bin/env python
"""stage1a_compare.py -- paired comparison of Stage-1a per-event CSVs (numpy only).

Each input is LABEL=PATH to a per-event CSV from `soft_particle.py eval` (<tag>_events.csv) or
`diagnose` (diagnose_events.csv, with --variant to pick one decode). Events are paired on
(batch, event). The reference is the first input unless --ref names another label.

    python stage1a_compare.py --ref m0.5 m0.3=a.csv m0.5=b.csv m0.7=c.csv
"""
import argparse
import csv
import math

import numpy as np


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load(path, variant=None):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if variant and r.get("variant") != variant:
                continue
            n_true = _f(r.get("n_true", r.get("n_true_all")))
            out[(r["batch"], r["event"])] = {
                "ari": _f(r["ari_particle"]),
                "hard": _f(r.get("ari_particle_hardseed", r.get("ari_hardseed"))),
                "ratio": _f(r.get("n_pred")) / n_true if n_true and n_true > 0 else math.nan,
                "groups": _f(r.get("n_groups"))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="LABEL=PATH")
    p.add_argument("--ref", default=None)
    p.add_argument("--variant", default=None, help="diagnose CSVs: decode variant to use")
    a = p.parse_args()
    data = {}
    for spec in a.inputs:
        label, _, path = spec.partition("=")
        data[label] = load(path, a.variant)
    ref = a.ref or next(iter(data))
    keys = set.intersection(*(set(d) for d in data.values()))
    print(f"paired events: {len(keys)}   reference: {ref}")
    print(f"{'label':14s} {'ARI':>7s} {'d_vs_ref':>9s} {'+-SE':>7s} {'d_vs_hard':>10s} {'+-SE':>7s} "
          f"{'pred/true':>9s} {'no_cl':>6s}")
    for label, d in data.items():
        ari = np.array([d[k]["ari"] for k in keys])
        rv = np.array([data[ref][k]["ari"] for k in keys])
        hard = np.array([d[k]["hard"] for k in keys])
        cols = []
        for other in (rv, hard):
            ok = np.isfinite(ari) & np.isfinite(other)
            dl = ari[ok] - other[ok]
            se = dl.std(ddof=1) / math.sqrt(dl.size) if dl.size > 1 else math.nan
            cols += [dl.mean() if dl.size else math.nan, se]
        ratio = np.array([d[k]["ratio"] for k in keys])
        groups = np.array([d[k]["groups"] for k in keys])
        print(f"{label:14s} {np.nanmean(ari):7.4f} {cols[0]:+9.4f} {cols[1]:7.4f} {cols[2]:+10.4f} "
              f"{cols[3]:7.4f} {np.nanmedian(ratio):9.2f} {np.mean(groups == 0):6.3f}")


if __name__ == "__main__":
    main()
