"""Tests for the coverage-anatomy numpy core and report (scripts/particle_hierarchy/coverage_anatomy.py).

Numpy + csv only (no torch, no pytest import), so they also run on a machine without torch.
Run:  pytest -q tests/test_coverage_anatomy.py
"""
import csv
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("coverage_anatomy", ROOT / "scripts/particle_hierarchy/coverage_anatomy.py")
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)


def _within(P, t_d):
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    return d <= t_d


def _outranks(bS, j, i):
    return bS[j] > bS[i] or (bS[j] == bS[i] and j < i)


def _brute_deployed(W, bS):
    k = bS.size
    dom = np.zeros(k, bool)
    dtor = np.full(k, -1)
    for i in range(k):
        cands = [j for j in range(k) if j != i and W[i, j] and _outranks(bS, j, i)]
        if cands:
            dom[i] = True
            best = cands[0]
            for j in cands[1:]:
                if _outranks(bS, j, best):
                    best = j
            dtor[i] = best
    return dom, dtor


def _brute_greedy(W, bS):
    k = bS.size
    order = sorted(range(k), key=lambda i: (-bS[i], i))
    keep = []
    for i in order:
        if not any(W[i, j] for j in keep):
            keep.append(i)
    out = np.zeros(k, bool)
    out[keep] = True
    return out


def _random_case(seed, k=60, ties=False):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 2, size=(k, 3))
    b = rng.uniform(0.5, 1.0, size=k)
    if ties:
        b = np.round(b, 1)
    return P, b.astype(np.float32)


def test_deployed_matches_brute_force_including_ties_and_chunks():
    for seed in range(12):
        P, b = _random_case(seed, ties=seed % 2 == 0)
        W = _within(P, 0.4)
        for chunk in (1, 7, 1024):
            dom, dtor = ca.deployed_from_within(W, b, chunk_rows=chunk)
            rdom, rdtor = _brute_deployed(W, b)
            assert (dom == rdom).all() and (dtor == rdtor).all()


def test_greedy_matches_brute_force_and_contains_deployed_survivors():
    for seed in range(12):
        P, b = _random_case(seed, ties=seed % 3 == 0)
        W = _within(P, 0.4)
        g = ca.greedy_from_within(W, b)
        assert (g == _brute_greedy(W, b)).all()
        dom, _ = ca.deployed_from_within(W, b)
        assert (g | dom).all() and (g[~dom]).all()          # every deployed survivor is kept by greedy
        # greedy output is itself an NMS-valid set: no two kept seeds within t_d
        kept = np.nonzero(g)[0]
        assert not (W[np.ix_(kept, kept)] & ~np.eye(kept.size, dtype=bool)).any()


def _chain_event():
    """Seeds on a line: A(0.9)@0, B(0.8)@0.25, C(0.7)@0.5, t_d 0.3. B is suppressed by A (direct), C by B, which is
    itself suppressed (chain). Particle 4 has hits only below tau; particle 5 is suppressed by a spurious seed."""
    pos = np.array([[0.0], [0.25], [0.5], [3.0], [3.0], [6.0], [6.1], [9.0]])
    beta = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.95, 0.6, 0.3], np.float32)
    y = np.array([1, 2, 3, 4, 4, -1, 5, 1])
    pop = y >= 0
    return pos, beta, y, pop


def test_chain_direct_spurious_threshold_causes():
    pos, beta, y, pop = _chain_event()
    tau = 0.5
    seed_list = np.nonzero(beta > tau)[0]                   # A B C spurious(5) particle5(6)
    W = _within(pos[seed_list], 0.3)
    dom, dtor = ca.deployed_from_within(W, beta[seed_list])
    surv = ~dom
    assert set(seed_list[surv].tolist()) == {0, 5}
    seeded = set(y[seed_list[surv]][y[seed_list[surv]] >= 0].tolist())
    causes = ca.classify_seedless(y, pop, beta, tau, seed_list, dom, dtor, surv, seeded)
    assert causes[2]["cause"] == "supp_foreign_direct" and causes[2]["suppressor_particle"] == 1
    assert causes[3]["cause"] == "supp_foreign_chain" and causes[3]["suppressor_particle"] == 2
    assert causes[4]["cause"] == "sub" and math.isclose(causes[4]["gap"], 0.5 - 0.1, rel_tol=1e-6)
    assert causes[5]["cause"] == "supp_spurious_direct"
    assert 1 not in causes                                  # seeded particle is not classified
    g = seed_list[ca.greedy_from_within(W, beta[seed_list])]
    assert 3 in set(y[g].tolist()) and 2 not in set(y[g].tolist())   # greedy recovers only the chain loss


def test_cap_and_unattributed_causes():
    pos, beta, y, pop = _chain_event()
    tau = 0.5
    seed_list = np.array([0, 5])                            # pretend the cap dropped B, C and particle 5's seed
    causes = ca.classify_seedless(y, pop, beta, tau, seed_list, np.array([False, False]), np.array([-1, -1]),
                                  np.array([True, True]), {1})
    assert causes[2]["cause"] == causes[3]["cause"] == causes[5]["cause"] == "cap"
    full = np.nonzero(beta > tau)[0]
    causes = ca.classify_seedless(y, pop, beta, tau, full, None, None, None, {1})
    assert causes[2]["cause"] == "unattributed" and causes[4]["cause"] == "sub"


def test_set_stats_counts_and_oracle_labels():
    # population: particle 1 (hits 0,1,2), particle 2 (hits 3,4), particle 3 (hit 5, seedless)
    y = np.array([1, 1, 1, 2, 2, 3, -1])
    pop = np.array([1, 1, 1, 1, 1, 1, 0], bool)
    beta = np.array([0.9, 0.2, 0.1, 0.8, 0.85, 0.1, 0.7], np.float32)
    cand = np.array([0, 3, 4, 6])                           # particle 1, particle 2 (twice: dup), spurious
    # nearest-candidate orderings (positions into cand) for the 6 population hits
    topm = np.array([[0, 1, 2, 3], [3, 0, 1, 2], [1, 2, 3, 0], [2, 1, 0, 3], [3, 1, 2, 0], [0, 1, 2, 3]])
    st, h = ca.set_stats(cand, y, pop, beta, topm, m_reach=2, m_own5=1)
    assert st["K"] == 4 and st["n_pop"] == 6 and st["n_dup"] == 1 and st["n_spur"] == 1
    assert st["n_seedless"] == 1                             # hit 5
    assert h["reach"].tolist() == [True, True, False, True, True, False]
    assert st["n_oor"] == 1 and st["n_reach"] == 4 and st["n_own5"] == 2       # own nearest: hits 0, 3
    assert st["n_seedless"] + st["n_oor"] + st["n_reach"] == st["n_pop"]
    # primary of particle 2 = cand position 2 (beta 0.85); hit 3's top-2 contains it
    assert h["lab_or20"].tolist() == [0, 0, 1, 2, 3, 0]       # hit 4: primary not in its top-2 -> nearest
    assert h["lab_all"].tolist() == [0, 0, 0, 2, 2, 0]
    assert h["lab_dec"].tolist() == [0, 3, 1, 2, 3, 0]


def test_empty_set_is_all_seedless():
    y = np.array([1, 2]); pop = np.array([True, True]); beta = np.array([0.1, 0.2], np.float32)
    st, h = ca.set_stats(np.zeros(0, np.int64), y, pop, beta, np.zeros((2, 0), np.int64))
    assert st["n_seedless"] == 2 and h["seedless"].all()


def test_oracle_seeds_pick_highest_beta_population_hit():
    y = np.array([1, 1, 2, 2, -1]); pop = np.array([1, 1, 1, 0, 1], bool)
    beta = np.array([0.3, 0.5, 0.2, 0.9, 0.99], np.float32)
    assert ca.oracle_seeds(y, pop, beta) == {1: 1, 2: 2}     # hit 3 is not population; hit 4 has no particle


def test_readings():
    base = {"seedless": 0.6, "oor": 0.03}
    assert ca.reading({**base, "seedless_supp": 0.4, "seedless_sub": 0.2})[0] == "SUPPRESSION"
    assert ca.reading({**base, "seedless_supp": 0.1, "seedless_sub": 0.5})[0] == "THRESHOLD"
    assert ca.reading({"seedless": 0.01, "oor": 0.5, "seedless_supp": 0.0, "seedless_sub": 0.0})[0] == "REACH"
    assert ca.reading({**base, "seedless_supp": 0.3, "seedless_sub": 0.3})[0] == "MIXED"
    assert ca.combine_readings(["THRESHOLD", "THRESHOLD"]) == "THRESHOLD"
    assert ca.combine_readings(["THRESHOLD", "SUPPRESSION"]) == "MIXED"


# ----------------------------------------------------------------------------- report on synthetic runs
def _write_run(root, name, n_ev=20, sub=30, supp=15, a2_bad=0, a1_fail=0, anchor=False, bad_book=0):
    """Synthetic run dir. Categories are consistent with FROZEN (A4); anchor=True reproduces the Stage-0 anchors."""
    d = Path(root) / name
    d.mkdir(parents=True)
    npop, oor = (1000, 0) if anchor else (100, 3)
    if anchor:
        sub, supp = 400, 195                                  # FROZEN seedless 0.595
    rows = []
    for e in range(n_ev):
        cats = {c: 0 for c in ca.CATS}
        cats["reach"], cats["oor"] = npop - sub - supp - oor, oor
        cats["seedless_sub"], cats["seedless_supp_foreign_direct"] = sub, supp
        r = {"batch": 0, "event": e, "n_pop": npop, "attributable": 1, "a2_ok": 0 if e < a2_bad else 1,
             "greedy.supp_hits": supp, "greedy.supp_hits_recovered": supp // 3,
             "rep.seedless_nearest_own_oracle_seeds": (sub + supp) // 4,
             "rep.seedless_nearest_own_oracle_add": (sub + supp) // 3, "rep.all_nearest_own_oracle_seeds": 40}
        for c in ca.CATS:
            r[f"cat.{c}"] = cats[c] + (1 if (c == "reach" and e < bad_book) else 0)
        for s in ca.SETS:
            sl = sub + supp if s != "LOWTAU" else (336 if anchor else sub)
            r.update({f"{s}.K": 10 if s != "GREEDY" else 12, f"{s}.n_pop": npop, f"{s}.n_seedless": sl,
                      f"{s}.n_oor": oor, f"{s}.n_reach": npop - sl - oor,
                      f"{s}.n_own5": 286 if anchor else 20, f"{s}.n_dup": 1, f"{s}.n_spur": 0,
                      f"{s}.ari_dec": 0.167 if anchor else 0.2, f"{s}.ari_or20": 0.3,
                      f"{s}.ari_all": (0.620 if s == "FROZEN" else 0.781) if anchor else 0.5})
        rows.append(r)
    keys = sorted({k for r in rows for k in r})
    with open(d / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (d / "meta.json").write_text(json.dumps({"a1_checked": 20, "a1_failed": a1_fail}))
    (d / "groups.json").write_text(json.dumps({"sem:1": {c: 1 for c in ca.CATS}, "size:3": {c: 2 for c in ca.CATS}}))
    return str(d)


class _A:
    pass


def _report(primary, anchor=None, descriptive=()):
    out = Path(tempfile.mkdtemp()) / "rep"
    a = _A()
    a.primary, a.anchor, a.descriptive, a.out = primary, anchor, list(descriptive), str(out)
    ca.cmd_report(a)
    return json.loads((out / "report.json").read_text()), (out / "report.md").read_text()


def test_report_reading_and_gates_pass():
    tmp = tempfile.mkdtemp()
    p1 = _write_run(tmp, "T_s1")
    p2 = _write_run(tmp, "T_s2")
    anc = _write_run(tmp, "ep29", anchor=True)
    rep, md = _report([p1, p2], anchor=anc)
    assert rep["gates_pass"] and rep["anchor"]["pass"]
    assert rep["reading"] == "THRESHOLD"                     # sub 30 / U 48 = 0.625
    assert "REPRESENTATION-LIMITED" in md
    s = rep["runs"]["T_s1"]
    assert math.isclose(s["cats"]["seedless"], 0.45) and math.isclose(s["fractions_of_U"]["sub"], 30 / 48)
    assert s["greedy_recovers"] and math.isclose(s["greedy_K_ratio"], 1.2)   # 5/15 >= 0.25 and K ratio <= 1.5


def test_report_greedy_flag_and_disagreeing_seeds_and_gate_failures():
    tmp = tempfile.mkdtemp()
    p1 = _write_run(tmp, "T_s1", sub=10, supp=40)            # suppression-dominated
    p2 = _write_run(tmp, "T_s2", sub=40, supp=10)            # threshold-dominated
    rep, _ = _report([p1, p2])
    assert rep["reading"] == "MIXED"
    assert rep["runs"]["T_s1"]["greedy_recovers"]           # 13/40 = 0.325 >= 0.25, K ratio 1.2 <= 1.5
    for kw, gate in (({"a2_bad": 3}, "A2"), ({"a1_fail": 1}, "A1"), ({"bad_book": 1}, "A4")):
        tmp = tempfile.mkdtemp()
        rep, md = _report([_write_run(tmp, "T_s1", **kw), _write_run(tmp, "T_s2")])
        assert not rep["gates_pass"] and "NOT READ" in md, gate
    tmp = tempfile.mkdtemp()
    rep, _ = _report([_write_run(tmp, "T_s1", a2_bad=2), _write_run(tmp, "T_s2")])
    assert rep["gates_pass"]                                  # <= 2 mismatching events tolerated


def test_report_anchor_failure_blocks_reading():
    tmp = tempfile.mkdtemp()
    rep, md = _report([_write_run(tmp, "T_s1"), _write_run(tmp, "T_s2")], anchor=_write_run(tmp, "ep29"))
    assert not rep["anchor"]["pass"] and not rep["gates_pass"] and "NOT READ" in md


if __name__ == "__main__":                                   # plain-python runner (no pytest available)
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:                           # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
