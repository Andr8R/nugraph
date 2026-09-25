"""Tests for Stage 1a-R7 (scripts/particle_hierarchy/sem_consistency.py): consistency loss, monitoring counts,
the Step 0 audit maths and GO/STOP rule, and the pilot readings.

numpy / pure-python parts run anywhere; the loss and monitoring tests need torch (EAF).
Run:  pytest -q tests/test_sem_consistency.py
"""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sem_consistency", ROOT / "scripts/particle_hierarchy/sem_consistency.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


# ---- audit maths (numpy) ------------------------------------------------------------------- #
def test_audit_event_on_a_hand_computed_example():
    pid = np.array([1, 1, 1, 2, 2, 3])
    pcls = np.array([0, 0, 1, 4, 4, 2])
    tcls = np.array([0, 0, 0, 4, 4, 2])
    seed = np.array([True, False, False, True, False, True])
    e = sc.audit_event(pid, pcls, tcls, seed)
    assert (e["pairs_dis"], e["pairs_tot"]) == (4.0, 8.0)                    # particle 1: 6 ordered pairs, 4 disagree
    assert e["part_dis_sum"] == pytest.approx(4 / 6) and e["n_part2"] == 2
    assert (e["n"], e["n_ok"]) == (6, 5)
    assert (e["seed_n"], e["seed_cons"], e["seed_ok"]) == (2, 2, 2)          # the single-hit particle is excluded
    assert (e["other_n"], e["other_cons"]) == (3, 2)
    assert e["cls_n"][0] == 3 and e["cls_ok"][0] == 2
    assert (e["cls_pairs_dis"][0], e["cls_pairs_tot"][0]) == (4.0, 6.0)
    s = sc.summarise(sc.merge({}, e))
    assert s["pair_disagree"] == 0.5 and s["particle_mean_disagree"] == pytest.approx(1 / 3)
    assert s["seed_consensus_agree"] == 1.0 and s["other_consensus_agree"] == pytest.approx(2 / 3)


def test_audit_event_with_no_labelled_hits():
    e = sc.audit_event(np.array([], int), np.array([], int), np.array([], int), np.array([], bool))
    assert e["pairs_tot"] == 0 and e["n"] == 0


def aud(d):
    return {"pair_disagree": d, "acc": 0.8}


@pytest.mark.parametrize("ep29, t1, t2, go, overlap", [
    (0.20, 0.14, 0.15, True, True),       # T already cuts inconsistency by >= 25 % on both seeds
    (0.20, 0.18, 0.14, True, False),      # only one seed
    (0.05, 0.20, 0.20, False, False),     # too little inconsistency in ep29 to be the lever
])
def test_step0_rule_fixed_in_advance(ep29, t1, t2, go, overlap):
    d = sc.audit_decision({"ep29": aud(ep29), "C_s1": aud(0.20), "C_s2": aud(0.20), "T_s1": aud(t1), "T_s2": aud(t2)})
    assert d["go"] is go and d["overlap_with_stage1b"] is overlap
    assert d["reading"] == ("GO" if go else "STOP")


def test_audit_report_stops_the_pipeline_on_stop(tmp_path):
    for t, v in {"ep29": 0.05, "C_s1": 0.2, "C_s2": 0.2, "T_s1": 0.2, "T_s2": 0.2}.items():
        (tmp_path / f"{t}.json").write_text(json.dumps({
            **aud(v), "particle_mean_disagree": v, "seed_acc": 0.7, "seed_consensus_agree": 0.8,
            "other_consensus_agree": 0.9, "fg_reclassified_diffuse": 0.0,
            "by_class": {c: {"acc": 0.8, "pair_disagree": v} for c in sc.CLASSES[:-1]}}))
    a = type("A", (), {"root": str(tmp_path)})()
    with pytest.raises(SystemExit) as e:
        sc.cmd_audit_report(a)
    assert e.value.code == 3 and "STOP" in (tmp_path / "step0_report.md").read_text()


# ---- pilot readings (pure python) ------------------------------------------------------------- #
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pilot_tree(tmp, gain=0.05, sem_disagree=0.10, sem_semrec=0.80):
    root, step0 = tmp / "pilot", tmp / "step0"
    step0.mkdir(parents=True)
    (step0 / "C_s1.json").write_text(json.dumps({"pair_disagree": 0.20, "acc": 0.82}))
    (step0 / "T_s1.json").write_text(json.dumps({"pair_disagree": 0.19, "acc": 0.82}))
    root.mkdir(parents=True)
    (root / "audit_SEM_s1.json").write_text(json.dumps({"pair_disagree": sem_disagree, "acc": 0.83}))
    hist = lambda rec: [{"epoch": -1, "dev_ng/semantic/recall-val": 0.79, "dev_ng/event/recall-val": 0.70,
                         "dev_ng/instance/adjusted-rand-val": 0.21},
                        {"epoch": 4, "dev_ng/semantic/recall-val": rec, "dev_ng/event/recall-val": 0.70,
                         "dev_ng/instance/adjusted-rand-val": 0.22}]
    write_csv(root / "trunks/SEM_s1/history.csv", hist(sem_semrec))
    write_csv(root / "ref/C_s1_history.csv", hist(0.80))
    rng = np.random.default_rng(0)
    base = rng.random(300) * 0.3
    ev = lambda g: [{"batch": i // 8, "event": i % 8, "ari_soft": b + g + 0.01 * rng.standard_normal(),
                     "ari_hard": b + g / 2} for i, b in enumerate(base)]
    write_csv(root / "ref/C_s1_dev_events_ep4.csv", ev(0.0))
    write_csv(root / "ref/T_s1_dev_events_ep4.csv", ev(0.03))
    write_csv(root / "trunks/SEM_s1/dev_events_ep4.csv", ev(gain))
    dg = lambda g: [{"batch": i // 8, "event": i % 8, "variant": v, "ari_particle": b + g + 0.01 * rng.standard_normal()}
                    for i, b in enumerate(base) for v in ("v2_nonull", "pick_only")]
    write_csv(root / "diag/C_s1_m20/diagnose_events.csv", dg(0.0))
    write_csv(root / "diag/T_s1_m20/diagnose_events.csv", dg(0.03))
    write_csv(root / "diag/SEM_s1_m20/diagnose_events.csv", dg(gain))
    return root, step0


@pytest.mark.parametrize("kw, reading", [
    ({"gain": 0.05}, "PROMISING"),
    ({"gain": 0.02}, "WEAK-POSITIVE"),
    ({"gain": 0.0}, "FUTILE"),
    ({"gain": 0.05, "sem_disagree": 0.18}, "MANIPULATION-FAILED"),     # only a 10 % cut
    ({"gain": 0.05, "sem_semrec": 0.75}, "FALLBACK"),                  # health gate fails
])
def test_pilot_readings(tmp_path, kw, reading):
    res = sc.pilot(*pilot_tree(tmp_path, **kw))
    assert res["reading"] == reading and res["exploratory"] is True
    if reading == "PROMISING":
        assert res["head_M20_vs_C"]["SEM"]["mean"] == pytest.approx(0.05, abs=0.005)
        assert res["manipulation"]["relative_reduction_vs_C"] == pytest.approx(0.5)


def test_pilot_report_renders(tmp_path):
    root, step0 = pilot_tree(tmp_path)
    sc.cmd_pilot_report(type("A", (), {"root": str(root), "step0": str(step0)})())
    md = (root / "report/pilot_report.md").read_text()
    assert "EXPLORATORY" in md and "PROMISING" in md


# ---- loss and monitoring (torch, EAF) ------------------------------------------------------- #
def test_loss_is_zero_for_consistent_particles_and_separates_events():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[2., 0, 0, 0, 0, 0, 0]] * 2 + [[0., 3, 0, 0, 0, 0, 0]] * 2, requires_grad=True)
    y_i = torch.tensor([0, 0, 0, 0])                    # the same particle id in two different events
    y_sem = torch.tensor([0, 0, 1, 1])
    bvec = torch.tensor([0, 0, 1, 1])
    assert float(sc.sem_consistency_loss(x, y_i, y_sem, bvec)) == pytest.approx(0.0, abs=1e-7)
    merged = sc.sem_consistency_loss(x, y_i, y_sem, torch.zeros(4, dtype=torch.long))
    assert float(merged) > 0.1                           # if events were not separated the loss would fire


def test_loss_is_positive_with_a_gradient_and_ignores_unlabelled_and_single_hits():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[2., 0, 0, 0, 0, 0, 0], [0., 2, 0, 0, 0, 0, 0],
                      [9., -9, 0, 0, 0, 0, 0], [0., 0, 5, 0, 0, 0, 0]], requires_grad=True)
    y_i = torch.tensor([0, 0, 1, -1])                   # hit 2: single-hit particle, hit 3: no particle
    y_sem = torch.tensor([0, 0, 1, 2])
    loss = sc.sem_consistency_loss(x, y_i, y_sem, torch.zeros(4, dtype=torch.long))
    loss.backward()
    assert float(loss) > 0
    assert x.grad[:2].abs().sum() > 0 and x.grad[2:].abs().sum() == 0


def test_monitoring_counts_equal_the_numpy_audit():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    n = 400
    y_i = rng.integers(0, 12, n)
    y_sem = rng.integers(-1, 6, n)
    pred = rng.integers(0, 7, n)
    bvec = rng.integers(0, 3, n)
    dis, tot, n_lab, n_ok = sc.pair_counts_torch(torch.tensor(pred), torch.tensor(y_i), torch.tensor(y_sem),
                                                 torch.tensor(bvec))
    lab = (y_sem >= 0) & (y_i >= 0)
    e = sc.audit_event(bvec[lab] * 100 + y_i[lab], pred[lab], y_sem[lab], np.zeros(lab.sum(), bool))
    assert (dis, tot) == (e["pairs_dis"], e["pairs_tot"]) and (n_lab, n_ok) == (e["n"], e["n_ok"])
