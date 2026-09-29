"""Tests for Stage 1a-R11 (dev-B confirmation of co-membership + no-merge decode): report gates and readings
(pure python) and the --no-merge eval path (torch, EAF).

Run:  pytest -q tests/test_comem_confirm.py
"""
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("comem_confirm", ROOT / "scripts/particle_hierarchy/comem_confirm.py")
N_EV = 200
EVAL_F = ["batch", "event", "ari_particle", "ari_particle_hardseed", "n_pred", "n_true_all"]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build(tmp, gains=None, ratio=None, single=None, meta_over=None, r10_shift=0, hard_shift=0.0):
    """gains[(split, dec, arm)] = per-seed gains vs CTRL; ratio / single: CM03 dev-B NM cluster shape per seed."""
    root, r10 = tmp / "root", tmp / "r10"
    g = {"CM03": (0.03, 0.025, 0.04), "CM1": (0.02, 0.03, 0.01)}
    rng = random.Random(3)
    base = [(0.25 + 0.1 * rng.random(), 0.15 + 0.05 * rng.random(), 5 + e % 20) for e in range(N_EV)]
    per_arm_rows = {}
    for split in ("devA", "devB"):
        for dec in cc.DECODES:
            for arm in cc.ARMS:
                for t in cc.SEEDS:
                    gain = 0.0 if arm == "CTRL" else (gains or {}).get((split, dec, arm), g[arm])[t - 1]
                    rows = []
                    for e, (a0, h, nt) in enumerate(base):
                        npred = nt
                        if arm == "CM03" and split == "devB" and dec == "NM":
                            if ratio:
                                npred = max(1, round(nt * ratio[t - 1]))
                            if single and e < int(single[t - 1] * N_EV):
                                npred = 1
                        rows.append({"batch": e // 8, "event": e % 8, "ari_particle": round(a0 + gain + 0.01 * ((e * 7 + t) % 5 - 2) / 2, 9),
                                     "ari_particle_hardseed": h + (hard_shift if arm == "CM1" and e == 2 else 0.0),
                                     "n_pred": npred, "n_true_all": nt})
                    d = root / split / dec / cc.name(arm, t)
                    write_csv(d / "eval_events.csv", EVAL_F, rows)
                    meta = {"head_sha256": cc.HEAD_SHA[arm][t - 1] + "0" * 48, "cache_checkpoint_sha256": "ck",
                            "head_latent_checkpoint_sha256": "ck", "decode_kwargs": cc.DECODE_KW[dec],
                            "git_dirty": False}
                    meta.update((meta_over or {}).get((split, dec, arm, t), {}))
                    (d / "eval_eval_meta.json").write_text(json.dumps(meta))
                    if split == "devA":
                        per_arm_rows[(dec, arm, t)] = rows
    for arm in cc.ARMS:
        for t in cc.SEEDS:
            rows = []
            for dec in cc.DECODES:
                for i, r in enumerate(per_arm_rows[(dec, arm, t)]):
                    ari = r["ari_particle"] + (0.01 if (arm, t, dec) == ("CM1", 2, "NM") and i < r10_shift else 0.0)
                    rows.append({"batch": r["batch"], "event": r["event"], "variant": cc.R10_VARIANT[dec],
                                 "ari_particle": ari})
            write_csv(r10 / "diag_last" / cc.name(arm, t) / "diagnose_events.csv",
                      ["batch", "event", "variant", "ari_particle"], rows)
    return root, r10


def test_confirmed(tmp_path):
    res = cc.evaluate(*build(tmp_path))
    assert res["gates"]["pass"], res["gates"]["failures"]
    v = res["verdict"]
    assert v["reading"] == "CONFIRMED" and all(v["collapse_guard"])
    assert v["delta"] == [pytest.approx(x, abs=0.003) for x in (0.03, 0.025, 0.04)]
    assert "T + co-membership" in v["next"]


@pytest.mark.parametrize("kw, reading", [
    ({"gains": {("devB", "NM", "CM03"): (0.03, 0.015, 0.04)}}, "PARTIAL"),
    ({"gains": {("devB", "NM", "CM03"): (0.0, 0.01, 0.0)}}, "NOT CONFIRMED"),
    ({"ratio": (1.0, 0.6, 1.0)}, "COLLAPSE"),
    ({"single": (0.0, 0.0, 0.2)}, "COLLAPSE"),
])
def test_readings(tmp_path, kw, reading):
    assert cc.evaluate(*build(tmp_path, **kw))["verdict"]["reading"] == reading


def test_the_secondary_arm_and_decode_never_decide(tmp_path):
    kw = {("devB", "NM", "CM1"): (0.0, 0.0, 0.0), ("devB", "DEP", "CM03"): (0.0, 0.0, 0.0)}
    assert cc.evaluate(*build(tmp_path, gains=kw))["verdict"]["reading"] == "CONFIRMED"


def test_f5_tolerates_two_near_tie_flips_but_not_three(tmp_path):
    assert cc.evaluate(*build(tmp_path, r10_shift=2))["gates"]["pass"]
    res = cc.evaluate(*build(tmp_path / "b", r10_shift=3))
    assert any(f.startswith("F5 NM CM1_s2: 3 dev-A events") for f in res["gates"]["failures"])
    assert res["verdict"]["reading"] == "INVALID"


def test_f5_subcommand_only_needs_dev_a(tmp_path, monkeypatch):
    root, r10 = build(tmp_path)
    import shutil
    shutil.rmtree(root / "devB")
    monkeypatch.setattr(sys, "argv", ["comem_confirm.py", "f5", "--root", str(root), "--r10-root", str(r10)])
    cc.main()                                          # raises SystemExit on any F1 / F4 / F5 failure
    root2, r102 = build(tmp_path / "b", r10_shift=3)
    shutil.rmtree(root2 / "devB")
    monkeypatch.setattr(sys, "argv", ["comem_confirm.py", "f5", "--root", str(root2), "--r10-root", str(r102)])
    with pytest.raises(SystemExit):
        cc.main()


@pytest.mark.parametrize("kw, gate", [
    ({"meta_over": {("devB", "NM", "CM03", 2): {"head_sha256": "deadbeef"}}}, "F1 devB/NM/CM03_s2"),
    ({"meta_over": {("devB", "DEP", "CTRL", 1): {"cache_checkpoint_sha256": "other"}}}, "F4 devB/DEP/CTRL_s1"),
    ({"meta_over": {("devB", "NM", "CM1", 3): {"decode_kwargs": cc.DECODE_KW["DEP"]}}}, "F4 devB/NM/CM1_s3"),
    ({"meta_over": {("devA", "NM", "CTRL", 2): {"git_dirty": True}}}, "F4 devA/NM/CTRL_s2"),
    ({"hard_shift": 1e-6}, "F3 devA/NM seed 1: CM1"),
])
def test_each_gate_fails_on_its_planted_defect(tmp_path, kw, gate):
    res = cc.evaluate(*build(tmp_path, **kw))
    assert not res["gates"]["pass"] and res["verdict"]["reading"] == "INVALID"
    assert any(f.startswith(gate) for f in res["gates"]["failures"]), res["gates"]["failures"]


def test_render(tmp_path):
    md = cc.render(cc.evaluate(*build(tmp_path)))
    assert "REUSED split" in md and "**Reading:** CONFIRMED" in md
    assert "| CM03 (primary) |" in md and "Merge cost on dev-B" in md


# ---- the --no-merge eval path (torch, EAF) --------------------------------------------------- #
def test_no_merge_is_the_diagnose_nomerge_variant():
    pytest.importorskip("torch")
    sp = _load("soft_particle_cc", ROOT / "scripts/particle_hierarchy/soft_particle.py")
    a = type("A", (), {"decode": "v2", "no_null": True, "no_merge": True, "merge_thr": None, "exist_thr": None})()
    variants = {n: kw for n, kw, _ in sp.VARIANTS}
    assert sp.decode_kwargs(a) == variants["v2_nonull_nomerge"] == cc.DECODE_KW["NM"]
    a.no_merge = False
    assert sp.decode_kwargs(a) == variants["v2_nonull"] == cc.DECODE_KW["DEP"]


def test_eval_no_merge_reproduces_diagnose_and_records_provenance(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_scatter")
    pytest.importorskip("sklearn")
    pytest.importorskip("nugraph.models.nugraph3.materialize")
    sp = _load("soft_particle_cc2", ROOT / "scripts/particle_hierarchy/soft_particle.py")
    tcp = _load("test_class_probe_cc", ROOT / "tests/test_class_probe.py")
    cache = tcp.synthetic_cache(tmp_path)
    (cache / "manifest.json").write_text(json.dumps({"checkpoint_sha256": "ck"}))
    out, _ = tcp.train(tmp_path, cache, "none", monkeypatch)
    head = out / "head.pt"
    monkeypatch.setattr(sys, "argv", ["soft_particle.py", "diagnose", "--head", str(head), "--cache", str(cache),
                                      "--batches", "0:1", "--device", "cpu", "--out", str(tmp_path / "diag")])
    sp.main()
    diag = {(int(r["batch"]), int(r["event"])): float(r["ari_particle"])
            for r in csv.DictReader(open(tmp_path / "diag" / "diagnose_events.csv"))
            if r["variant"] == "v2_nonull_nomerge"}
    ev = tmp_path / "ev"
    monkeypatch.setattr(sys, "argv", ["soft_particle.py", "eval", "--head", str(head), "--cache", str(cache),
                                      "--batches", "0:1", "--device", "cpu", "--decode", "v2", "--no-null",
                                      "--no-merge", "--tag", "eval", "--out", str(ev)])
    sp.main()
    got = {(int(r["batch"]), int(r["event"])): float(r["ari_particle"])
           for r in csv.DictReader(open(ev / "eval_events.csv"))}
    assert set(got) == set(diag) and all(got[k] == pytest.approx(diag[k], abs=1e-12) for k in got)
    meta = json.loads((ev / "eval_eval_meta.json").read_text())
    assert meta["decode_kwargs"] == cc.DECODE_KW["NM"]
    assert meta["head_sha256"] == sp.mb.sha256(head) and meta["cache_checkpoint_sha256"] == "ck"
    assert meta["head_latent_checkpoint_sha256"] == "ck"
    assert torch.load(head, weights_only=False)["cfg"]["comem"] == 0.0
