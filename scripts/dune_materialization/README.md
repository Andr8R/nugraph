# DUNE: stable NuGraph3 training (LayerNorm) and GPU instance materialization

This directory reproduces two results obtained on DUNE beam simulation with the #170 NuGraph3
architecture (base commit `2197a3e`):

1. **Training.** Adding a `LayerNorm` on every `NuGraphBlock` output was **sufficient to
   restore stable DUNE training** in this configuration (one run). It is not shown to be the
   unique root cause of the instability. After 30 epochs, full-validation instance ARI is
   **0.2118**, using the legacy NuGraph metric.
2. **Materialization.** On that checkpoint, a GPU **seed** (condensation-point) decoder gives
   **particle-level ARI comparable to DBSCAN within the measured uncertainty** (no predefined
   equivalence test was run). Total clustering-only time on the held-out set is **~314× lower**
   than canonical DBSCAN (829 s vs 2.6 s); median per-event latency improves from 90 ms to
   0.92 ms (~98×). It had **zero runtime/materialization errors** and used ≤ 17 MB GPU memory,
   but **6.6% of selected-seed events produce no cluster** and require a fallback (see
   caveats). The radius-graph backend is not viable as implemented.

The full write-up (measurements, interpretation, caveats) is
`reports/nugraph_dune_layernorm_materialization_report.md` in this repository. This README is the
reproduction guide.

## Branch contents

| commit | what | status |
|---|---|---|
| `nugraph3: LayerNorm on NuGraphBlock outputs` | `models/nugraph3/core.py`: `self.norm = nn.LayerNorm(out_features)`; `update()` returns `self.norm(self.net(...))`. No residual. | **Upstream candidate.** It is unconditional here, exactly as tested; upstream it should be opt-in (it changes the parameter set, so older checkpoints no longer load). |
| `nugraph3: GPU instance-materialization backends + unit tests` | `models/nugraph3/materialize.py` (dbscan / seed / graph / greedy, shared label + status contract, memory-bounded) and `tests/test_materialize.py` (32 CPU tests). **Not wired into `InstanceDecoder`**: validation still runs its own DBSCAN. | **Upstream candidate**, the `seed` backend in particular. It needs decoder wiring on current `main` (#174 refactored `InstanceDecoder.materialize/dbscan`). `graph` needs a `max_num_neighbors` cap before it is usable; `greedy` is reference-only. |
| `scripts/dune_materialization: ... tooling` | `train_with_checkpoints.py`, `validate_checkpoint.py`, `mat_bench.py`, `mat_compare.py`, `diag_eps_scale.py` | **Research tooling**, not for upstream as-is. |
| this README | reproduction guide | research |

## Environment used

| item | value |
|---|---|
| hardware | NVIDIA A100 80GB PCIe, **MIG 2g.20gb slice (~20.9 GB)**; Fermilab EAF pod with a **32-CPU cgroup quota** (`/sys/fs/cgroup/cpu.max = 3200000 100000`), although `nproc` reports 127 |
| software | Python 3.14.6, torch 2.12.0 (CUDA 12.9), torch_geometric 2.7.0, torch_scatter 2.1.2, **torch_cluster 1.6.3** (needed for the fast `graph` path), scikit-learn 1.9.0, torchmetrics 1.9.0 |
| data | `dune-beam-100k-a29a7b02.gnn.h5`; planes u/v/y; 7 semantic classes (pion, muon, kaon, hadron, shower, michel, diffuse); 4 event classes; validation split = 5013 events (627 batches at batch size 8) |
| install | `pip install -e nugraph -e pynuml` from this checkout, so that `import nugraph` resolves here (check with `python -c "import nugraph; print(nugraph.__file__)"`) |

**Rules that matter for trustworthy numbers:**
- **`--num-workers 0`.** On this stack, h5py file handles cannot be pickled into loader workers.
- **Pin CPU threads in every process:**
  `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`. Otherwise torch starts
  127 threads against a 32-CPU quota.
- **Run timing measurements solo.** An earlier run with a CPU benchmark and a GPU benchmark in
  parallel inflated DBSCAN totals by 2.3× while leaving labels identical.
- **Do not run two CUDA processes on the same MIG slice.**
- **OOM on MIG can surface as an NVML assert.** On this MIG slice, a CUDA OOM can appear as
  `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED (CUDACachingAllocator.cpp)`.
  `mat_bench.py` treats that as an OOM event, records it and continues.
- **Set `NUGRAPH_LOG`.** The training launcher writes TensorBoard logs and checkpoints under
  `$NUGRAPH_LOG/<name>/<version>/`.

```bash
export DATA=/path/to/dune-beam-100k-a29a7b02.gnn.h5
export NUGRAPH_LOG=/path/to/logs            # outside the repository
export OUT=/path/to/materialization_eval    # outside the repository
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
S=scripts/dune_materialization
```

## 1. Train (LayerNorm commit checked out, clean tree)

Configuration: event + semantic + instance heads, no filter / particle loss / feature
extension, `--in-feats 7`, `--coord-feats 8`, batch size 32, learning rate 1e-4 **constant**
(`--no-lr-scheduler`), validation on the first 100 batches each epoch.

```bash
ARGS="--device 0 --logger tensorboard --name runs/dune_layernorm --version train \
  --data-path $DATA --event --semantic --instance --coord-feats 8 --in-feats 7 \
  --batch-size 32 --num-workers 0 --learning-rate 0.0001 --no-lr-scheduler --limit_val_batches 100"
python $S/train_with_checkpoints.py $ARGS --epochs 8
python $S/train_with_checkpoints.py $ARGS --epochs 30 \
  --resume $NUGRAPH_LOG/runs/dune_layernorm/train/checkpoints/last.ckpt
```

**How the published run was made.** It ran exactly these two commands: 8 epochs, then a resume
to 30. The learning rate is constant, so a resume introduces no schedule discontinuity; a
single `--epochs 30` run is equivalent up to data order.

**Cost:** about 53–56 min per epoch on the hardware above (~37 min training, 2820 steps at
1.26 it/s; ~18 min validation, dominated by CPU DBSCAN).

**Expected TensorBoard validation values** (first 100 validation batches, fixed eps 0.3):

| epoch | instance ARI | semantic recall |
|---|---|---|
| 0 | 0.047 | — |
| 7 | 0.160 | 0.75 |
| 19 | 0.194 | — |
| 29 | **0.215** | 0.78 |

- ARI climbs monotonically.
- Event recall reaches 0.71.
- OC potential loss stays flat at about 0.47 from epoch 4.

**Published checkpoint:** `epoch=29-step=84600.ckpt`, sha256 `b56d6137aa20…`.

## 2. Validate the checkpoint (NuGraph's own validation loop)

```bash
CKPT=$NUGRAPH_LOG/runs/dune_layernorm/train/checkpoints/epoch=29-step=84600.ckpt
# control: must reproduce the TensorBoard value for epoch 29 (0.2146)
python $S/validate_checkpoint.py --ckpt $CKPT --out ctrl.json --device 0 --data-path $DATA \
  --num-workers 0 --batch-size 32 --limit_val_batches 100
# full validation set, batch size 8 (same protocol as the pre-#170 0.192 reference)
python $S/validate_checkpoint.py --ckpt $CKPT --out full.json --device 0 --data-path $DATA \
  --num-workers 0 --batch-size 8
```

| run | instance ARI | semantic recall | event recall | OC potential | wall time |
|---|---|---|---|---|---|
| control | 0.214447 | — | — | — | 29 min |
| full validation | **0.211783** | 0.786340 | 0.707361 | 0.459741 | 35 min |

## 3. Cache model outputs once

```bash
python $S/mat_bench.py preflight
python $S/mat_bench.py cache --checkpoint $CKPT --data-path $DATA --out $OUT/cache --batch-size 8
```

- **What it runs:** one forward pass per validation batch (~33 min, ≤ 1.8 GB GPU).
- **What it writes:**
  - `shard_*.pt`: per-event `ox`, `beta`, foreground mask, `y_i`, `y_semantic`, and NuGraph's own DBSCAN labels;
  - `batches.csv`: NuGraph's own per-batch ARI;
  - `manifest.json`: checkpoint sha, commit, clean-tree flag, number of LayerNorm blocks.
- **Size:** about 0.8 GB.
- **Clean tree required:** the step refuses to run on a dirty checkout unless you pass `--allow-dirty`.

Checks:
- `wc -l $OUT/cache/batches.csv` should be 628;
- the manifest should show `tree_clean: true` and `layernorm_blocks: 5`.

## 4. Stage A: original defaults, parity gate

```bash
CUDA_VISIBLE_DEVICES= python $S/mat_bench.py bench --cache $OUT/cache --out $OUT/stageA --backends dbscan --device cpu
python $S/mat_bench.py bench --cache $OUT/cache --out $OUT/stageA --backends seed  --device 0
python $S/mat_bench.py bench --cache $OUT/cache --out $OUT/stageA --backends graph --device 0
python $S/mat_bench.py report --bench $OUT/stageA --cache $OUT/cache --ref 0.211783
```

**The parity gate must PASS on all 627 batches** before any backend is compared:
- benchmark DBSCAN's per-batch ARI equals NuGraph's own to about 4e-8;
- labels are identical (ARI = 1);
- the event mean is 0.2119.

`bench` runs are resumable (`--resume`), save every 250 events, and record OOM events instead of
crashing.

Stage A (defaults: seed/graph `t_beta 0.1`, `t_d 0.3`; DBSCAN eps 0.3 / min_samples 15) showed:
- **DBSCAN:** particle ARI 0.160.
- **graph:** particle ARI 0.101, with 17.6% of events invalid (878 hit the edge cap, 2 OOM);
  these are the large events.
- **seed:** particle ARI 0.083, with 93% of foreground hits unassigned.

**Ignore Stage A timings if runs overlapped.**

## 5. Diagnostics and tuning (tuning set = validation batches 0–149, 1200 events)

```bash
python $S/mat_bench.py geometry --cache $OUT/cache --batches 0:150   # truth geometry + oracle-seed ceiling
python $S/mat_bench.py beta     --cache $OUT/cache                   # candidate seeds per beta threshold
python $S/mat_bench.py sizes    --cache $OUT/cache --gb 8            # events at risk for graph memory
```

**Geometry measured on the tuning set:**
- median particle radius around its condensation point: r50 = 1.07, r90 = 1.27;
- nearest *other* condensation point: median 0.46;
- only 0.4% of particles are separable;
- **oracle-seed / nearest-assignment ceiling:** particle ARI 0.244 when every hit is assigned to
  the nearest *true* condensation point (saturates at `t_assign` ≥ 2). This bounds seed-style
  decoding on this latent, not every possible clustering algorithm.

**Tuning grid** (quality only; timings ignored here):

```bash
T=$OUT/tune
for eps in 0.1 0.15 0.2 0.25 0.3; do
  CUDA_VISIBLE_DEVICES= python $S/mat_bench.py bench --cache $OUT/cache --out $T --backends dbscan \
    --device cpu --batches 0:150 --dbscan-eps $eps
done
for tb in 0.3 0.4 0.5; do for td in 0.1 0.2 0.3; do for ta in 1.5 2 3 inf; do
  python $S/mat_bench.py bench --cache $OUT/cache --out $T --backends seed --device 0 \
    --batches 0:150 --t-beta $tb --t-d $td --t-assign $ta
done; done; done
python $S/mat_bench.py report --bench $T --cache $OUT/cache --batches 0:150
```

(The published tuning also ran `t_d` ∈ {0.3, 1, 3, 10} × `t_beta` ∈ {0.3, 0.5, 0.7} with
`t_assign = t_d`. That grid showed that a single radius cannot work on this latent, which
motivated `t_assign`.)

**Selection rule (fixed before looking at the grid):**
1. Keep only configurations with zero invalid events.
2. Take the maximum mean particle ARI.
3. Ties within 0.005 → lower **median** unassigned fraction → predicted/true multiplicity
   closer to 1 → lower mean unassigned fraction.

**Result:**
- **tuned DBSCAN** = eps 0.2 / min_samples 15;
- **seed** = `t_beta 0.5`, NMS `t_d 0.3`, `t_assign inf`.

The rule did not originally specify median vs mean. By *mean* unassigned fraction, `t_beta 0.3`
would win (at 4.2× over-segmentation), so `t_beta 0.4` and `0.3` are carried as held-out
sensitivity rows.

## 6. Held-out evaluation (batches 314–626, 2501 events; solo, sequential, 8 threads)

```bash
H=$OUT/heldout; HR=$OUT/heldout_repeat; HB=314:627
CUDA_VISIBLE_DEVICES= python $S/mat_bench.py bench --cache $OUT/cache --out $H --backends dbscan --device cpu --batches $HB --dbscan-eps 0.3
CUDA_VISIBLE_DEVICES= python $S/mat_bench.py bench --cache $OUT/cache --out $H --backends dbscan --device cpu --batches $HB --dbscan-eps 0.2
for tb in 0.5 0.4 0.3; do
  python $S/mat_bench.py bench --cache $OUT/cache --out $H --backends seed --device 0 --batches $HB --t-beta $tb --t-d 0.3 --t-assign inf
done
python $S/mat_bench.py bench --cache $OUT/cache --out $H --backends graph --device 0 --batches $HB
python $S/mat_bench.py bench --cache $OUT/cache --out $HR --backends seed --device 0 --batches $HB --t-beta 0.5 --t-d 0.3 --t-assign inf
CUDA_VISIBLE_DEVICES= python $S/mat_bench.py bench --cache $OUT/cache --out $HR --backends dbscan --device cpu --batches 314:364 --dbscan-eps 0.3
CUDA_VISIBLE_DEVICES= python $S/mat_bench.py bench --cache $OUT/cache --out $HR --backends dbscan --device cpu --batches 314:364 --dbscan-eps 0.2
python $S/mat_bench.py report --bench $H --cache $OUT/cache --batches $HB        # parity gate must PASS
python $S/mat_compare.py --bench $H --canonical dbscan_eps0.3_min15 --tuned dbscan_eps0.2_min15 \
  --repeat $HR --out $OUT/final_comparison.md
```

**Published held-out result.**

How to read it:
- **Particle ARI (primary):** ARI over predicted-foreground, labelled hits that belong to a true
  particle; invalid events are excluded, and Δ columns are paired differences on the same events.
- **Legacy ARI (secondary):** NuGraph's validation metric at `2197a3e` (labelled hits, with
  "unassigned" kept as a label). It rewards merging an event into one cluster, so it is shown
  only for continuity.
- **Clustering-only timing:** the materialization call alone, CUDA-synchronised, warmed up,
  one process at a time, 8 CPU threads. It excludes the forward pass, data loading, particle-graph
  assembly and metrics.
- **Sensitivity rows** (`t_beta` 0.4 / 0.3) were declared before the held-out run and were not
  used to re-select.
- **No-cluster events:** events where the method returns no cluster at all.

| config | valid | **particle ARI ± SE** | Δ vs canonical | Δ vs tuned | legacy ARI | predicted particles (true 8) | **no-cluster events** |
|---|---|---|---|---|---|---|---|
| DBSCAN eps 0.3 (canonical) | 2501 | 0.160 ± 0.004 | — | −0.012 ± 0.003 | 0.212 | 1 | 73 (2.9%) |
| DBSCAN eps 0.2 (tuned) | 2501 | 0.172 ± 0.004 | +0.012 ± 0.003 | — | 0.196 | 1 | 117 (4.7%) |
| **seed tb0.5 / td0.3 / ta inf (selected)** | 2501 | **0.167 ± 0.004** | +0.007 ± 0.005 | −0.005 ± 0.005 | 0.238 | 10 | **165 (6.6%)** |
| seed tb0.4 (sensitivity) | 2501 | 0.167 ± 0.004 | +0.006 ± 0.005 | −0.005 ± 0.005 | 0.239 | 18 | 101 (4.0%) |
| seed tb0.3 (sensitivity) | 2501 | 0.162 ± 0.003 | +0.001 ± 0.005 | −0.011 ± 0.005 | 0.233 | 30 | 57 (2.3%) |
| graph (defaults) | 2042 (458 edge-cap, 1 OOM) | 0.102 ± 0.004 | −0.058 ± 0.004 | −0.070 ± 0.005 | 0.226 | 16 | 0 |

| config | ms/event (median / p95), clustering only | total over 2501 events | speedup vs canonical / tuned | peak GPU memory |
|---|---|---|---|---|
| DBSCAN eps 0.3 (canonical, CPU) | 90 / 1543 | 829 s | 1× / 0.7× | – |
| DBSCAN eps 0.2 (tuned, CPU) | 66 / 1022 | 557 s | 1.5× / 1× | – |
| **seed (selected, GPU)** | **0.92 / 1.9** | **2.6 s** | **314× / 211×** | 17 MB |
| seed tb0.4 / tb0.3 (sensitivity) | 0.98 / 1.5 · 1.07 / 1.6 | 2.5 s · 2.7 s | 336× · 311× (vs canonical) | 38 MB · 53 MB |
| graph (defaults, GPU) | 9.2 / 64 (valid events) | 46 s; 630 s with DBSCAN re-running its 459 invalid events | 18× raw; 1.3× with fallback | 15.9 GB |

- **Timing repeatability:** DBSCAN within 1%, seed within 7%.
- **Seed time is flat in event size:** 0.8–1.3 ms per event, while canonical DBSCAN grows
  from 2 ms to 3.0 s.

## Caveats

- **One checkpoint, one dataset, one training run** (n = 1 for the LayerNorm result).
- **For these decoders, the latent is the main limit on quality.** The oracle-seed /
  nearest-assignment ceiling is ~0.244 particle ARI, and DBSCAN and seed land at 0.16–0.17.
  The ceiling applies to seed-style decoding, not to every clustering algorithm.
- **The legacy validation ARI rewards collapse.** DBSCAN puts ~70% of events into a single
  cluster yet scores 0.21. Report particle-level metrics alongside it.
- **Seed leaves some events without any cluster.** 165 of 2501 held-out events (6.6%) have no
  hit with beta > 0.5, versus 2.9% (canonical) and 4.7% (tuned) for DBSCAN. They are included
  in the particle-ARI mean. Add a per-event fallback (lower `t_beta`, or DBSCAN) before deployment.
- **Scope of the speedups:** clustering only, GPU vs CPU (8 threads; sklearn's DBSCAN
  neighbour search is effectively single-threaded). The in-model GPU→CPU hop DBSCAN needs
  (~0.25 ms per event) is not included. No end-to-end validation time was measured with seed
  wired in.
- **Metric mismatch with current `main`.** Current `main` (#174) uses DBSCAN
  `min_samples=5` and an unmasked ARI; its numbers are not comparable to these.
