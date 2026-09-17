# NuGraph3 on DUNE: stable training with LayerNorm, and GPU particle materialization

*Technical report, 2026-09-17. Data: DUNE beam simulation (`dune-beam-100k-a29a7b02.gnn.h5`).
Code base: NuGraph `2197a3e` (#170). Reproduction guide:
`scripts/dune_materialization/README.md` in this repository.*

Each section separates **Measured** (numbers from the runs), **Interpretation** and **Caveats**.

---

## 1. Summary

- **Training.** With the post-#170 architecture, DUNE training was unstable: semantic recall
  flipped between ~0.1 and ~0.5, the event head stayed near chance, the object-condensation (OC)
  latent drifted to very large scales, and instance ARI never improved.
  - **What restored it:** a single change, `LayerNorm` on the output of every `NuGraphBlock`,
    was **sufficient** to make the same configuration train stably. This shows sufficiency on
    our setup; it does not prove that missing normalisation is the unique root cause.
  - **Result:** a healthy run. After 30 epochs, **full-validation instance ARI is 0.2118**,
    above the pre-#170 reference of 0.192 under the same metric and protocol, with semantic
    recall 0.786 and event recall 0.707.
- **Materialization.** On that checkpoint, a GPU **seed** decoder (β-seeded condensation points)
  was compared with DBSCAN on a held-out set.
  - **Quality:** particle-level ARI comparable to canonical and tuned DBSCAN within the
    measured uncertainty (paired differences +0.007 ± 0.005 and −0.005 ± 0.005), with
    realistic particle multiplicity.
  - **Cost:** total clustering-only time on the held-out set is **~314× lower** than canonical
    DBSCAN (829 s vs 2.6 s). Median per-event latency improves from 90 ms to 0.92 ms (~98×);
    the total ratio is larger because DBSCAN's cost is concentrated in large events.
  - **Robustness:** zero runtime/materialization errors and ≤ 17 MB GPU memory. However,
    **6.6% of events get no cluster** from the selected seed setting and need a fallback.
  - **Graph:** the radius-graph backend is not viable as implemented.
- **Main limitation.** For seed-style decoding, even perfect seeds (true condensation points
  with nearest-point assignment) reach only ~0.24 particle ARI on this latent, and DBSCAN and
  seed both sit at 0.16–0.17. This is a ceiling for that decoding family, not for every possible
  clustering algorithm, but it points to the embedding as the main limitation.

---

## 2. What was wrong with DUNE training

### Measured
- **Recent runs before the fix** (#170 code, event + semantic + instance heads) behaved badly:
  - Validation instance ARI (NuGraph's fixed `DBSCAN(eps=0.3)`) peaked at epoch 0 (0.05–0.07)
    and then fell.
  - The OC potential loss exploded in the higher-learning-rate runs (614 → 1238 → 12968).
  - **Semantic recall was bistable**, flipping between about 0.09 and 0.51 from epoch to epoch.
  - The event head stayed stuck (event loss ≈ 1.7).
- **The pre-#170 DUNE run** trained stably: semantic recall reached ~0.78 and instance ARI
  climbed to 0.192 on full validation.
- **The eps diagnostic showed part of the ARI "decay" was a metric artefact.** On early vs late
  checkpoints of an unnormalised run:
  - fixed eps=0.3 ARI fell from 0.059 to 0.034;
  - the best ARI over an eps sweep was flat (0.063 → 0.066);
  - the latent RMS grew to 10²–10³.
- **Hypotheses tested and rejected** in controlled runs from the same base:
  - **Detaching the semantic→instance conditioning** (`x_semantic.detach()`): semantic recall
    stayed bistable, loss spikes remained, and the event head stayed stuck.
  - **Gradient clipping at 1.0** (see §3): ARI oscillated, with one degenerate epoch.

### Interpretation
- **Most likely not primarily a failure of the OC loss.** The DUNE "instance instability"
  behaved like a symptom of an unstable **shared trunk**. The instance embedding follows the
  semantic head epoch by epoch; for example, an epoch where every hit is predicted diffuse has no foreground
  to cluster.
- **Two contributing factors on top of that:**
  - the unconstrained, drifting latent scale;
  - a fixed-radius clustering metric that is not scale-invariant.

### Caveats
- **The root cause is not established.** Which #170-era change destabilises the trunk on DUNE
  was not bisected, and the mechanism by which normalisation helps was not isolated.
  Candidates are the encoder/core rewrite and the earlier detach removal (#160).
- **MicroBooNE was not tested.** The file we were given turned out to be DUNE-schema.

---

## 3. The change that restored stable training

`nugraph/nugraph/models/nugraph3/core.py`, class `NuGraphBlock`:

```diff
+        self.norm = nn.LayerNorm(out_features)
 ...
-        return self.net(torch.cat((aggr_out, x), dim=1))
+        return self.norm(self.net(torch.cat((aggr_out, x), dim=1)))
```

This applies to all five core blocks; there is no residual connection.

**Training configuration (identical across the controlled comparison):**
- heads: event + semantic + instance; no filter head, no particle loss, no feature extension;
- `--in-feats 7`, `--coord-feats 8`;
- batch size 32, learning rate 1e-4 **constant** (no OneCycle);
- `--num-workers 0`;
- validation on the first 100 batches each epoch.

### Measured — controlled 8-epoch comparison
All runs start from the same commit and use identical flags; each differs only by the listed
change.

| variant | TB instance ARI, epoch 7 (fixed eps) | adaptive-eps ARI, epoch 0 → 7 | semantic recall, epoch 7 | event recall, epoch 7 | latent RMS range |
|---|---|---|---|---|---|
| base (no change) | 0.018 | 0.033 → 0.023 (flat) | 0.36 (unstable) | 0.23 | 36 – 2663 |
| gradient clip 1.0 | 0.042 (0.013–0.072) | 0.026 → 0.082 (oscillating; 1 degenerate epoch) | 0.17 (bistable 0.09–0.50) | 0.28 | 1425 – 5837 |
| residual + LayerNorm | 0.153 | 0.062 → 0.160 (monotone) | 0.70 | 0.63 | 3.7 – 19.8 |
| **LayerNorm only** | **0.160** | 0.046 → 0.167 (peak 0.185 at epoch 5) | **0.75** | **0.67** | 5.6 – 27 |

Adaptive-eps ARI uses a per-event radius, eps = ρ × median 15th-nearest-neighbour distance, on
40 fixed validation batches.

### Interpretation
- **LayerNorm alone is sufficient** to restore stable training in this configuration; the
  residual adds nothing measurable.
- **This is a sufficiency result, not a root-cause proof.** Other interventions that might also
  work (e.g. reverting individual #170 components, other normalisation placements) were not
  tested.
- **It fixes the whole picture:** trunk stability (semantic and event heads), a small and
  stationary latent scale, and steadily rising instance quality.
- **Fixed-eps and scale-adaptive ARI agree for this run.** The metric artefact disappears once
  the latent is normalised.

### Caveats
- **n = 1:** one training seed per variant.
- **Not tested:** a residual-only variant and a coordinate-normalisation variant were designed
  but not run.
- **Upstream compatibility:** the change is unconditional, so checkpoints trained without it no
  longer load. On current `main`, `NuGraphBlock` is also used 9× by the new optical hierarchy.

---

## 4. Final healthy LayerNorm-only run (30 epochs)

### Measured
- **Schedule:** 8 epochs, then a resume to 30. The learning rate is constant, so the resume
  introduces no schedule discontinuity. Wall time: 7h01m + 20h22m (about 55 min per epoch).
- **TensorBoard validation** (first 100 batches), epochs 0 → 7 → 19 → 29:

| quantity | epoch 0 | epoch 7 | epoch 19 | epoch 29 |
|---|---|---|---|---|
| instance ARI | 0.047 | 0.160 | 0.194 | **0.215** |
| semantic recall | 0.58 | 0.75 | 0.76 | 0.78 |
| event recall | 0.47 | 0.67 | 0.70 | 0.71 |

- The largest drop between epochs is 0.004.
- The best epoch is the last one, and ARI is still rising.
- OC potential loss has been flat at about 0.47 since epoch 4.
- **Checkpoint validation** (`epoch=29-step=84600.ckpt`, NuGraph's own validation loop):

| run | instance ARI | semantic recall | event recall | OC potential |
|---|---|---|---|---|
| control (batch size 32, 100 batches) | 0.2144 (reproduces TensorBoard) | — | — | — |
| **full validation** (batch size 8, 627 batches, 5013 events) | **0.2118** | 0.786 | 0.707 | 0.460 |

### Comparison with previous runs (same legacy metric)

| run | full-validation instance ARI | behaviour |
|---|---|---|
| pre-#170 reference (batch size 8, in11, feature extension zeroed, filter head, particle loss, learning rate 1e-3) | 0.192 (peak 0.200) | stable, but ARI dips to 0.126 / 0.136 in some epochs |
| #170-era runs without the fix | ~0.01–0.07 (100-batch validation) | peak at epoch 0, then decay or oscillation |
| **LayerNorm only, epoch 29** | **0.212** | monotone, no dips |

### Caveats
- **The configurations differ** between the pre-#170 reference and the LayerNorm run, so the
  0.212 vs 0.192 comparison is protocol-matched but not a controlled experiment.
- **The legacy ARI is dominated by unassigned hits.** It is computed over labelled hits with
  "unassigned" (−1) kept as a label on both sides. See §5.

---

## 5. Materialization study on the healthy checkpoint

### Setup
- **Cache.** One forward pass over the full validation set stores, per event:
  - latent coordinates and β;
  - the predicted foreground mask (non-diffuse);
  - truth;
  - NuGraph's own DBSCAN labels.
- **Backends are compared on identical events** (`models/nugraph3/materialize.py`):
  - **DBSCAN** (sklearn, CPU);
  - **seed:** hits with β > `t_beta` become candidates; latent non-max suppression within
    `t_d`; each hit is assigned to the nearest surviving seed within `t_assign`;
  - **graph:** `torch_cluster` radius graph plus connected components.
- **Parity gate.** Benchmark DBSCAN must reproduce NuGraph's own DBSCAN: the per-batch ARI
  difference was ≤ 4×10⁻⁸ and labels were identical on all 627 batches.
- **Primary metric: particle ARI.** This is ARI over predicted-foreground, labelled hits that
  belong to a true particle. The legacy ARI is reported as a secondary number because it
  rewards collapse.
- **Timing.** Clustering-only, CUDA-synchronised, warmed up, one process at a time, 8 CPU
  threads, on an A100 MIG 2g.20gb slice with a 32-CPU pod quota.

### Measured — development stages
- **Stage A (original defaults, full validation):**
  - **DBSCAN:** particle ARI 0.160. It puts **72% of events into one cluster**, with median
    purity 0.044, yet still scores 0.212 legacy ARI.
  - **graph:** particle ARI 0.101. **17.6% of events are invalid** (878 hit the edge cap,
    2 OOM), all of them large events; memory grows as O(n²).
  - **seed:** particle ARI 0.083, leaving **93% of hits unassigned**.
- **Geometry (truth, 1200 tuning events).** The latent does not separate particles:
  - hits sit a median 1.07 from their own condensation point (r90 = 1.27);
  - the nearest *other* particle's condensation point is a median 0.46 away;
  - only 0.4% of particles are separable;
  - **oracle-seed / nearest-assignment ceiling:** particle ARI 0.244 when every hit is assigned
    to the nearest *true* condensation point. This bounds seed-style decoders on this latent; it
    is not a bound on arbitrary clustering algorithms.
- **B1 (single radius) showed a structural problem with the original seed code.** It used one
  radius both to suppress duplicate seeds and to assign hits, and on this latent those needs
  contradict each other.
  - **Fix:** a separate assignment radius `t_assign`.
  - **Performance bug fixed at the same time:** assignment ran one hit per distance call when
    only one seed survived.
  - **Regression check:** 0 label mismatches on 1200 events for two configurations, and a
    median time drop of 9–289 ms → ~1 ms.
- **B2 tuning** (tuning set = batches 0–149, quality only, selection rule fixed in advance):
  - **tuned DBSCAN = eps 0.2**;
  - **seed = `t_beta` 0.5, `t_d` 0.3, `t_assign` ∞.**

### Measured — held-out result (batches 314–626, 2501 events, never used for tuning)

**How to read these tables**
- **Particle ARI (primary quality metric):** ARI over predicted-foreground, labelled hits that
  belong to a true particle. It measures how well real particles are separated. Invalid events
  are excluded; Δ columns are paired differences on the same events.
- **Legacy ARI (secondary):** NuGraph's validation metric at `2197a3e` (labelled hits, with
  "unassigned" kept as a label). It is dominated by the unassigned population and rewards
  merging an event into one cluster, so it is shown only for continuity.
- **Clustering-only timing:** time inside the materialization call, CUDA-synchronised and
  warmed up, one process at a time, 8 CPU threads. It excludes the model forward pass, data
  loading, particle-graph assembly and metric computation.
- **Sensitivity rows:** `t_beta` 0.4 and 0.3 were declared before the held-out run, because the
  selection tie-break was ambiguous. They were not used to re-select.
- **No-cluster events:** events where the method produces no cluster at all. For seed this
  happens when no hit has β above `t_beta`.

**Quality and multiplicity**

| config | valid events | **particle ARI ± SE** | Δ vs canonical | Δ vs tuned | legacy ARI | purity / completeness (median) | predicted particles (true 8) | **no-cluster events** |
|---|---|---|---|---|---|---|---|---|
| DBSCAN eps 0.3 (canonical) | 2501 | 0.160 ± 0.004 | — | −0.012 ± 0.003 | 0.212 | 0.044 / 0.159 | 1 | 73 (2.9%) |
| DBSCAN eps 0.2 (tuned) | 2501 | 0.172 ± 0.004 | +0.012 ± 0.003 | — | 0.196 | 0.088 / 0.188 | 1 | 117 (4.7%) |
| **seed (selected: `t_beta` 0.5, `t_d` 0.3, `t_assign` ∞)** | **2501** | **0.167 ± 0.004** | +0.007 ± 0.005 | −0.005 ± 0.005 | 0.238 | 0.251 / 0.235 | **10** | **165 (6.6%)** |
| seed, `t_beta` 0.4 (sensitivity) | 2501 | 0.167 ± 0.004 | +0.006 ± 0.005 | −0.005 ± 0.005 | 0.239 | 0.330 / 0.224 | 18 | 101 (4.0%) |
| seed, `t_beta` 0.3 (sensitivity) | 2501 | 0.162 ± 0.003 | +0.001 ± 0.005 | −0.011 ± 0.005 | 0.233 | 0.379 / 0.217 | 30 | 57 (2.3%) |
| graph (reference, original defaults) | 2042 (458 edge-cap, 1 OOM) | 0.102 ± 0.004 | −0.058 ± 0.004 | −0.070 ± 0.005 | 0.226 | 0.051 / 0.271 | 16 | 0 |

**Clustering-only runtime and memory**

| config | ms/event (median / p95) | total over 2501 events | speedup vs canonical / tuned | peak GPU memory |
|---|---|---|---|---|
| DBSCAN eps 0.3 (canonical, CPU) | 90 / 1543 | 829 s | 1× / 0.7× | – |
| DBSCAN eps 0.2 (tuned, CPU) | 66 / 1022 | 557 s | 1.5× / 1× | – |
| **seed (selected, GPU)** | **0.92 / 1.9** | **2.6 s** | **314× / 211×** | **17 MB** |
| seed, `t_beta` 0.4 (sensitivity) | 0.98 / 1.5 | 2.5 s | 336× / 226× | 38 MB |
| seed, `t_beta` 0.3 (sensitivity) | 1.07 / 1.6 | 2.7 s | 311× / 209× | 53 MB |
| graph (reference, GPU) | 9.2 / 64 (valid events) | 46 s timed; **630 s with DBSCAN re-running its 459 invalid events** | 18× raw; **1.3× with fallback** | 15.9 GB |

- **Median clustering time by event size** (foreground hits):

| foreground hits | canonical DBSCAN | seed (selected) |
|---|---|---|
| < 500 | 2 ms | 0.8 ms |
| 2k–5k | 129 ms | 0.9 ms |
| 10k–20k | 1.4 s | 1.3 ms |
| > 20k | 3.0 s | 1.2 ms |

- **Repeatability** (repeat runs, per-event time ratio): DBSCAN 1.01, seed 1.07.

### Interpretation
- **Quality.** On the primary metric (particle ARI) there is **no clear difference between the
  selected seed decoder and DBSCAN at the measured uncertainty**: both paired differences are
  within ~1.3 standard errors. No predefined equivalence test was run, so this is not a
  formal equivalence claim. Its higher legacy ARI is consistent with this but is not the basis
  of the claim.
  - It produces **meaningful multiplicity** (10 predicted vs 8 true), where DBSCAN merges almost
    everything into one cluster.
  - It has 3–6× higher purity and higher legacy ARI.
  - The choice is robust: `t_beta` 0.4 gives the same particle ARI.
- **Cost.** Seed's runtime is essentially **independent of event size** and its memory is
  negligible, so it removes CPU DBSCAN as a bottleneck.
  - The two speedup figures measure different things: **~314×** is the ratio of total
    clustering-only time over the held-out set (829 s vs 2.6 s), while the **median per-event**
    latency ratio is **~98×** (90 ms vs 0.92 ms). The total ratio is higher because DBSCAN's time
    is dominated by the large events, where seed stays near 1 ms.
  - On this data canonical DBSCAN costs ~0.33 s per event on average, about 28 min per full
    validation pass. That is an estimate from these numbers, not a measured end-to-end run.
  - Seed is also fast enough to consider materializing *during training*.
- **Graph.** The radius graph fails structurally on this latent:
  - single-linkage chaining produces giant components;
  - the dense `max_num_neighbors=n` buffer makes memory grow as O(n²);
  - its failures are exactly the expensive events, so the DBSCAN fallback erases the speedup.
- **Ceiling.** DBSCAN and seed reach ~65–70% of the oracle-seed / nearest-assignment ceiling
  (0.244). For this decoding family the remaining headroom is small, which suggests the
  **embedding is now the main limitation**. Other clustering algorithms are not bounded by this
  number, but they would still have to cope with the particle overlap measured above.

### Caveats
- **Seed leaves some events without any cluster.** 165 of 2501 held-out events (6.6%) have no
  hit with β > 0.5, versus 2.9% for canonical DBSCAN and 4.7% for tuned DBSCAN. These events
  are included in the particle-ARI mean. A fallback (lower `t_beta` for that event, or DBSCAN)
  should be added before deployment.
- **Selection tie-break.** The rule did not specify median vs mean unassigned fraction. By the
  mean, `t_beta` 0.3 would have been chosen; its particle ARI is comparable within the measured uncertainty, but it over-segments 4×. Both
  alternatives are shown above.
- **Scope of the speedups.** They are clustering-only, GPU vs CPU (8 threads; sklearn DBSCAN's
  neighbour search is effectively single-threaded). The GPU→CPU transfer DBSCAN needs inside
  the model (~0.25 ms) is excluded. Seed is not yet wired into `InstanceDecoder`, so no
  end-to-end validation time has been measured.
- **Oracle ceiling scope.** It was measured on the tuning subset, and only for seed-style
  decoding with nearest-point assignment.
- **Graph was not tuned.** Its failure modes are structural, but a capped-neighbour
  implementation might behave differently.
- **Metric differences with current `main`.** `main` (#174) changed DBSCAN to `min_samples=5`
  and the ARI to unmasked. Numbers computed there are not comparable to these.

---

## 6. Practical conclusions

1. **Carry LayerNorm forward as a proposed, opt-in DUNE change** for the NuGraph3 trunk. It
   was sufficient to restore stable training in our configuration (n = 1), but it should be
   validated on current `main` and on MicroBooNE before being adopted. The underlying root
   cause remains open.
2. **The GPU seed decoder is a credible replacement for CPU DBSCAN** in NuGraph3 instance
   materialization on DUNE: particle ARI comparable to DBSCAN within the measured uncertainty,
   better multiplicity, ~314× less total clustering-only time (~98× at the median event),
   bounded memory, zero runtime/materialization errors. Parameters: `t_beta` 0.5, `t_d` 0.3,
   `t_assign` ∞. Before deployment it needs decoder wiring and a fallback for the 6.6% of
   events where it produces no cluster.
3. **Do not pursue the radius-graph backend** in its current form.
4. **For these decoders, clustering quality is now limited mainly by the learned embedding**
   (particles overlap in latent space), so the next quality gains most likely belong to OC
   training.

## 7. Recommended next steps

1. **Discussion with the NuGraph group (upstream).**
   - Propose LayerNorm as an opt-in `NuGraphBlock` option on current `main`. The commit applies
     cleanly there; decide whether it should also apply to the optical blocks.
   - Re-validate on `main`, and on MicroBooNE once a genuine 5-class file is available.
2. **Wire seed into `InstanceDecoder` on `main`** as a selectable materializer. Add an
   empty-event fallback, then measure end-to-end validation time and train-time
   materialization cost.
3. **Improve particle separation in the embedding**, e.g. OC attractive/repulsive balance,
   particle loss, or β calibration. Use the oracle-seed / nearest-assignment ceiling and the
   geometry diagnostic as target metrics, with this benchmark unchanged.
4. **Replicate the LayerNorm run with a second seed**, to firm up n = 1. Optionally continue
   past epoch 30, since ARI was still rising.
5. **Graph (low priority):** cap `max_num_neighbors` and add a minimum cluster size, only if a
   linkage-based decoder is still wanted.

---

## Appendix A — Provenance

| item | value |
|---|---|
| base commit | `2197a3e99a45cda6df795c2ce26768fd64939f07` (#170) |
| training tree during runs | `2197a3e` + LayerNorm diff (sha256 `1aced571…`); committed unchanged as `d9b0338` |
| training run | `logs/runs/run_20260910_162510_dunefam_bs32_in7_lr1e-4_coord8_nolrsched_norm/` (manifest: command, commit, diff, wall time; epochs 0–7 then `--resume --epochs 30`) |
| checkpoint | `epoch=29-step=84600.ckpt`, sha256 `b56d6137aa201558e862d7bbb4566239885a363156c04f213be5238310b2567b` |
| validation output | `logs/materialization_eval/run_20260910_162510_…_norm__epoch=29-step=84600__val.json` |
| cache | `logs/materialization_eval/norm_ep29/cache/` (manifest: tree = base + LayerNorm, forward peak 1.81 GB, wall 33 min) |
| materializer | `materialize.py` sha256 `3b191928…` (held-out) — committed unchanged as `3482f04`; Stage A used the pre-fix version `d1716311…` |
| outputs | `norm_ep29/{bench,tuneB1,tuneB2,regressB2b,heldout,heldout_repeat}/`, `norm_ep29/final_comparison.md` |
| controlled-family runs | `logs/runs/run_20260905_182657_…_base`, `…_20260906_065027_…_gradclip1`, `…_20260908_214006_…_resnorm` |
| environment | Python 3.14.6, torch 2.12.0 / CUDA 12.9, torch_geometric 2.7.0, torch_scatter 2.1.2, torch_cluster 1.6.3, sklearn 1.9.0, torchmetrics 1.9.0; A100 80GB MIG 2g.20gb; 32-CPU cgroup quota; `OMP_NUM_THREADS=8`; `--num-workers 0` |
| detailed notes | `experiments/training/{dune-family,exp-dune-norm}.md`, `experiments/materialization/stageA_norm_ep29.md` |

## Appendix B — Measurement lessons

- **CPU quota.** The pod's effective CPU quota is 32 cores, not the 127 that `nproc` reports.
  Unpinned torch threads plus concurrent jobs inflated DBSCAN totals 2.3×, with identical labels.
- **OOM on MIG.** A CUDA OOM on this MIG slice can surface as an NVML assert from PyTorch's
  allocator.
- **`torch.cdist` precision.** It switches to a less precise formula when an input has more than
  25 rows. Changing chunk sizes can therefore flip hits that sit exactly at a threshold.
- **Legacy validation ARI.** It rewards merging an event into one cluster, so always report
  particle-level metrics next to it.
