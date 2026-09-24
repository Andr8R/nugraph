#!/usr/bin/env bash
# Stage 1a-R2: T/C latent x M=20 heads (pre-registration: experiments/particle_hierarchy/retrieval_tc.md).
#
#   bash run_retrieval_tc.sh smoke    # sampler-seed determinism + gate, ~5 min
#   bash run_retrieval_tc.sh all      # caches -> identity gates -> heads -> dev-B evals -> report
#
# Resumable: a finished step (cache manifest with "finished", head_best.pt + train_meta.json,
# devB_events.csv) is skipped on rerun. Any gate failure stops the pipeline.
# Needs: MB (norm_ep29 dir), NUGRAPH_DATA, NUGRAPH_MAT_REPO; env nugraph-materialization-gpu.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_DATA:?set NUGRAPH_DATA}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

SEED=20260924                                   # train-split sampler seed (same for all trunks)
RT="$MB/retrieval_tc"; S1B="$MB/stage1b"
MBP="$NUGRAPH_MAT_REPO/scripts/dune_materialization/mat_bench.py"
SP="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/soft_particle.py"
RTC="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/retrieval_tc.py"
mkdir -p "$RT/logs"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$RT/logs/pipeline.log"; }

finished() { [ -f "$1/manifest.json" ] && grep -q '"finished"' "$1/manifest.json"; }

cache() {   # cache <out> <trunk> <split> <batches> [extra...]
  local out=$1 trunk=$2 split=$3 batches=$4; shift 4
  if finished "$out"; then log "skip cache $(basename "$out") (finished)"; return; fi
  rm -rf "$out"
  log "cache $(basename "$out")"
  python "$MBP" cache --checkpoint "$trunk" --data-path "$NUGRAPH_DATA" --split "$split" \
    --batches "$batches" --store-hit-x --batch-size 8 --device 0 --out "$out" "$@" \
    > "$RT/logs/$(basename "$out").log" 2>&1
}

train_head() {    # train_head <arm> <trunk seed> <M>
  local n="$1_s$2_m$3" out="$RT/heads/$1_s$2_m$3"
  if [ -f "$out/head_best.pt" ] && [ -f "$out/train_meta.json" ]; then log "skip head $n"; return; fi
  log "train head $n"
  python "$SP" train --train-cache "$RT/cache_train_$1_s$2" --dev-cache "$RT/cache_devA_$1_s$2" \
    --dev-batches 0:150 --epochs 8 --m "$3" --seed 1 --no-null --device 0 --out "$out" \
    > "$RT/logs/train_$n.log" 2>&1
  tail -1 "$RT/logs/train_$n.log" | tee -a "$RT/logs/pipeline.log"
}

evaldevb() { # evaldevb <arm> <trunk seed> <M>
  local n="$1_s$2_m$3" out="$RT/eval/$1_s$2_m$3"
  if [ -f "$out/devB_events.csv" ]; then log "skip eval $n"; return; fi
  log "eval dev-B $n"
  python "$SP" eval --head "$RT/heads/$n/head_best.pt" --cache "$S1B/eval_devB/$1_s$2/cache" \
    --batches 150:314 --tag devB --decode v2 --no-null --merge-thr 0.5 --device 0 --out "$out" \
    > "$RT/logs/eval_$n.log" 2>&1
}

gate() { log "GATE $*"; python "$RTC" identity --caches "$@" 2>&1 | tee -a "$RT/logs/pipeline.log"; }

trunk_seed() {  # everything for one trunk seed, T and C
  local t=$1
  for A in T C; do
    cache "$RT/cache_train_${A}_s$t" "$S1B/${A}_s$t/trunk_ep4.ckpt" train 0: --limit-events 4000 --sampler-seed $SEED
    cache "$RT/cache_devA_${A}_s$t"  "$S1B/${A}_s$t/trunk_ep4.ckpt" validation 0:150
  done
  gate "$RT/cache_train_T_s1" "$RT"/cache_train_{T,C}_s"$t"                       # A1
  gate "$MB/cache_val_hx_devA" "$RT"/cache_devA_{T,C}_s"$t"                       # A2
  for M in 5 20; do for A in T C; do train_head "$A" "$t" "$M"; done; done
  for M in 5 20; do for A in T C; do evaldevb "$A" "$t" "$M"; done; done
}

case "${1:-}" in
  smoke)
    for tag in a b; do
      cache "$RT/smoke_seed_$tag" "$S1B/T_s1/trunk_ep4.ckpt" train 0: --limit-events 16 --sampler-seed $SEED
    done
    cache "$RT/smoke_seed_c" "$S1B/C_s1/trunk_ep4.ckpt" train 0: --limit-events 16 --sampler-seed $SEED
    cache "$RT/smoke_seed_d" "$S1B/T_s1/trunk_ep4.ckpt" train 0: --limit-events 16 --sampler-seed 1
    gate "$RT/smoke_seed_a" "$RT/smoke_seed_b" "$RT/smoke_seed_c"      # same seed, any trunk: PASS
    neg=$(python "$RTC" identity --caches "$RT/smoke_seed_a" "$RT/smoke_seed_d" 2>&1 || true)
    case "$neg" in
      "FAIL: "*"events differ"*) ;;                 # the only acceptable outcome
      *) log "SMOKE FAIL: a different sampler seed did not give different events: $neg"; exit 1 ;;
    esac
    log "SMOKE PASS: same seed -> identical events across trunks; different seed -> different events"
    ;;
  all)
    log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
    [ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }
    trunk_seed 1
    trunk_seed 2
    log "report"
    python "$RTC" report --root "$RT" --stage1b-devb "$S1B/eval_devB" --out "$RT/report" \
      2>&1 | tee "$RT/logs/report.log"
    log "done"
    ;;
  *) echo "usage: $0 smoke|all"; exit 2 ;;
esac
