#!/usr/bin/env bash
# Stage 1a-R11, Amendment 1: deterministic CPU check of the one F5-failing head (CM1_s2), then dev-B as registered.
# Pre-registration: experiments/particle_hierarchy/comem_confirm.md (Amendment 1). The first run ($MB/comem_confirm)
# is never modified; this run lives in $MB/comem_confirm_v2.
#
#   bash run_comem_confirm_amend1.sh    # copy dev-A GPU evals -> CPU diagnose + CPU eval (NM, DEP) of CM1_s2
#                                       # [+ GPU repeat, descriptive] -> amended F5 -> dev-B evals -> report
#
# Resumable. Needs MB, NUGRAPH_MAT_REPO; env nugraph-materialization-gpu.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R0="$MB/comem_confirm"; R="$MB/comem_confirm_v2"; R10="$MB/comem"
SP="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/soft_particle.py"
CC="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/comem_confirm.py"
DEVA="$MB/cache_val_hx_devA"; DEVB="$MB/cache_val_hx_devB"
HEAD="$R10/heads/CM1_s2/head.pt"
AMEND=(--f5-cpu-root "$R/f5_cpu" --f5-cpu-heads CM1_s2 --gpu-repeat-root "$R/f5_gpu_repeat")
mkdir -p "$R/logs"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$R/logs/pipeline.log"; }

log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
[ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }
# the eval path must be byte-identical to the first run (ff016d3): only these files may have changed
changed=$(git -C "$NUGRAPH_MAT_REPO" diff --name-only ff016d3 HEAD | grep -v -x -e scripts/particle_hierarchy/comem_confirm.py \
  -e scripts/particle_hierarchy/run_comem_confirm_amend1.sh -e tests/test_comem_confirm.py || true)
[ -z "$changed" ] || { log "eval path changed since ff016d3: $changed"; exit 1; }

# 1. copy (never move) the first run's dev-A GPU evals; verify byte-identical
if [ ! -d "$R/devA" ]; then log "copy dev-A GPU evals from the first run"; cp -a "$R0/devA" "$R/devA"; fi
diff -r "$R0/devA" "$R/devA" > /dev/null || { log "dev-A copy differs from the first run"; exit 1; }

# 2. CPU re-execution of CM1_s2 (R10 diagnose + R11 eval NM / DEP) and a descriptive GPU repeat, in parallel
pids=()
if [ ! -f "$R/f5_cpu/diag_last/CM1_s2/diagnose_events.csv" ]; then
  log "CPU diagnose CM1_s2"
  ( CUDA_VISIBLE_DEVICES= python "$SP" diagnose --head "$HEAD" --cache "$DEVA" --batches 0:150 --device cpu \
      --out "$R/f5_cpu/diag_last/CM1_s2" > "$R/logs/cpu_diagnose_CM1_s2.log" 2>&1 ) & pids+=($!)
fi
for dec in NM DEP; do
  o="$R/f5_cpu/devA/$dec/CM1_s2"; flags="--decode v2 --no-null"; [ "$dec" = NM ] && flags="$flags --no-merge"
  if [ ! -f "$o/eval_eval_meta.json" ]; then
    log "CPU eval $dec CM1_s2"
    # shellcheck disable=SC2086
    ( CUDA_VISIBLE_DEVICES= python "$SP" eval --head "$HEAD" --cache "$DEVA" --batches 0:150 --device cpu $flags \
        --tag eval --out "$o" > "$R/logs/cpu_eval_${dec}_CM1_s2.log" 2>&1 ) & pids+=($!)
  fi
done
o="$R/f5_gpu_repeat/NM/CM1_s2"
if [ ! -f "$o/eval_eval_meta.json" ]; then
  log "GPU repeat eval NM CM1_s2 (descriptive)"
  ( python "$SP" eval --head "$HEAD" --cache "$DEVA" --batches 0:150 --device 0 --decode v2 --no-null --no-merge \
      --tag eval --out "$o" > "$R/logs/gpu_repeat_NM_CM1_s2.log" 2>&1 ) & pids+=($!)
fi
fail=0
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { log "a CPU / repeat job failed (see logs); dev-B not read"; exit 1; }

# 3. amended F5 on dev-A; dev-B is read only if it passes
log "amended F5 (CPU exact for CM1_s2) before reading dev-B"
python "$CC" f5 --root "$R" --r10-root "$R10" "${AMEND[@]}" 2>&1 | tee -a "$R/logs/pipeline.log" \
  || { log "amended F5 FAILED: code-path discrepancy; dev-B not read"; exit 1; }

# 4. dev-B GPU evals exactly as registered
declare -A DECFLAGS=([NM]="--decode v2 --no-null --no-merge" [DEP]="--decode v2 --no-null")
for dec in NM DEP; do
  for arm in CTRL CM03 CM1; do
    for t in 1 2 3; do
      n="${arm}_s$t"; o="$R/devB/$dec/$n"
      if [ -f "$o/eval_events.csv" ] && [ -f "$o/eval_eval_meta.json" ]; then log "skip devB $dec $n"; continue; fi
      log "eval devB $dec $n"
      # shellcheck disable=SC2086
      python "$SP" eval --head "$R10/heads/$n/head.pt" --cache "$DEVB" --batches 150:314 --device 0 \
        ${DECFLAGS[$dec]} --tag eval --out "$o" > "$R/logs/eval_devB_${dec}_$n.log" 2>&1
    done
  done
done
log "report"
python "$CC" report --root "$R" --r10-root "$R10" "${AMEND[@]}" --out "$R/report" 2>&1 | tee "$R/logs/report.log"
log "done"
