#!/usr/bin/env bash
# Stage 1a-R11: confirm co-membership (lambda 0.3) + the no-merge decode on dev-B (a REUSED split, 4th read).
# Nine frozen R10 heads, no training. Pre-registration: experiments/particle_hierarchy/comem_confirm.md
#
#   bash run_comem_confirm.sh    # eval {devA (gate F5), devB} x {NM, DEP} x 9 heads -> report   (~1 h)
#
# Resumable: finished evals are skipped. Needs MB (norm_ep29 dir), NUGRAPH_MAT_REPO; env nugraph-materialization-gpu.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R="$MB/comem_confirm"; R10="$MB/comem"
SP="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/soft_particle.py"
CC="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/comem_confirm.py"
mkdir -p "$R/logs"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$R/logs/pipeline.log"; }
declare -A CACHE=([devA]="$MB/cache_val_hx_devA" [devB]="$MB/cache_val_hx_devB")
declare -A BATCHES=([devA]=0:150 [devB]=150:314)
declare -A DECFLAGS=([NM]="--decode v2 --no-null --no-merge" [DEP]="--decode v2 --no-null")

log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
[ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }

# dev-A first: gate F5 (the eval path reproduces R10) is checked before dev-B is read
for split in devA devB; do
  if [ "$split" = devB ]; then
    log "gate F5 pre-check on dev-A before reading dev-B"
    python "$CC" f5 --root "$R" --r10-root "$R10" 2>&1 | tee -a "$R/logs/pipeline.log" \
      || { log "F5 FAILED: dev-B not read"; exit 1; }
  fi
  for dec in NM DEP; do
    for arm in CTRL CM03 CM1; do
      for t in 1 2 3; do
        n="${arm}_s$t"; o="$R/$split/$dec/$n"
        if [ -f "$o/eval_events.csv" ] && [ -f "$o/eval_eval_meta.json" ]; then log "skip $split $dec $n"; continue; fi
        log "eval $split $dec $n"
        # shellcheck disable=SC2086
        python "$SP" eval --head "$R10/heads/$n/head.pt" --cache "${CACHE[$split]}" --batches "${BATCHES[$split]}" \
          --device 0 ${DECFLAGS[$dec]} --tag eval --out "$o" > "$R/logs/eval_${split}_${dec}_$n.log" 2>&1
      done
    done
  done
done
log "report"
python "$CC" report --root "$R" --r10-root "$R10" --out "$R/report" 2>&1 | tee "$R/logs/report.log"
log "done"
