#!/usr/bin/env bash
# Stage 1a-R5: class-information probe — ORACLE UPPER BOUND, not a deployable model.
# Pre-registration: experiments/particle_hierarchy/class_probe.md
#
#   bash run_class_probe.sh      # 4 heads (CTRL/ORACLE x seeds 1,2) -> diagnose -> ranking -> report
#
# Resumable: a finished step (head.pt + train_meta.json, diagnose_events.csv, rank_summary.json) is skipped.
# Needs MB (norm_ep29 dir), NUGRAPH_MAT_REPO; env nugraph-materialization-gpu.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R="$MB/class_probe"
SP="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/soft_particle.py"
RD="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/ranking_diag.py"
CP="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/class_probe.py"
TRAIN="$MB/cache_train_hx"; DEVA="$MB/cache_val_hx_devA"
mkdir -p "$R/logs"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$R/logs/pipeline.log"; }
declare -A MODE=([CTRL]=zeros [ORACLE]=oracle)

log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
[ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }

for t in 1 2; do
  for arm in CTRL ORACLE; do
    n="${arm}_s$t"; h="$R/heads/$n"
    if [ -f "$h/head.pt" ] && [ -f "$h/train_meta.json" ]; then log "skip train $n"; else
      log "train $n (class_channels=${MODE[$arm]})"
      python "$SP" train --train-cache "$TRAIN" --dev-cache "$DEVA" --dev-batches 0:150 --epochs 8 \
        --m 20 --seed "$t" --no-null --device 0 --class-channels "${MODE[$arm]}" --out "$h" \
        > "$R/logs/train_$n.log" 2>&1
      tail -1 "$R/logs/train_$n.log" | tee -a "$R/logs/pipeline.log"
    fi
    for which in last best; do
      ck="$h/head.pt"; [ "$which" = best ] && ck="$h/head_best.pt"
      o="$R/diag_$which/$n"
      if [ -f "$o/diagnose_events.csv" ]; then log "skip diagnose $which $n"; else
        log "diagnose $which $n"
        python "$SP" diagnose --head "$ck" --cache "$DEVA" --batches 0:150 --device 0 --out "$o" \
          > "$R/logs/diag_${which}_$n.log" 2>&1
      fi
    done
    o="$R/rank/$n"
    if [ -f "$o/rank_summary.json" ]; then log "skip rank $n"; else
      log "rank $n (CPU: deterministic, no near-tie flips)"
      CUDA_VISIBLE_DEVICES= python "$RD" run --head "$h/head.pt" --cache "$DEVA" --batches 0:150 \
        --device cpu --out "$o" > "$R/logs/rank_$n.log" 2>&1
    fi
  done
done
log "report"
python "$CP" report --root "$R" --mb-root "$MB" --out "$R/report" 2>&1 | tee "$R/logs/report.log"
log "done"
