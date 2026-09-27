#!/usr/bin/env bash
# Stage 1a-R9: consistency vs accuracy. Six matched 8-channel arms x head seeds {1, 2}; all truth-derived arms
# (incl. CONS) are ORACLE UPPER BOUNDS.
# Pre-registration: experiments/particle_hierarchy/cons_acc.md
#
#   bash run_cons_acc.sh        # read-out checks -> 12 heads -> diagnose -> ranking (CPU, background) -> report
#
# Resumable: finished steps are skipped. Needs MB (norm_ep29 dir), NUGRAPH_MAT_REPO; env nugraph-materialization-gpu.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R="$MB/cons_acc"
SP="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/soft_particle.py"
RD="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/ranking_diag.py"
CA="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/cons_acc.py"
TRAIN="$MB/cache_train_hx"; DEVA="$MB/cache_val_hx_devA"
mkdir -p "$R/logs" "$R/readout"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$R/logs/pipeline.log"; }
ARMS=(CTRL ORACLE TRUECLS CONS FIX50 FIXTRACK)
declare -A MODE=([CTRL]=zeros [ORACLE]=oracle [TRUECLS]=trueclass [CONS]=cons [FIX50]=fix50 [FIXTRACK]=fixtrack)

log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
[ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }

# gate D0: the label arms are built from the trunk read-out, which must reproduce both caches' foreground
for split in train devA; do
  c="$TRAIN"; b="0:"; [ "$split" = devA ] && { c="$DEVA"; b="0:150"; }
  if [ -f "$R/readout/$split.json" ]; then log "skip readout $split"; else
    log "readout-check $split"
    python "$SP" readout-check --cache "$c" --batches "$b" --out "$R/readout/$split.json" \
      > "$R/logs/readout_$split.log" 2>&1 || { log "D0 FAILED on $split"; tail -5 "$R/logs/readout_$split.log"; exit 1; }
  fi
done

RANK_PIDS=()
for t in 1 2; do
  for arm in "${ARMS[@]}"; do
    n="${arm}_s$t"; h="$R/heads/$n"
    if [ -f "$h/head.pt" ] && [ -f "$h/train_meta.json" ]; then log "skip train $n"; else
      rm -rf "$h"                                        # a killed run leaves no resumable state: restart it
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
      log "rank $n (CPU, background, with the trunk's predicted classes)"
      ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
          python "$RD" run --head "$h/head.pt" --cache "$DEVA" --batches 0:150 --device cpu --pred-class \
          --out "$o" > "$R/logs/rank_$n.log" 2>&1 && echo "[$(date +%F' '%T)] rank $n done" >> "$R/logs/pipeline.log" \
        || echo "[$(date +%F' '%T)] rank $n FAILED" >> "$R/logs/pipeline.log" ) &
      RANK_PIDS+=($!)
    fi
  done
done
log "waiting for ${#RANK_PIDS[@]} CPU ranking jobs"
fail=0
for p in "${RANK_PIDS[@]}"; do wait "$p" || fail=1; done
for t in 1 2; do for arm in "${ARMS[@]}"; do
  [ -f "$R/rank/${arm}_s$t/rank_summary.json" ] || { log "missing rank ${arm}_s$t"; fail=1; }
done; done
[ "$fail" = 0 ] || { log "ranking incomplete; re-run this script to resume"; exit 1; }
log "report"
python "$CA" report --root "$R" --r6-root "$MB/pred_class" --out "$R/report" 2>&1 | tee "$R/logs/report.log"
log "done"
