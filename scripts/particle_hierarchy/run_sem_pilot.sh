#!/usr/bin/env bash
# Stage 1a-R7 overnight pipeline: Step 0 audit -> GO/STOP -> SEM_s1 trunk (formal arm) -> exploratory pilot readout.
# Pre-registration: experiments/particle_hierarchy/sem_consistency.md
#
#   bash run_sem_pilot.sh        # resumable; every finished step is skipped on rerun
#
# Needs MB (norm_ep29 dir), NUGRAPH_DATA, NUGRAPH_MAT_REPO; env nugraph-materialization-gpu.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_DATA:?set NUGRAPH_DATA}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R="$MB/sem_pilot"; S1B="$MB/stage1b"; RTC="$MB/retrieval_tc"
PH="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy"
SC="$PH/sem_consistency.py"; FT="$PH/finetune.py"; SP="$PH/soft_particle.py"; RD="$PH/ranking_diag.py"
RTCPY="$PH/retrieval_tc.py"; MBP="$NUGRAPH_MAT_REPO/scripts/dune_materialization/mat_bench.py"
EP29="$(python -c "import json;print(json.load(open('$S1B/C_s1/manifest.json'))['args']['checkpoint'])")"
EP29_SHA=b56d6137aa201558e862d7bbb4566239885a363156c04f213be5238310b2567b
HEAD="$MB/stage1a/run2/head_best.pt"; HEAD_SHA=785ae44443b3afaf862cbc1d9e84b273fcbe477f03a784d7bb5746c032d305e3
LAMBDA="${SEM_LAMBDA:-1.0}"                     # pre-registered; fallbacks only as the pre-registration says
mkdir -p "$R/logs" "$R/step0" "$R/ref"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$R/logs/pipeline.log"; }
finished_cache() { [ -f "$1/manifest.json" ] && grep -q '"finished"' "$1/manifest.json"; }

log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)  lambda_cons=$LAMBDA"
[ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }

# ---------------- Step 0: audit of existing trunks, then the pre-registered GO / STOP ----------------
declare -A DEVA=([ep29]="$MB/cache_val_hx_devA" [T_s1]="$RTC/cache_devA_T_s1" [T_s2]="$RTC/cache_devA_T_s2"
                 [C_s1]="$RTC/cache_devA_C_s1" [C_s2]="$RTC/cache_devA_C_s2")
for t in ep29 T_s1 T_s2 C_s1 C_s2; do
  if [ -f "$R/step0/$t.json" ]; then log "skip audit $t"; else
    log "audit $t"
    python "$SC" audit --cache "${DEVA[$t]}" --batches 0:150 --out "$R/step0/$t.json" > "$R/logs/audit_$t.log" 2>&1
  fi
done
set +e
python "$SC" audit-report --root "$R/step0" 2>&1 | tee "$R/logs/step0.log"; rc=${PIPESTATUS[0]}
set -e
if [ "$rc" = 3 ]; then log "Step 0 says STOP: the SEM hypothesis is not justified; nothing trained"; exit 0; fi
[ "$rc" = 0 ] || { log "Step 0 report failed (rc $rc)"; exit 1; }
log "Step 0: GO"

FTARGS=(--arm C --seed 1 --checkpoint "$EP29" --checkpoint-sha256 "$EP29_SHA" --head "$HEAD" --head-sha256 "$HEAD_SHA"
        --data-path "$NUGRAPH_DATA" --dev-cache "$MB/cache_val_hx_devA" --device 0
        --num-workers 4 --prefetch-factor 2 --no-soft-checkpoint --sem-cons "$LAMBDA")

# ---------------- smoke: 20 steps, so a bug costs minutes, not the night ----------------
if [ -f "$R/smoke/steps.csv" ] && grep -q cons "$R/smoke/steps.csv"; then log "skip smoke"; else
  log "smoke (20 steps)"
  rm -rf "$R/smoke"
  python "$FT" "${FTARGS[@]}" --epochs 1 --limit-train-batches 20 --dev-batches 0:2 --out "$R/smoke" \
    > "$R/logs/smoke.log" 2>&1
  python - "$R/smoke/steps.csv" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 20 and all(math.isfinite(float(r["cons"])) and float(r["cons"]) >= 0 for r in rows), "smoke: bad cons"
print(f"smoke OK: 20 steps, cons {float(rows[0]['cons']):.4f} -> {float(rows[-1]['cons']):.4f}")
PY
fi

# ---------------- SEM_s1: the FORMAL SEM seed-1 arm (5 epochs, ~10.5 h), via the formal trunk script ----------------
T="$MB/sem_formal/trunks/SEM_s1"
log "train SEM_s1 (formal arm; resumes if interrupted)"
SEM_LAMBDA="$LAMBDA" bash "$PH/run_sem_trunk.sh" SEM 1 >> "$R/logs/train_SEM_s1.log" 2>&1
grep -h "epoch 4:" "$R/logs/train_SEM_s1.log" | tail -1 | tee -a "$R/logs/pipeline.log" || true
mkdir -p "$R/trunks" && ln -sfn "$T" "$R/trunks/SEM_s1"

# ---------------- exploratory pilot readout (controls = Stage-1b C_s1 / T_s1, older commit) ----------------
cp -n "$S1B/C_s1/history.csv" "$R/ref/C_s1_history.csv" 2>/dev/null || true
cp -n "$S1B/C_s1/dev_events_ep4.csv" "$R/ref/C_s1_dev_events_ep4.csv" 2>/dev/null || true
cp -n "$S1B/T_s1/dev_events_ep4.csv" "$R/ref/T_s1_dev_events_ep4.csv" 2>/dev/null || true

cache() { # cache <out> <split> <batches> [extra]
  local out=$1 split=$2 b=$3; shift 3
  if finished_cache "$out"; then log "skip cache $(basename "$out")"; return; fi
  rm -rf "$out"; log "cache $(basename "$out")"
  python "$MBP" cache --checkpoint "$T/trunk_ep4.ckpt" --data-path "$NUGRAPH_DATA" --split "$split" --batches "$b" \
    --store-hit-x --batch-size 8 --device 0 --out "$out" "$@" > "$R/logs/$(basename "$out").log" 2>&1
}
cache "$R/cache_devA_SEM_s1" validation 0:150
cache "$R/cache_train_SEM_s1" train 0: --limit-events 4000 --sampler-seed 20260924
log "GATE train events == R2 seeded train events";  python "$RTCPY" identity --caches "$RTC/cache_train_C_s1" "$R/cache_train_SEM_s1" | tee -a "$R/logs/pipeline.log"
log "GATE dev-A events == ep29 dev-A events";       python "$RTCPY" identity --caches "$MB/cache_val_hx_devA" "$R/cache_devA_SEM_s1" | tee -a "$R/logs/pipeline.log"

[ -f "$R/audit_SEM_s1.json" ] || { log "audit SEM_s1 (manipulation check)"; \
  python "$SC" audit --cache "$R/cache_devA_SEM_s1" --batches 0:150 --out "$R/audit_SEM_s1.json" > "$R/logs/audit_SEM_s1.log" 2>&1; }

H="$R/heads/SEM_s1_m20"
if [ -f "$H/head.pt" ] && [ -f "$H/train_meta.json" ]; then log "skip head SEM_s1_m20"; else
  log "train head SEM_s1_m20 (M=20, seed 1, same recipe as R2's C_s1_m20 / T_s1_m20)"
  python "$SP" train --train-cache "$R/cache_train_SEM_s1" --dev-cache "$R/cache_devA_SEM_s1" --dev-batches 0:150 \
    --epochs 8 --m 20 --seed 1 --no-null --device 0 --out "$H" > "$R/logs/head_SEM_s1_m20.log" 2>&1
fi
declare -A HD=([SEM_s1_m20]="$H/head.pt" [C_s1_m20]="$RTC/heads/C_s1_m20/head.pt" [T_s1_m20]="$RTC/heads/T_s1_m20/head.pt")
declare -A HC=([SEM_s1_m20]="$R/cache_devA_SEM_s1" [C_s1_m20]="$RTC/cache_devA_C_s1" [T_s1_m20]="$RTC/cache_devA_T_s1")
for n in SEM_s1_m20 C_s1_m20 T_s1_m20; do
  [ -f "$R/diag/$n/diagnose_events.csv" ] || { log "diagnose $n"; python "$SP" diagnose --head "${HD[$n]}" \
    --cache "${HC[$n]}" --batches 0:150 --device 0 --out "$R/diag/$n" > "$R/logs/diag_$n.log" 2>&1; }
  [ -f "$R/rank/$n/rank_summary.json" ] || { log "rank $n (CPU)"; CUDA_VISIBLE_DEVICES= python "$RD" run \
    --head "${HD[$n]}" --cache "${HC[$n]}" --batches 0:150 --device cpu --pred-class --out "$R/rank/$n" \
    > "$R/logs/rank_$n.log" 2>&1; }
done
log "pilot report"
python "$SC" pilot-report --root "$R" --step0 "$R/step0" 2>&1 | tee "$R/logs/pilot_report.log"
log "done"
