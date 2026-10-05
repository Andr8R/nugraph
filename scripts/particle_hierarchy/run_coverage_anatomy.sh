#!/usr/bin/env bash
# Coverage anatomy (T1 diagnostic, dev-A only, no training). Pre-registration:
# experiments/particle_hierarchy/coverage_anatomy.md
#
#   bash run_coverage_anatomy.sh      # tests -> 5 dev-A caches (resumable) -> report
#
# Needs MB and NUGRAPH_MAT_REPO; env nugraph-materialization-gpu (the nupart session).
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R="$MB/coverage_anatomy"
PH="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy"
mkdir -p "$R/logs"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$R/logs/pipeline.log"; }

log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
[ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }
T="$( cd "$NUGRAPH_MAT_REPO" && CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_coverage_anatomy.py \
    tests/test_coverage_anatomy_torch.py 2>&1 | tail -3 || true )"
echo "$T" | tee -a "$R/logs/pipeline.log"
echo "$T" | grep -q "22 passed" && ! echo "$T" | grep -q "failed" \
  || { log "tests did not report 22 passed: stopping"; exit 1; }

declare -A CACHE=(
  [ep29]="$MB/cache_val_hx_devA"
  [T_s1]="$MB/stage1c/cache_devA_T_s1"  [T_s2]="$MB/stage1c/cache_devA_T_s2"
  [TCM_s1]="$MB/stage1c/cache_devA_TCM_s1"  [TCM_s2]="$MB/stage1c/cache_devA_TCM_s2")
for n in ep29 T_s1 T_s2 TCM_s1 TCM_s2; do
  o="$R/runs/$n"
  if [ -f "$o/meta.json" ] && grep -q '"finished"' "$o/meta.json"; then log "skip $n (finished)"; continue; fi
  rm -rf "$o"; log "run $n <- ${CACHE[$n]}"
  python "$PH/coverage_anatomy.py" run --cache "${CACHE[$n]}" --out "$o" --batches 0:150 --device 0 \
    > "$R/logs/run_$n.log" 2>&1
done
log "report"
python "$PH/coverage_anatomy.py" report --primary "$R/runs/T_s1" "$R/runs/T_s2" \
  --descriptive "$R/runs/TCM_s1" "$R/runs/TCM_s2" --anchor "$R/runs/ep29" --out "$R/report" \
  > "$R/logs/report.log" 2>&1
head -40 "$R/report/report.md" | tee -a "$R/logs/pipeline.log"
log "done"
