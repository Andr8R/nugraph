#!/usr/bin/env bash
# Stage 1c campaign: Stage-1b T vs T + co-membership (lambda 0.3) in the trunk. Decision on `test` (2nd read).
# Pre-registration: experiments/particle_hierarchy/stage1c.md
#
#   bash run_stage1c.sh smoke   # ~40 min, dev-A only: resume identity, matched order, logged vs trained term
#   bash run_stage1c.sh all     # ~58 h: 4 trunks -> K1/K2 -> train/dev-A caches (K3) -> 12 heads (K4)
#                               #        -> TEST LOCK opens -> test caches (K3) -> evals -> report
#
# Resumable: re-issue the same command after any interruption (trunks resume from resume.pt; finished steps are
# skipped). Needs MB, NUGRAPH_DATA, NUGRAPH_MAT_REPO; env nugraph-materialization-gpu.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_DATA:?set NUGRAPH_DATA}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
R="$MB/stage1c"
PH="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy"
FT="$PH/finetune.py"; SP="$PH/soft_particle.py"; RD="$PH/ranking_diag.py"; RTC="$PH/retrieval_tc.py"
S1C="$PH/stage1c.py"; MBP="$NUGRAPH_MAT_REPO/scripts/dune_materialization/mat_bench.py"
EP29="$(python -c "import json;print(json.load(open('$MB/stage1b/C_s1/manifest.json'))['args']['checkpoint'])")"
EP29_SHA=b56d6137aa201558e862d7bbb4566239885a363156c04f213be5238310b2567b
HEAD0="$MB/stage1a/run2/head_best.pt"
HEAD0_SHA=785ae44443b3afaf862cbc1d9e84b273fcbe477f03a784d7bb5746c032d305e3
TRAIN_SEED=20260924
declare -A LAM=([T]=0.0 [TCM]=0.3)
declare -A HLAM=([CMH_h1]=0.3 [CMH_h2]=0.3 [CTH_h1]=0.0)
mkdir -p "$R/logs" "$R/gates"
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$R/logs/pipeline.log"; }
finished() { [ -f "$1/manifest.json" ] && grep -q '"finished"' "$1/manifest.json"; }

trunk_run() {   # trunk_run <out> <comem> <seed> [extra finetune args]  (resumes automatically)
  local out=$1 lam=$2 seed=$3; shift 3
  if finished "$out"; then log "skip trunk ${out#$R/} (finished)"; return; fi
  local extra=()
  if [ -f "$out/resume.pt" ]; then extra=(--resume)
  elif [ -d "$out" ]; then rm -rf "$out"              # died before its first resume point: start clean
  fi
  log "trunk ${out#$R/} (comem=$lam, seed=$seed) ${extra[*]:-fresh}"
  mkdir -p "$(dirname "$out")"
  python "$FT" --arm T --seed "$seed" --comem "$lam" --comem-log \
    --checkpoint "$EP29" --checkpoint-sha256 "$EP29_SHA" --head "$HEAD0" --head-sha256 "$HEAD0_SHA" \
    --data-path "$NUGRAPH_DATA" --dev-cache "$MB/cache_val_hx_devA" --device 0 \
    --num-workers 4 --prefetch-factor 2 --no-soft-checkpoint --out "$out" "$@" ${extra[@]+"${extra[@]}"} \
    >> "$R/logs/$(echo "${out#$R/}" | tr / _).log" 2>&1
}

cache() {       # cache <out> <trunk ckpt> <split> <batches> [extra]
  local out=$1 ck=$2 split=$3 b=$4; shift 4
  if finished "$out"; then log "skip cache $(basename "$out")"; return; fi
  rm -rf "$out"
  log "cache $(basename "$out")"
  python "$MBP" cache --checkpoint "$ck" --data-path "$NUGRAPH_DATA" --split "$split" --batches "$b" \
    --store-hit-x --batch-size 8 --device 0 --out "$out" "$@" > "$R/logs/$(basename "$out").log" 2>&1
}

identity() {    # identity <gate name> <caches...>: writes gates/<name>.txt, stops on failure
  local g=$1; shift
  if grep -q '^PASS' "$R/gates/$g.txt" 2>/dev/null; then log "skip gate $g (PASS)"; return; fi
  log "gate $g"
  python "$RTC" identity --caches "$@" > "$R/gates/$g.txt" 2>&1 || true
  tee -a "$R/logs/pipeline.log" < "$R/gates/$g.txt"
  grep -q '^PASS' "$R/gates/$g.txt" || { log "$g FAILED: stopping"; exit 1; }
}

TRUNKS=(T_s1 TCM_s1 T_s2 TCM_s2)
arm_of() { echo "${1%_s*}"; }
seed_of() { echo "${1##*_s}"; }

case "${1:-}" in
smoke)
  S="$R/smoke"; SM=(--limit-train-batches 30 --epochs 2 --resume-every 10 --dev-batches 0:10)
  log "smoke start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
  [ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }
  trunk_run "$S/R" 0.3 1 "${SM[@]}"                                   # TCM, uninterrupted reference
  if ! finished "$S/I" && [ ! -f "$S/I/resume.pt" ]; then             # TCM, crashed after step 45 ...
    rm -rf "$S/I"
    trunk_run "$S/I" 0.3 1 "${SM[@]}" --crash-at-step 45 || true
    grep -q "SimulatedCrash" "$R/logs/smoke_I.log" || { log "smoke: the planned crash did not happen"; exit 1; }
  fi
  trunk_run "$S/I" 0.3 1 "${SM[@]}"                                   # ... and resumed
  trunk_run "$S/C" 0.0 1 --limit-train-batches 20 --epochs 1 --resume-every 10 --dev-batches 0:10   # T control
  python "$S1C" smoke-check --root "$S" 2>&1 | tee -a "$R/logs/pipeline.log"
  ;;
all)
  log "start $(git -C "$NUGRAPH_MAT_REPO" log --oneline -1)"
  [ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { log "repo not clean"; exit 1; }
  grep -q "SMOKE PASS" "$R/logs/pipeline.log" 2>/dev/null || { log "run the smoke first (it must PASS)"; exit 1; }
  # 1. trunks, seed pair 1 then seed pair 2
  for n in "${TRUNKS[@]}"; do trunk_run "$R/trunks/$n" "${LAM[$(arm_of "$n")]}" "$(seed_of "$n")"; done
  python "$S1C" train-gates --root "$R" 2>&1 | tee -a "$R/logs/pipeline.log"
  # 2. train and dev-A caches (never test), identity gates
  for n in "${TRUNKS[@]}"; do
    cache "$R/cache_train_$n" "$R/trunks/$n/trunk_ep4.ckpt" train 0: --limit-events 4000 --sampler-seed "$TRAIN_SEED"
    cache "$R/cache_devA_$n" "$R/trunks/$n/trunk_ep4.ckpt" validation 0:150
  done
  identity K3_train "$R"/cache_train_{T_s1,TCM_s1,T_s2,TCM_s2}
  identity K3_devA "$MB/cache_val_hx_devA" "$R"/cache_devA_{T_s1,TCM_s1,T_s2,TCM_s2}
  # 3. heads (M=20, 8 epochs, last epoch) + CPU rankings on dev-A in the background
  pids=()
  for n in "${TRUNKS[@]}"; do
    for h in CMH_h1 CMH_h2 CTH_h1; do
      o="$R/heads/${n}_$h"
      if [ -f "$o/head.pt" ] && [ -f "$o/train_meta.json" ]; then log "skip head ${n}_$h"; else
        rm -rf "$o"; log "head ${n}_$h (comem=${HLAM[$h]})"
        python "$SP" train --train-cache "$R/cache_train_$n" --dev-cache "$R/cache_devA_$n" --dev-batches 0:150 \
          --epochs 8 --m 20 --seed "${h##*_h}" --no-null --device 0 --comem "${HLAM[$h]}" --out "$o" \
          > "$R/logs/head_${n}_$h.log" 2>&1
      fi
    done
    o="$R/rank/${n}_CMH_h1"
    if [ ! -f "$o/rank_summary.json" ]; then
      log "rank ${n}_CMH_h1 (CPU, background, dev-A)"
      ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
          python "$RD" run --head "$R/heads/${n}_CMH_h1/head.pt" --cache "$R/cache_devA_$n" --batches 0:150 \
          --device cpu --out "$o" > "$R/logs/rank_${n}.log" 2>&1 ) & pids+=($!)
    fi
  done
  python "$S1C" head-gates --root "$R" 2>&1 | tee -a "$R/logs/pipeline.log"
  # 4. TEST LOCK: opens only when all trunks and heads are frozen and K1 / K2 / K3(train, dev-A) / K4 have passed
  python "$S1C" train-gates --root "$R" > /dev/null && python "$S1C" head-gates --root "$R" > /dev/null \
    && grep -q '^PASS' "$R/gates/K3_train.txt" && grep -q '^PASS' "$R/gates/K3_devA.txt" \
    || { log "test lock stays closed: a pre-test gate is not PASS"; exit 1; }
  log "TEST LOCK OPEN: test split, 2nd read (ledger)"
  for n in "${TRUNKS[@]}"; do cache "$R/cache_test_$n" "$R/trunks/$n/trunk_ep4.ckpt" test 0:; done
  identity K3_test "$R"/cache_test_{T_s1,TCM_s1,T_s2,TCM_s2}
  for n in "${TRUNKS[@]}"; do
    for h in CMH_h1 CMH_h2 CTH_h1 soft; do
      hp="$R/heads/${n}_$h/head.pt"; [ "$h" = soft ] && hp="$R/trunks/$n/soft_ep4.pt"
      for dec in NM DEP; do
        o="$R/eval/$n/$h/$dec"; flags=(--decode v2 --no-null); [ "$dec" = NM ] && flags+=(--no-merge)
        if [ -f "$o/test_eval_meta.json" ]; then continue; fi
        log "eval test $n $h $dec"
        python "$SP" eval --head "$hp" --cache "$R/cache_test_$n" --batches 0: --allow-heldout --device 0 \
          "${flags[@]}" --tag test --out "$o" > "$R/logs/eval_${n}_${h}_$dec.log" 2>&1
      done
    done
  done
  fail=0; for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] || log "a ranking job failed (mechanism table will be incomplete; re-run to retry)"
  log "report"
  python "$S1C" report --root "$R" --out "$R/report" 2>&1 | tee "$R/logs/report.log"
  log "done"
  ;;
*) echo "usage: $0 smoke|all"; exit 2 ;;
esac
