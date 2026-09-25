#!/usr/bin/env bash
# Stage 1a-R7 formal trunk arm: fine-tune ep29 for 5 epochs with the Stage-1b arm-C recipe,
# plus the within-particle semantic-consistency loss for SEM.  Resumable.
#
#   bash run_sem_trunk.sh SEM 1      # formal SEM seed 1 (tonight)
#   bash run_sem_trunk.sh C 1        # formal matched control seed 1 (must run at the SAME commit)
#
# Output: $MB/sem_formal/trunks/{ARM}_s{SEED}. Needs MB, NUGRAPH_DATA, NUGRAPH_MAT_REPO.
set -euo pipefail
: "${MB:?set MB}" "${NUGRAPH_DATA:?set NUGRAPH_DATA}" "${NUGRAPH_MAT_REPO:?set NUGRAPH_MAT_REPO}"
ARM=${1:?ARM: SEM or C}; SEED=${2:?SEED}
case "$ARM" in SEM) LAMBDA="${SEM_LAMBDA:-1.0}" ;; C) LAMBDA=0.0 ;; *) echo "ARM must be SEM or C"; exit 2 ;; esac
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
FT="$NUGRAPH_MAT_REPO/scripts/particle_hierarchy/finetune.py"
OUT="$MB/sem_formal/trunks/${ARM}_s${SEED}"
EP29="$(python -c "import json;print(json.load(open('$MB/stage1b/C_s1/manifest.json'))['args']['checkpoint'])")"
mkdir -p "$(dirname "$OUT")"
[ -z "$(git -C "$NUGRAPH_MAT_REPO" status --porcelain)" ] || { echo "repo not clean"; exit 1; }
if [ -f "$OUT/manifest.json" ] && grep -q '"finished"' "$OUT/manifest.json"; then
  echo "$ARM s$SEED finished"; exit 0
fi
extra=()
if [ -f "$OUT/resume.pt" ]; then extra=(--resume)
elif [ -d "$OUT" ]; then rm -rf "$OUT"                           # died before its first resume point: start clean
fi
python "$FT" --arm C --seed "$SEED" --sem-cons "$LAMBDA" \
  --checkpoint "$EP29" --checkpoint-sha256 b56d6137aa201558e862d7bbb4566239885a363156c04f213be5238310b2567b \
  --head "$MB/stage1a/run2/head_best.pt" \
  --head-sha256 785ae44443b3afaf862cbc1d9e84b273fcbe477f03a784d7bb5746c032d305e3 \
  --data-path "$NUGRAPH_DATA" --dev-cache "$MB/cache_val_hx_devA" --device 0 \
  --num-workers 4 --prefetch-factor 2 --no-soft-checkpoint --out "$OUT" "${extra[@]}"
