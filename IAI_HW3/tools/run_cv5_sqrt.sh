#!/usr/bin/env bash
# Train 5 grouped-CV folds (sqrt-inverse class weights, seed 42), then run OOF evaluation.
# Run from the IAI_HW3 directory:  bash tools/run_cv5_sqrt.sh
# A fold whose output already has summary.json is skipped; an incomplete output dir stops the run.
set -euo pipefail

PY=.venv/Scripts/python.exe
ts() { date '+%Y-%m-%d %H:%M:%S'; }

for i in 0 1 2 3 4; do
  out="outputs/cv5_sqrt/fold${i}"
  ckpt="saved_models/cv5_sqrt/fold${i}"
  if [ -f "${out}/summary.json" ]; then
    echo "[$(ts)] fold ${i}: summary.json exists, skipping"
    continue
  fi
  if [ -e "${out}" ] || [ -e "${ckpt}" ]; then
    echo "[$(ts)] fold ${i}: ${out} or ${ckpt} exists but is incomplete (no summary.json); remove or inspect it before rerunning" >&2
    exit 1
  fi
  echo "[$(ts)] fold ${i}: start"
  "${PY}" main.py train --split "outputs/cv5_seed42/fold${i}.json" --class-weighting sqrt-inverse --seed 42 \
    --output-dir "${out}" --checkpoint "${ckpt}"
  echo "[$(ts)] fold ${i}: done"
done

echo "[$(ts)] OOF evaluation: start"
"${PY}" -m tools.cv_oof --fold-dir outputs/cv5_seed42 \
  --checkpoints saved_models/cv5_sqrt/fold0 saved_models/cv5_sqrt/fold1 saved_models/cv5_sqrt/fold2 \
                saved_models/cv5_sqrt/fold3 saved_models/cv5_sqrt/fold4 \
  --output-dir outputs/cv5_sqrt_oof
echo "[$(ts)] OOF evaluation: done"
