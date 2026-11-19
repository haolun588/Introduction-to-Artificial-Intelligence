#!/usr/bin/env bash
# Train 5 grouped-CV folds (seed 42) with the given class weighting, then run OOF evaluation.
# Run from the IAI_HW3 directory:  bash tools/run_cv5.sh <name> <none|sqrt-inverse|quarter-inverse> [epochs] [train args...]
# epochs is optional (positive integer, default 3). Arguments from the 4th on are passed through to "main.py train"
# (e.g. --label-smoothing 0.1); to pass them, epochs must be given explicitly.
# A fold whose output already has summary.json is skipped; an incomplete output dir stops the run.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: bash tools/run_cv5.sh <name> <none|sqrt-inverse|quarter-inverse> [epochs] [train args...]" >&2
  exit 2
fi
NAME="$1"
WEIGHTING="$2"
EPOCHS="${3:-3}"
EXTRA=("${@:4}")
case "${EPOCHS}" in
  ''|*[!0-9]*|0|0[0-9]*) echo "epochs must be a positive integer" >&2; exit 2 ;;
esac
case "${WEIGHTING}" in
  none|sqrt-inverse|quarter-inverse) ;;
  *) echo "class-weighting must be none, sqrt-inverse or quarter-inverse" >&2; exit 2 ;;
esac

PY=.venv/Scripts/python.exe
ts() { date '+%Y-%m-%d %H:%M:%S'; }

for i in 0 1 2 3 4; do
  out="outputs/${NAME}/fold${i}"
  ckpt="saved_models/${NAME}/fold${i}"
  if [ -f "${out}/summary.json" ]; then
    echo "[$(ts)] fold ${i}: summary.json exists, skipping"
    continue
  fi
  if [ -e "${out}" ] || [ -e "${ckpt}" ]; then
    echo "[$(ts)] fold ${i}: ${out} or ${ckpt} exists but is incomplete (no summary.json); remove or inspect it before rerunning" >&2
    exit 1
  fi
  echo "[$(ts)] fold ${i}: start"
  "${PY}" main.py train --split "outputs/cv5_seed42/fold${i}.json" --class-weighting "${WEIGHTING}" --seed 42 \
    --epochs "${EPOCHS}" --output-dir "${out}" --checkpoint "${ckpt}" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "[$(ts)] fold ${i}: done"
done

echo "[$(ts)] OOF evaluation: start"
"${PY}" -m tools.cv_oof --fold-dir outputs/cv5_seed42 \
  --checkpoints saved_models/${NAME}/fold0 saved_models/${NAME}/fold1 saved_models/${NAME}/fold2 \
                saved_models/${NAME}/fold3 saved_models/${NAME}/fold4 \
  --output-dir outputs/${NAME}_oof
echo "[$(ts)] OOF evaluation: done"
