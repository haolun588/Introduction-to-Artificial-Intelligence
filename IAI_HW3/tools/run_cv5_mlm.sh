#!/usr/bin/env bash
# Per fold: MLM continued pre-training on the fold's training texts only, then fine-tune from that MLM checkpoint;
# after all 5 folds are complete, run OOF evaluation.
# Run from the IAI_HW3 directory:
#   bash tools/run_cv5_mlm.sh <name> <none|sqrt-inverse|quarter-inverse> <finetune_epochs> <mlm_epochs> [folds...]
# folds default to 0 1 2 3 4. A stage whose output already has summary.json is skipped; an incomplete output/checkpoint
# directory stops the run. OOF evaluation only runs when all 5 folds have summary.json (otherwise it is skipped).
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: bash tools/run_cv5_mlm.sh <name> <none|sqrt-inverse|quarter-inverse> <finetune_epochs> <mlm_epochs> [folds...]" >&2
  exit 2
fi
NAME="$1"
WEIGHTING="$2"
EPOCHS="$3"
MLM_EPOCHS="$4"
if [ "$#" -gt 4 ]; then FOLDS=("${@:5}"); else FOLDS=(0 1 2 3 4); fi
for value in "${EPOCHS}" "${MLM_EPOCHS}"; do
  case "${value}" in
    ''|*[!0-9]*|0|0[0-9]*) echo "epochs must be positive integers" >&2; exit 2 ;;
  esac
done
case "${WEIGHTING}" in
  none|sqrt-inverse|quarter-inverse) ;;
  *) echo "class-weighting must be none, sqrt-inverse or quarter-inverse" >&2; exit 2 ;;
esac
for i in "${FOLDS[@]}"; do
  case "${i}" in
    0|1|2|3|4) ;;
    *) echo "folds must be among 0 1 2 3 4" >&2; exit 2 ;;
  esac
done

PY=.venv/Scripts/python.exe
ts() { date '+%Y-%m-%d %H:%M:%S'; }

for i in "${FOLDS[@]}"; do
  mlm_out="outputs/${NAME}_mlm/fold${i}"
  mlm_ckpt="saved_models/${NAME}_mlm/fold${i}"
  out="outputs/${NAME}/fold${i}"
  ckpt="saved_models/${NAME}/fold${i}"
  if [ -f "${mlm_out}/summary.json" ]; then
    echo "[$(ts)] fold ${i}: MLM summary.json exists, skipping pretrain-mlm"
  else
    if [ -e "${mlm_out}" ] || [ -e "${mlm_ckpt}" ]; then
      echo "[$(ts)] fold ${i}: ${mlm_out} or ${mlm_ckpt} exists but is incomplete (no summary.json); remove or inspect it before rerunning" >&2
      exit 1
    fi
    echo "[$(ts)] fold ${i}: pretrain-mlm start"
    "${PY}" main.py pretrain-mlm --split "outputs/cv5_seed42/fold${i}.json" --seed 42 \
      --epochs "${MLM_EPOCHS}" --output-dir "${mlm_out}" --checkpoint "${mlm_ckpt}"
    echo "[$(ts)] fold ${i}: pretrain-mlm done"
  fi
  if [ -f "${out}/summary.json" ]; then
    echo "[$(ts)] fold ${i}: summary.json exists, skipping train"
    continue
  fi
  if [ -e "${out}" ] || [ -e "${ckpt}" ]; then
    echo "[$(ts)] fold ${i}: ${out} or ${ckpt} exists but is incomplete (no summary.json); remove or inspect it before rerunning" >&2
    exit 1
  fi
  echo "[$(ts)] fold ${i}: train start"
  "${PY}" main.py train --split "outputs/cv5_seed42/fold${i}.json" --class-weighting "${WEIGHTING}" --seed 42 \
    --epochs "${EPOCHS}" --output-dir "${out}" --checkpoint "${ckpt}" --init-from "${mlm_ckpt}"
  echo "[$(ts)] fold ${i}: train done"
done

if [ ! -f "outputs/${NAME}/fold0/summary.json" ] || [ ! -f "outputs/${NAME}/fold1/summary.json" ] \
   || [ ! -f "outputs/${NAME}/fold2/summary.json" ] || [ ! -f "outputs/${NAME}/fold3/summary.json" ] \
   || [ ! -f "outputs/${NAME}/fold4/summary.json" ]; then
  echo "[$(ts)] not all 5 folds are complete; skipping OOF evaluation (rerun with all folds, or run tools.cv_oof manually)"
  exit 0
fi

echo "[$(ts)] OOF evaluation: start"
"${PY}" -m tools.cv_oof --fold-dir outputs/cv5_seed42 \
  --checkpoints saved_models/${NAME}/fold0 saved_models/${NAME}/fold1 saved_models/${NAME}/fold2 \
                saved_models/${NAME}/fold3 saved_models/${NAME}/fold4 \
  --output-dir outputs/${NAME}_oof
echo "[$(ts)] OOF evaluation: done"
