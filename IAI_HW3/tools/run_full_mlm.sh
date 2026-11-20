#!/usr/bin/env bash
# Per seed: MLM continued pre-training on the default split's 8,100 training texts, then sqrt-inverse fine-tuning from
# that MLM checkpoint; after all seeds are complete, build the equal-weight ensemble (at most 3 seeds).
# Run from the IAI_HW3 directory:
#   bash tools/run_full_mlm.sh <name> <mlm_epochs> <finetune_epochs> <seeds...>
# A stage whose output already has summary.json is skipped; an incomplete output/checkpoint directory stops the run.
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: bash tools/run_full_mlm.sh <name> <mlm_epochs> <finetune_epochs> <seeds...>" >&2
  exit 2
fi
NAME="$1"
MLM_EPOCHS="$2"
EPOCHS="$3"
SEEDS=("${@:4}")
case "${NAME}" in
  ''|*[!A-Za-z0-9_-]*) echo "name must contain only letters, digits, _ and -" >&2; exit 2 ;;
esac
for value in "${MLM_EPOCHS}" "${EPOCHS}"; do
  case "${value}" in
    ''|*[!0-9]*|0|0[0-9]*) echo "epochs must be positive integers" >&2; exit 2 ;;
  esac
done
for s in "${SEEDS[@]}"; do
  case "${s}" in
    ''|*[!0-9]*) echo "seeds must be integers" >&2; exit 2 ;;
  esac
done
if [ "${#SEEDS[@]}" -gt 3 ]; then
  echo "at most 3 seeds (ensemble limit is 3 models)" >&2
  exit 2
fi
if [ "$(printf '%s\n' "${SEEDS[@]}" | sort -u | wc -l)" -ne "${#SEEDS[@]}" ]; then
  echo "seeds must be distinct" >&2
  exit 2
fi

PY=.venv/Scripts/python.exe
SPLIT=outputs/eda/split.json
ts() { date '+%Y-%m-%d %H:%M:%S'; }

for s in "${SEEDS[@]}"; do
  mlm_out="outputs/${NAME}_mlm/seed${s}"
  mlm_ckpt="saved_models/${NAME}_mlm/seed${s}"
  out="outputs/${NAME}/seed${s}"
  ckpt="saved_models/${NAME}/seed${s}"
  if [ -f "${mlm_out}/summary.json" ]; then
    echo "[$(ts)] seed ${s}: MLM summary.json exists, skipping pretrain-mlm"
  else
    if [ -e "${mlm_out}" ] || [ -e "${mlm_ckpt}" ]; then
      echo "[$(ts)] seed ${s}: ${mlm_out} or ${mlm_ckpt} exists but is incomplete (no summary.json); remove or inspect it before rerunning" >&2
      exit 1
    fi
    echo "[$(ts)] seed ${s}: pretrain-mlm start"
    "${PY}" main.py pretrain-mlm --split "${SPLIT}" --seed "${s}" \
      --epochs "${MLM_EPOCHS}" --output-dir "${mlm_out}" --checkpoint "${mlm_ckpt}"
    echo "[$(ts)] seed ${s}: pretrain-mlm done"
  fi
  if [ -f "${out}/summary.json" ]; then
    echo "[$(ts)] seed ${s}: summary.json exists, skipping train"
    continue
  fi
  if [ -e "${out}" ] || [ -e "${ckpt}" ]; then
    echo "[$(ts)] seed ${s}: ${out} or ${ckpt} exists but is incomplete (no summary.json); remove or inspect it before rerunning" >&2
    exit 1
  fi
  echo "[$(ts)] seed ${s}: train start"
  "${PY}" main.py train --split "${SPLIT}" --class-weighting sqrt-inverse --seed "${s}" \
    --epochs "${EPOCHS}" --output-dir "${out}" --checkpoint "${ckpt}" --init-from "${mlm_ckpt}"
  echo "[$(ts)] seed ${s}: train done"
done

CKPTS=()
for s in "${SEEDS[@]}"; do
  if [ ! -f "outputs/${NAME}/seed${s}/summary.json" ]; then
    echo "[$(ts)] seed ${s} is not complete; skipping ensemble (rerun, or run tools.ensemble_checkpoints manually)"
    exit 0
  fi
  CKPTS+=("saved_models/${NAME}/seed${s}")
done

SEED_TAG=$(IFS=-; echo "${SEEDS[*]}")
ENS_OUT="outputs/${NAME}_ensemble_seed${SEED_TAG}"
if [ -f "${ENS_OUT}/summary.json" ]; then
  echo "[$(ts)] ensemble exists, skipping (${ENS_OUT})"
  exit 0
fi
if [ -e "${ENS_OUT}" ]; then
  echo "[$(ts)] ${ENS_OUT} exists but is incomplete (no summary.json); remove or inspect it before rerunning" >&2
  exit 1
fi
echo "[$(ts)] ensemble: start"
"${PY}" -m tools.ensemble_checkpoints --split "${SPLIT}" --checkpoints "${CKPTS[@]}" \
  --output-dir "${ENS_OUT}"
echo "[$(ts)] ensemble: done"
