#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full"
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-python}"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="${WORK_DIR:-exp/${CFG_NAME}}"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
MASTER_PORT="${MASTER_PORT:-21873}"

EPOCH="${EPOCH:-10}"
if [[ "${EPOCH}" == "latest" ]]; then
  CKPT_DIR="${WORK_DIR}/checkpoints"
  CKPT_FILE=""
  latest_epoch=-1
  for candidate in "${CKPT_DIR}"/epoch_*.pth; do
    [[ -f "${candidate}" && "${candidate##*/}" =~ ^epoch_([0-9]+)\.pth$ ]] || continue
    epoch=$((10#${BASH_REMATCH[1]}))
    if (( epoch > latest_epoch )); then
      CKPT_FILE="${candidate}"
      latest_epoch=${epoch}
    fi
  done
  if [[ -z "${CKPT_FILE}" ]]; then
    echo "[FATAL] no epoch_*.pth found under ${CKPT_DIR}" >&2
    exit 2
  fi
  CKPT="${CKPT_FILE}"
  EPOCH_TAG="latest_$(basename "${CKPT_FILE}" .pth)"
else
  CKPT="${CKPT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
  EPOCH_TAG="epoch_${EPOCH}"
fi
LOG_NAME="${LOG_NAME:-test_${EPOCH_TAG}_v1}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[FATAL] checkpoint not found: ${CKPT}" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"

{
  echo "=== test launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${CFG_NAME}"
  echo "config     : ${CONFIG}"
  echo "ckpt       : ${CKPT}"
  echo "gpus       : ${GPUS}  port: ${MASTER_PORT}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unavailable)"
  echo "git commit : $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" \
  test.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CKPT}" \
  --log-name "${LOG_NAME}" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
