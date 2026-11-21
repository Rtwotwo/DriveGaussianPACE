#!/usr/bin/env bash
set -euo pipefail

CFG="nuscenes_gs25600_v3_se3_full"
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-python}"
CONFIG="config/${CFG}/${CFG}.py"
EVALUATOR="config/${CFG}/eval_occ.py"
WORK_DIR="${WORK_DIR:-exp/${CFG}}"

EPOCH="${EPOCH:-20}"
CKPT="${CKPT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
MASTER_PORT="${MASTER_PORT:-21874}"
LOG_NAME="${LOG_NAME:-test_epoch_${EPOCH}_occ}"

[[ -f "${CKPT}" ]] || { echo "[FATAL] checkpoint not found: ${CKPT}" >&2; exit 2; }

mkdir -p "${WORK_DIR}"
CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" \
  "${EVALUATOR}" \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CKPT}" \
  --log-name "${LOG_NAME}" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
