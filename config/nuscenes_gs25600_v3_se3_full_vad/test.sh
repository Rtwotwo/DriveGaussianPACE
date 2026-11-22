#!/usr/bin/env bash
set -euo pipefail

CFG="nuscenes_gs25600_v3_se3_full_vad"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${CFG}/${CFG}.py"
EVALUATOR="config/${CFG}/eval_plan_v1.py"
WORK_DIR="exp/${CFG}"

EPOCH="${EPOCH:-20}"
CKPT="${CKPT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-21876}"
LOG_NAME="${LOG_NAME:-test_epoch_${EPOCH}_plan_v1_no_cumsum}"

cd "${REPO}"
[[ -f "${CKPT}" ]] || {
  echo "[FATAL] checkpoint not found: ${CKPT}" >&2
  exit 2
}
unset RANK WORLD_SIZE LOCAL_RANK

echo "[PLAN EVAL] full nuScenes, current predicted Gaussian only, epoch=${EPOCH}, gpus=${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
"${PYTHON}" "${EVALUATOR}" \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CKPT}" \
  --log-name "${LOG_NAME}" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
