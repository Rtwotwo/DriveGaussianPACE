#!/usr/bin/env bash
set -euo pipefail

CFG="nuscenes_gs25600_v3_se3_full_vad"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
TORCHRUN="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${CFG}/${CFG}.py"
WORK_DIR="exp/${CFG}"
SOURCE_CKPT="${SOURCE_CKPT:-exp/nuscenes_gs25600_v3_se3_full/checkpoints/epoch_20.pth}"
FRONTEND_CKPT="${FRONTEND_CKPT:-${WORK_DIR}/bootstrap/v3_se3_full_epoch20_weights.pth}"

NNODES="${NNODES:-${PET_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
NPROC="${NPROC:-${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-8}}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-12876}}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-4}"

if [[ "${NNODES}" != "1" || "${NODE_RANK}" != "0" ]]; then
  echo "[FATAL] this isolated launch script supports one node only" >&2
  exit 2
fi
case "${NPROC}" in
  8) DEFAULT_LR="2e-4" ;;
  4) DEFAULT_LR="1e-4" ;;
  2) DEFAULT_LR="5e-5" ;;
  1) DEFAULT_LR="2.5e-5" ;;
  *) echo "[FATAL] NPROC must be 1, 2, 4 or 8" >&2; exit 2 ;;
esac
LR="${LR:-${DEFAULT_LR}}"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC - 1)))}}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if (( ${#GPU_IDS[@]} != NPROC )); then
  echo "[FATAL] GPUS=${GPUS} has ${#GPU_IDS[@]} IDs, NPROC=${NPROC}" >&2
  exit 2
fi

cd "${REPO}"
[[ -f "${SOURCE_CKPT}" ]] || {
  echo "[FATAL] Stage-1 epoch20 checkpoint not found: ${SOURCE_CKPT}" >&2
  exit 2
}
mkdir -p "${WORK_DIR}/bootstrap"
if [[ ! -f "${FRONTEND_CKPT}" ]]; then
  "${PYTHON}" "config/${CFG}/prepare_frontend_checkpoint.py" \
    --input "${SOURCE_CKPT}" \
    --output "${FRONTEND_CKPT}"
fi

VISIBLE_GPUS=$(CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -c \
  'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPUS}" != "${NPROC}" ]]; then
  echo "[FATAL] PyTorch sees ${VISIBLE_GPUS} GPUs, expected ${NPROC}" >&2
  exit 2
fi

export FRONTEND_CKPT LR MAX_EPOCHS EVAL_EVERY_EPOCHS
unset RANK WORLD_SIZE LOCAL_RANK

{
  echo "=== Stage-2 current-Gaussian VAD launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "experiment : ${CFG}"
  echo "source     : ${SOURCE_CKPT} (V3-SE3 full epoch20)"
  echo "frontend   : frozen/no_grad/eval"
  echo "planner    : original VADHead, random initialization"
  echo "inputs     : predicted current Gaussian + agents + map only"
  echo "data       : full nuScenes train + full val"
  echo "schedule   : ${MAX_EPOCHS} epochs, evaluate every ${EVAL_EVERY_EPOCHS}"
  echo "batch      : per_gpu=1, global=${NPROC}"
  echo "lr         : ${LR}"
  echo "gpus       : ${GPUS}"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  train.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --dataset nuscenes \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"
