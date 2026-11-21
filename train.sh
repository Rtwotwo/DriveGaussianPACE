#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
num_gpus="${NUM_GPUS:-$("${PYTHON:-python}" -c 'import torch; print(torch.cuda.device_count())')}"
if [[ ! "${num_gpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Training requires at least one visible GPU." >&2
  exit 2
fi
exec "${TORCHRUN:-torchrun}" --nproc_per_node "${num_gpus}" \
  --master_port "${MASTER_PORT:-$((10000 + RANDOM % 1000))}" train.py "$@"
