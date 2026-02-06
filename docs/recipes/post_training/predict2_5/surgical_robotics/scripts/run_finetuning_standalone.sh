#!/usr/bin/env bash
# Run finetuning on a single machine without Slurm.
# Set env vars (see below), then: ./scripts/run_finetuning_standalone.sh
# For multi-node Slurm, use run_finetuning.sh instead.

set -euo pipefail

# --- Config (override with env vars) ---
# Path to the cosmos-predict2.5-cookbook repo (clone from step 1.1). Default: current directory.
WORKSPACE="${COSMOS_WORKSPACE:-$PWD}"
# Path to the SutureBot dataset in LeRobot format (output of convert_suturebot_to_lerobot_v3.py).
# The training config may expect this at /SutureBot; if so, symlink: sudo ln -snf "$SUTUREBOT_LEROBOT_PATH" /SutureBot
SUTUREBOT_LEROBOT_PATH="${SUTUREBOT_LEROBOT_PATH:?Set SUTUREBOT_LEROBOT_PATH to the LeRobot dataset path (e.g. \$HF_LEROBOT_HOME/suturebot_lerobot)}"
# Number of GPUs to use on this machine (default: all visible).
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-25001}"

export LOGLEVEL="${LOGLEVEL:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTHONFAULTHANDLER=1

echo "WORKSPACE=$WORKSPACE"
echo "SUTUREBOT_LEROBOT_PATH=$SUTUREBOT_LEROBOT_PATH"
echo "NGPUS=$NGPUS"
echo "MASTER_ADDR=$MASTER_ADDR"

cd "$WORKSPACE"

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "Using conda env: $CONDA_DEFAULT_ENV"
else
  echo "Warning: no .venv found; ensure dependencies are installed (see setup.md)."
fi

# Optional: if the repo supports overriding dataset path via Hydra, uncomment and set:
# export SUTUREBOT_DATASET_OVERRIDE="dataset_path=$SUTUREBOT_LEROBOT_PATH"

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NGPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- \
  experiment="cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss" \
  checkpoint.save_iter=200 \
  ~dataloader_train.dataloaders
