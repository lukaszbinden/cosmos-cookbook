#!/usr/bin/env bash
# Run finetuning on a single machine without Slurm.
# Set env vars (see below), then: ./scripts/run_finetuning_standalone.sh
# For multi-node Slurm, use run_finetuning.sh instead.
#
# You can run either inside a Docker container (recommended) or on the host.
# For Docker: build the image from the cosmos-predict2.5-cookbook repo first (see post_training.md §1.2.1),
# then set COSMOS_CONTAINER_IMAGE to that image tag.
# For gated model nvidia/Cosmos-Predict2.5-2B: accept the model terms on Hugging Face, then set HF_TOKEN
# (e.g. export HF_TOKEN=hf_xxx) so the container can download the checkpoint.

set -euo pipefail

# --- Config (override with env vars) ---
# Path to the cosmos-predict2.5-cookbook repo (training code). Default: /ephemeral/cosmos-predict2.5-cookbook.
CODE_PATH="${COSMOS_CODE_PATH:-/ephemeral/cosmos-predict2.5-cookbook}"
# Path to the SutureBot dataset in LeRobot format (output of convert_suturebot_to_lerobot_v3.py or create_mini_suturebot.py).
SUTUREBOT_LEROBOT_PATH="${SUTUREBOT_LEROBOT_PATH:?Set SUTUREBOT_LEROBOT_PATH to the LeRobot dataset path (e.g. \$HF_LEROBOT_HOME/suturebot_lerobot or .../suturebot_lerobot_mini)}"
# Docker image built from cosmos-predict2.5-cookbook (see post_training.md §1.2.1). If unset, run on host using CODE_PATH as workspace.
COSMOS_CONTAINER_IMAGE="${COSMOS_CONTAINER_IMAGE:-}"
# Number of GPUs to use (default: all visible).
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-25001}"

export LOGLEVEL="${LOGLEVEL:-INFO}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export PYTHONFAULTHANDLER=1

echo "CODE_PATH=$CODE_PATH"
echo "SUTUREBOT_LEROBOT_PATH=$SUTUREBOT_LEROBOT_PATH"
echo "NGPUS=$NGPUS"
echo "MASTER_ADDR=$MASTER_ADDR"
if [[ -n "$COSMOS_CONTAINER_IMAGE" ]]; then
  echo "Running inside container: $COSMOS_CONTAINER_IMAGE"
  if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HF_HOME:-}" ]]; then
    echo "Warning: HF_TOKEN and HF_HOME unset. Gated model nvidia/Cosmos-Predict2.5-2B may fail. Set HF_TOKEN (and accept model terms on HF) or HF_HOME."
  fi
  echo "On failure, traceback may be in \${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/torchelastic_error.json or run with NGPUS=1 to see it in the log."
fi

_run_train() {
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
    dataloader_train.dataset.dataset_path="$SUTUREBOT_LEROBOT_PATH" \
    ~dataloader_train.dataloaders
}

if [[ -n "$COSMOS_CONTAINER_IMAGE" ]]; then
  # Run training inside Docker; code and dataset are mounted (config uses /SutureBot for dataset).
  # Use --gpus all so container sees all GPUs; we spawn NGPUS processes (no CUDA_VISIBLE_DEVICES inside container to avoid device mapping issues).
  # To use fewer GPUs, set CUDA_VISIBLE_DEVICES or NVIDIA_VISIBLE_DEVICES on the host before running (e.g. CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_finetuning_standalone.sh) and set NGPUS to match.
  OUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}"
  DOCKER_GPU_FLAG="${DOCKER_GPU_FLAG:---gpus all}"
  HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
  ELASTIC_ERROR_FILE="$OUT_ROOT/torchelastic_error.json"
  DOCKER_ARGS=(
    $DOCKER_GPU_FLAG
    --ipc=host
    -v "$CODE_PATH:/workspace"
    -v "$SUTUREBOT_LEROBOT_PATH:/SutureBot:ro"
    -v "$OUT_ROOT:$OUT_ROOT"
    -e NGPUS="$NGPUS"
    -e MASTER_ADDR="$MASTER_ADDR"
    -e MASTER_PORT="$MASTER_PORT"
    -e LOGLEVEL="$LOGLEVEL"
    -e NCCL_DEBUG="$NCCL_DEBUG"
    -e PYTHONFAULTHANDLER="$PYTHONFAULTHANDLER"
    -e HF_TOKEN="${HF_TOKEN:-}"
    -e IMAGINAIRE_OUTPUT_ROOT="$OUT_ROOT"
    -e TORCHELASTIC_ERROR_FILE="$ELASTIC_ERROR_FILE"
  )
  # Mount HF cache so container can use host token/cache (e.g. from huggingface-cli login)
  if [[ -d "$HF_HOME_HOST" ]]; then
    DOCKER_ARGS+=( -v "$HF_HOME_HOST:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface )
  fi
  # Preflight: ensure GPUs are visible in container (avoids "CUDA device busy or unavailable").
  TRAIN_CMD='source .venv/bin/activate 2>/dev/null || true;
echo "=== GPU preflight ===";
nvidia-smi -L || true;
python -c "import torch; n=torch.cuda.device_count(); print(\"PyTorch sees\", n, \"GPU(s):\", [torch.cuda.get_device_name(i) for i in range(n)] if n else \"none\")" || true;
echo "=== Starting training ===";
torchrun --nnodes=1 --nproc_per_node="${NGPUS:-8}" --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-25001}" -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- experiment="cosmos_predict2p5_2B_action_conditioned_suturebot_13frame_4nodes_release_oss" checkpoint.save_iter=200 dataloader_train.dataset.dataset_path=/SutureBot ~dataloader_train.dataloaders'
  docker run -it --rm \
    "${DOCKER_ARGS[@]}" \
    -w /workspace \
    "$COSMOS_CONTAINER_IMAGE" \
    bash -c "$TRAIN_CMD"
else
  # Run on host: use CODE_PATH as workspace and venv there.
  cd "$CODE_PATH"
  if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
  elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "Using conda env: $CONDA_DEFAULT_ENV"
  else
    echo "Warning: no .venv found in $CODE_PATH; ensure dependencies are installed (see setup.md)."
  fi
  _run_train
fi
