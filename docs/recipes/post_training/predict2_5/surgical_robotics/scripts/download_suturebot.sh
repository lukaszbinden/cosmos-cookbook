#!/usr/bin/env bash
# Download SutureBot dataset from Hugging Face.
# Set SUTUREBOT_DATASET_DIR to the target directory before running:
#   export SUTUREBOT_DATASET_DIR=/path/to/dataset/SutureBot
#   ./scripts/download_suturebot.sh

set -euo pipefail

if [[ -z "${SUTUREBOT_DATASET_DIR:-}" ]]; then
  echo "Error: SUTUREBOT_DATASET_DIR is not set. Export it to the target directory, e.g.:" >&2
  echo "  export SUTUREBOT_DATASET_DIR=/path/to/dataset/SutureBot" >&2
  exit 1
fi

python - << EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jchen396/SutureBot",
    repo_type="dataset",
    local_dir="${SUTUREBOT_DATASET_DIR}",
    local_dir_use_symlinks=False,
)
EOF
