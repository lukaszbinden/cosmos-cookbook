#!/bin/bash

# Detect the ultimate Brev user.
if id "nvidia" &>/dev/null; then
  RUN_USER="nvidia"
elif id "shadeform" &>/dev/null; then
  RUN_USER="shadeform"
elif id "ubuntu" &>/dev/null; then
  RUN_USER="ubuntu"
else
  RUN_USER="root"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Write inner script to temp file (avoids nested-quote syntax errors; SCRIPT_DIR baked in)
INNER_SCRIPT="/tmp/02_guided_cosmos_setup_inner_$$.sh"
cat << INNEREOF > "$INNER_SCRIPT"
#!/bin/bash
set -e
SCRIPT_DIR='$SCRIPT_DIR'

section() { echo ""; echo "=== Section \$1: \$2 ==="; }

section 1 "Install packages"
sudo apt-get update && sudo apt-get install -y git-lfs bc

section 2 "Find largest drive and set paths"
LARGEST_MOUNT=\$(df -h --output=size,target -x tmpfs -x devtmpfs -x squashfs | tail -n +2 | awk '{print \$1 " " \$2}' | sort -hr | head -1)
LARGEST_PATH=\$(echo "\$LARGEST_MOUNT" | awk '{print \$2}')
if [ "\$LARGEST_PATH" = "/" ]; then
  REPO_DIR="\$HOME/guided-cosmos-transfer2.5"
  HF_HOME=\$HOME/.cache/huggingface
  VENV_DIR=\$HOME/.venv_guided_transfer2.5
  TORCH_HOME=\$HOME/.cache/torch
  PIP_CACHE_DIR=\$HOME/.cache/pip
  I4H_DIR="\$HOME/i4h-workflows-internal"
  ROBOTIC_US_DIR="\$HOME/robotic_us_example"
else
  REPO_DIR="\${LARGEST_PATH}/guided-cosmos-transfer2.5"
  HF_HOME="\${LARGEST_PATH}/.cache/huggingface"
  VENV_DIR="\${LARGEST_PATH}/.venv_guided_transfer2.5"
  TORCH_HOME="\${LARGEST_PATH}/.cache/torch"
  PIP_CACHE_DIR="\${LARGEST_PATH}/.cache/pip"
  I4H_DIR="\${LARGEST_PATH}/i4h-workflows-internal"
  ROBOTIC_US_DIR="\${LARGEST_PATH}/robotic_us_example"
fi
echo "REPO_DIR=\$REPO_DIR"

section 3 "Cache dirs and symlinks"
sudo mkdir -p \$HF_HOME \$VENV_DIR \$TORCH_HOME \$PIP_CACHE_DIR \$I4H_DIR "\$ROBOTIC_US_DIR"
sudo chown -R $RUN_USER:$RUN_USER \$HF_HOME \$VENV_DIR \$TORCH_HOME \$PIP_CACHE_DIR \$I4H_DIR "\$ROBOTIC_US_DIR"
for d in huggingface torch pip; do
  [ -L "\$HOME/.cache/\$d" ] && rm -f "\$HOME/.cache/\$d"
  [ -d "\$HOME/.cache/\$d" ] && rm -rf "\$HOME/.cache/\$d"
done
if [ "\$LARGEST_PATH" != "/" ]; then
  mkdir -p \$HOME/.cache
  ln -sf \$HF_HOME \$HOME/.cache/huggingface
  ln -sf \$TORCH_HOME \$HOME/.cache/torch
  ln -sf \$PIP_CACHE_DIR \$HOME/.cache/pip
fi
chmod -R u+rwX \$HF_HOME \$TORCH_HOME \$PIP_CACHE_DIR

section 4 "GPU/CPU and run script"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_CPUS=\$(nproc)
OMP_NUM_THREADS=\$(echo "scale=0; \$NUM_CPUS/\$NUM_GPUS" | bc)
VOL="-v \$HF_HOME:/workspace/.cache/huggingface -v \$REPO_DIR:/workspace -v \$VENV_DIR:/workspace/.venv"
VOL="\$VOL -v \$I4H_DIR:/workspace/i4h-workflows-internal -v \$ROBOTIC_US_DIR:/workspace/robotic_us_example"
ENV="-e HOME=/workspace -e HF_HOME=/workspace/.cache/huggingface -e OMP_NUM_THREADS=\$OMP_NUM_THREADS -e NUM_GPUS=\$NUM_GPUS"
echo "docker run -it --rm --runtime=nvidia --network=host --ipc=host --entrypoint bash --name guided-transfer2.5 \$VOL \$ENV guided-transfer2.5" > /tmp/guided_transfer2.5_run_cmd
echo "Run script content written to /tmp/guided_transfer2.5_run_cmd"

section 5 "Clone repo and checkout branch"
if [ ! -d "\$REPO_DIR" ]; then
  echo "Cloning guided-cosmos-transfer2.5 to \$REPO_DIR..."
  gh repo clone maximilianofir/guided-cosmos-transfer2.5 "\$REPO_DIR"
else
  echo "Repository already exists at \$REPO_DIR"
fi
cd "\$REPO_DIR"
git checkout maxo/dli_testing
git lfs install
git lfs pull

# Populate I4H_DIR: copy from repo root if we're inside i4h-workflows-internal, else clone
I4H_REPO_ROOT="\$(cd "\$SCRIPT_DIR/../../../.." && pwd)"
if [ -d "\$I4H_REPO_ROOT/.git" ] && [ "\$I4H_REPO_ROOT" != "\$I4H_DIR" ]; then
  if [ -z "\$(ls -A "\$I4H_DIR" 2>/dev/null)" ]; then
    echo "Copying i4h-workflows-internal from \$I4H_REPO_ROOT to \$I4H_DIR"
    rsync -a "\$I4H_REPO_ROOT/" "\$I4H_DIR/"
  fi
elif [ -z "\$(ls -A "\$I4H_DIR" 2>/dev/null)" ]; then
  echo "Cloning i4h-workflows-internal to \$I4H_DIR..."
  gh repo clone isaac-for-healthcare/i4h-workflows-internal "\$I4H_DIR" -- --branch maxo/dli-cosmos-transfer --single-branch --depth 1
fi

section 6 "Build Docker image"
# Exclude .nv and other cache/restricted dirs from build context (avoids "open .nv: permission denied")
for entry in .nv .cache; do
  [ -e "\$entry" ] && { grep -qFx "\$entry" .dockerignore 2>/dev/null || echo "\$entry" >> .dockerignore; }
done
docker build --network=host --ulimit nofile=131071:131071 -f Dockerfile . -t guided-transfer2.5

echo ""
echo "Setup complete! Repository at: \$REPO_DIR"
echo "Run container with: \$SCRIPT_DIR/run_guided_transfer2.5_docker.sh"
echo "Attach with: \$SCRIPT_DIR/03-attach_guided_transfer2.5_docker.sh"
INNEREOF

# Run inner script as RUN_USER; report failure by section
echo "Running setup as user: $RUN_USER"
if sudo -u "$RUN_USER" bash "$INNER_SCRIPT"; then
  echo "[OK] All sections completed."
else
  echo "[FAIL] Setup failed in the section shown above." >&2
  rm -f "$INNER_SCRIPT"
  exit 1
fi
rm -f "$INNER_SCRIPT"

# Copy run script into brev_deployment (always; outer script has write permission)
if [ -f /tmp/guided_transfer2.5_run_cmd ]; then
  cp /tmp/guided_transfer2.5_run_cmd "$SCRIPT_DIR/run_guided_transfer2.5_docker.sh"
  chmod +x "$SCRIPT_DIR/run_guided_transfer2.5_docker.sh"
  rm -f /tmp/guided_transfer2.5_run_cmd
  echo "Created $SCRIPT_DIR/run_guided_transfer2.5_docker.sh"
else
  echo "[WARN] /tmp/guided_transfer2.5_run_cmd missing; run script not created." >&2
  exit 1
fi
