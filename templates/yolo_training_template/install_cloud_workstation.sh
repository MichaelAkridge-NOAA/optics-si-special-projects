#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${ENV_NAME:-yolo-cloud}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
SHM_SIZE="${SHM_SIZE:-16G}"
UPDATE_SHM="${UPDATE_SHM:-true}"
REPO_URL="${REPO_URL:-https://github.com/MichaelAkridge-NOAA/optics-si-special-projects.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REPO_DIR="${REPO_DIR:-$HOME/optics-si-special-projects}"
ACCEPT_ANACONDA_TOS="${ACCEPT_ANACONDA_TOS:-false}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-true}"

log() {
    printf '\n[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

if ! command -v conda >/dev/null 2>&1; then
    printf 'ERROR: conda is required but was not found on PATH.\n' >&2
    exit 1
fi

if [[ "$INSTALL_SYSTEM_PACKAGES" == "true" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing common workstation system packages"
        if [[ "$(id -u)" -eq 0 ]]; then
            apt-get update
            DEBIAN_FRONTEND=noninteractive apt-get install -y \
                git \
                wget \
                unzip \
                zip \
                libgl1 \
                libglib2.0-0 \
                libsm6 \
                libxext6
        elif command -v sudo >/dev/null 2>&1; then
            sudo apt-get update
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
                git \
                wget \
                unzip \
                zip \
                libgl1 \
                libglib2.0-0 \
                libsm6 \
                libxext6
        else
            printf 'WARNING: sudo is unavailable; skipping system package install. Missing libGL/libglib errors may need administrator help.\n' >&2
        fi
    else
        printf 'WARNING: apt-get was not found; skipping system package install.\n' >&2
    fi
fi

if ! command -v git >/dev/null 2>&1; then
    printf 'ERROR: git is required but was not found on PATH.\n' >&2
    exit 1
fi

log "Syncing project repository into $REPO_DIR"
if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH"
elif [[ -e "$REPO_DIR" ]]; then
    printf 'ERROR: %s exists but is not a git repository. Set REPO_DIR to another path or move it aside.\n' "$REPO_DIR" >&2
    exit 1
else
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
fi

log "Loading conda shell support"
eval "$(conda shell.bash hook)"

if [[ "$ACCEPT_ANACONDA_TOS" == "true" ]]; then
    log "Accepting Anaconda Terms of Service for default channels"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi

if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    log "Reusing conda environment: $ENV_NAME"
else
    log "Creating conda environment $ENV_NAME with Python $PYTHON_VERSION"
    create_log="$(mktemp)"
    if ! conda create --name "$ENV_NAME" "python=$PYTHON_VERSION" pip -y 2>&1 | tee "$create_log"; then
        if grep -q 'CondaToSNonInteractiveError' "$create_log"; then
            cat >&2 <<'EOF'

ERROR: Anaconda channel Terms of Service must be accepted before conda can create this environment.

Review the terms, then run either:
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  ./install_cloud_workstation.sh

Or rerun this installer with explicit opt-in:
  ACCEPT_ANACONDA_TOS=true ./install_cloud_workstation.sh
EOF
        fi
        rm -f "$create_log"
        exit 1
    fi
    rm -f "$create_log"
fi

log "Upgrading pip build tools"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel

log "Installing CUDA-enabled PyTorch from $PYTORCH_INDEX_URL"
conda run -n "$ENV_NAME" python -m pip install --upgrade \
    --index-url "$PYTORCH_INDEX_URL" \
    torch torchvision torchaudio

log "Installing YOLO, cloud, Label Studio, and Jupyter dependencies"
conda run -n "$ENV_NAME" python -m pip install --upgrade \
    'ultralytics>=8.4.92' \
    'PyYAML>=6.0' \
    requests \
    pillow \
    matplotlib \
    numpy \
    pandas \
    label-studio-sdk \
    google-cloud-storage \
    ipykernel \
    jupyterlab

log "Checking installed Python packages for dependency conflicts"
conda run -n "$ENV_NAME" python -m pip check

log "Registering the conda environment as a Jupyter kernel"
conda run -n "$ENV_NAME" python -m ipykernel install --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

log "Checking Jupyter kernel registration"
conda run -n "$ENV_NAME" python -m jupyter kernelspec list

log "Checking optional workstation tools"
if command -v gcloud >/dev/null 2>&1; then
    gcloud --version | head -n 1
else
    printf 'WARNING: gcloud was not found. Install Google Cloud CLI before using GCS cells.\n' >&2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
else
    printf 'WARNING: nvidia-smi was not found. This workstation may not expose an NVIDIA GPU.\n' >&2
fi

if [[ "$UPDATE_SHM" == "true" ]]; then
    log "Requesting /dev/shm size $SHM_SIZE"
    if [[ ! -d /dev/shm ]]; then
        printf 'WARNING: /dev/shm does not exist; skipping shared-memory resize.\n' >&2
    elif [[ "$(id -u)" -eq 0 ]]; then
        mount -o remount,size="$SHM_SIZE" /dev/shm || \
            printf 'WARNING: Could not remount /dev/shm. Update the workstation/container configuration instead.\n' >&2
    elif command -v sudo >/dev/null 2>&1; then
        sudo mount -o remount,size="$SHM_SIZE" /dev/shm || \
            printf 'WARNING: Could not remount /dev/shm. Sudo or container mount privileges may be unavailable.\n' >&2
    else
        printf 'WARNING: sudo is unavailable; skipping /dev/shm resize.\n' >&2
    fi
fi

log "Verifying the Python training environment"
conda run -n "$ENV_NAME" python -c \
    'import importlib; modules = ["torch", "ultralytics", "matplotlib", "requests", "yaml", "PIL", "numpy", "pandas", "google.cloud.storage", "label_studio_sdk", "ipykernel", "jupyterlab"]; [importlib.import_module(module) for module in modules]; import torch, ultralytics, matplotlib; print({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "ultralytics": ultralytics.__version__, "matplotlib": matplotlib.__version__})'

if command -v df >/dev/null 2>&1 && [[ -d /dev/shm ]]; then
    df -h /dev/shm
fi

cat <<EOF

Setup complete.

1. Refresh JupyterLab in the browser.
2. Open the notebook and select the "Python ($ENV_NAME)" kernel.
3. If imports fail in a notebook, confirm the notebook is using this kernel:
    import sys; print(sys.executable)
4. Run: gcloud auth login
5. Run: gcloud auth application-default login

The /dev/shm remount is runtime-only on many managed workstations and containers.
Configure the workstation image or container runtime for a persistent shared-memory size.
EOF