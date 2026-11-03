#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ALIGNMENT_ENV_NAME:-alignment-toolkit}"
WITH_VERL=0
WITH_OPENRLHF=0

usage() {
    cat <<'EOF'
Usage: scripts/setup_env.sh [--all-backends] [--with-verl] [--with-openrlhf]

Creates or updates the Conda environment used by Alignment Toolkit.
The default environment includes PyTorch CUDA 12.4 and the Hugging Face/TRL
training stack. Optional backend flags install veRL or OpenRLHF as well.
EOF
}

while (($#)); do
    case "$1" in
        --all-backends) WITH_VERL=1; WITH_OPENRLHF=1 ;;
        --with-verl) WITH_VERL=1 ;;
        --with-openrlhf) WITH_OPENRLHF=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required; install Miniconda or Anaconda first" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "warning: nvidia-smi was not found; install an NVIDIA driver before GPU training" >&2
fi

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
    conda env update -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml" --prune
else
    conda env create -n "$ENV_NAME" -f "$PROJECT_ROOT/environment.yml"
fi

PIP_EXTRAS="train,test"
if ((WITH_VERL)); then
    PIP_EXTRAS="$PIP_EXTRAS,verl"
fi
if ((WITH_OPENRLHF)); then
    PIP_EXTRAS="$PIP_EXTRAS,openrlhf"
fi

conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install -e "$PROJECT_ROOT[$PIP_EXTRAS]"
conda run --no-capture-output -n "$ENV_NAME" python -m pip check
conda run --no-capture-output -n "$ENV_NAME" \
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"

cat <<EOF

Environment ready: $ENV_NAME
Activate it with:
  conda activate $ENV_NAME
EOF
