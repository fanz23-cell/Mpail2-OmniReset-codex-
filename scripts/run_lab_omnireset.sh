#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
NUM_ENVS="${NUM_ENVS:-32}"
DEMO_PATH="${DEMO_PATH:-${REPO_ROOT}/mpail2/train/isaac_franka/demos/omnireset_peg_state_filtered.pt}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

CONDA="${CONDA_EXE:-$(command -v conda 2>/dev/null || echo "${HOME}/miniconda3/bin/conda")}"

CMD=(
  "${CONDA}" run -n "${ENV_NAME}"
  python -m mpail2.train.train
  --env omnireset_peg
  --headless
  log.no_wandb=True
  log.video=False
  enable_cameras=False
  "num_envs=${NUM_ENVS}"
)

if [[ -n "${DEMO_PATH}" ]]; then
  CMD+=("runner.path_to_demonstrations=${DEMO_PATH}")
fi

LOG_FILE="${LOG_FILE:-/tmp/mpail2_train.log}"
echo "[run_lab] logging to ${LOG_FILE}"
"${CMD[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
