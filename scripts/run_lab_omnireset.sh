#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
NUM_ENVS="${NUM_ENVS:-1}"
DEMO_PATH="${DEMO_PATH:-}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}"

CMD=(
  conda run -n "${ENV_NAME}"
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

"${CMD[@]}" "$@"
