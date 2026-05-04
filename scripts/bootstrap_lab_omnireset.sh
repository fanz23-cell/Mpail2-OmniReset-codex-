#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="mpail2-omnireset"
PYTHON_VERSION="3.11"
RUN_SMOKE_TEST=1
RUN_TRAIN=0
NUM_ENVS=1
DEMO_SRC=""
DEMO_DST="${REPO_ROOT}/mpail2/train/isaac_franka/demos/omnireset_peg_state.pt"
SKIP_APT=0

UWLAB_REPO_URL="https://github.com/UW-Lab/UWLab.git"
UWLAB_COMMIT="36d98afe1166f546083fc6e3c5d5bee04b486d84"
UWLAB_DIR="${REPO_ROOT}/_vendor/UWLab"

ISAACLAB_REPO_URL="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_TAG="v2.3.2"
ISAACLAB_DIR="${REPO_ROOT}/_isaaclab/IsaacLab"

PYTORCH3D_WHEEL="https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/pytorch3d-0.7.8%2Bpt2.7.0cu128-cp311-cp311-linux_x86_64.whl"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bootstrap_lab_omnireset.sh [options]

Options:
  --env-name NAME          Conda env name. Default: mpail2-omnireset
  --num-envs N             Num envs for smoke/train command. Default: 1
  --demo-src PATH          Source OmniReset demo file (.pt or .zarr)
  --demo-dst PATH          Converted MPAIL demo output path
  --train                  Convert demo if provided, then launch mpail2 OmniReset train
  --skip-smoke             Skip environment smoke-test
  --skip-apt               Do not attempt apt install of system packages
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --demo-src)
      DEMO_SRC="$2"
      shift 2
      ;;
    --demo-dst)
      DEMO_DST="$2"
      shift 2
      ;;
    --train)
      RUN_TRAIN=1
      shift
      ;;
    --skip-smoke)
      RUN_SMOKE_TEST=0
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

run_env() {
  conda run -n "${ENV_NAME}" "$@"
}

need_cmd git
need_cmd conda

if [[ "${SKIP_APT}" -eq 0 ]] && command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential cmake git git-lfs pkg-config
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

run_env python -m pip install --upgrade pip

run_env python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0

run_env python -m pip install --no-cache-dir \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

mkdir -p "$(dirname "${ISAACLAB_DIR}")"
if [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
  git clone --branch "${ISAACLAB_TAG}" --depth 1 "${ISAACLAB_REPO_URL}" "${ISAACLAB_DIR}"
fi

mkdir -p "$(dirname "${UWLAB_DIR}")"
if [[ ! -d "${UWLAB_DIR}/.git" ]]; then
  git clone "${UWLAB_REPO_URL}" "${UWLAB_DIR}"
fi
git -C "${UWLAB_DIR}" fetch --depth 1 origin "${UWLAB_COMMIT}"
git -C "${UWLAB_DIR}" checkout "${UWLAB_COMMIT}"

run_env python -m pip install --no-cache-dir \
  psutil hydra-core omegaconf pybullet cmaes iopath typeguard \
  "zarr==2.18.3" \
  "numcodecs==0.13.1" \
  "pytorch3d @ ${PYTORCH3D_WHEEL}"

run_env python -m pip install --no-cache-dir --no-deps \
  -e "${ISAACLAB_DIR}/source/isaaclab" \
  -e "${ISAACLAB_DIR}/source/isaaclab_assets" \
  -e "${ISAACLAB_DIR}/source/isaaclab_tasks" \
  -e "${ISAACLAB_DIR}/source/isaaclab_rl"

run_env python -m pip install --no-cache-dir --no-deps \
  -e "${UWLAB_DIR}/source/uwlab" \
  -e "${UWLAB_DIR}/source/uwlab_assets" \
  -e "${UWLAB_DIR}/source/uwlab_rl" \
  -e "${UWLAB_DIR}/source/uwlab_tasks"

run_env python -m pip install --no-cache-dir \
  av imageio imageio-ffmpeg wandb "gymnasium[mujoco]>=1.0"

run_env python -m pip install --no-cache-dir --no-deps -e "${REPO_ROOT}"

ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"

mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/omnireset_env.sh" <<EOF
export OMNI_KIT_ACCEPT_EULA=YES
export MPLCONFIGDIR="\${CONDA_PREFIX}/.cache/mpl"
mkdir -p "\${MPLCONFIGDIR}"
EOF
cat > "${DEACTIVATE_DIR}/omnireset_env.sh" <<'EOF'
unset OMNI_KIT_ACCEPT_EULA
unset MPLCONFIGDIR
EOF

export OMNI_KIT_ACCEPT_EULA=YES
export MPLCONFIGDIR="/tmp/mpl"
mkdir -p "${MPLCONFIGDIR}"

run_env python scripts/check_omnireset_readiness.py

if [[ "${RUN_SMOKE_TEST}" -eq 1 ]]; then
  run_env python scripts/omnireset_env_smoke.py --headless --num-envs "${NUM_ENVS}"
fi

if [[ -n "${DEMO_SRC}" ]]; then
  mkdir -p "$(dirname "${DEMO_DST}")"
  run_env python scripts/convert_omnireset_demo.py "${DEMO_SRC}" "${DEMO_DST}"
fi

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  if [[ -z "${DEMO_SRC}" && ! -f "${DEMO_DST}" ]]; then
    echo "Training requested but no converted demo was found. Pass --demo-src /abs/path/to/demo.pt or .zarr" >&2
    exit 1
  fi
  DEMO_PATH_TO_USE="${DEMO_DST}"
  if [[ -z "${DEMO_SRC}" && -f "${DEMO_DST}" ]]; then
    DEMO_PATH_TO_USE="${DEMO_DST}"
  fi
  OMNI_KIT_ACCEPT_EULA=YES MPLCONFIGDIR=/tmp/mpl DEMO_PATH="${DEMO_PATH_TO_USE}" NUM_ENVS="${NUM_ENVS}" \
    bash "${REPO_ROOT}/scripts/run_lab_omnireset.sh"
fi

echo ""
echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Smoke test with: conda run -n ${ENV_NAME} python scripts/omnireset_env_smoke.py --headless --num-envs ${NUM_ENVS}"
echo "Train with: DEMO_PATH=${DEMO_DST} NUM_ENVS=${NUM_ENVS} bash scripts/run_lab_omnireset.sh"
