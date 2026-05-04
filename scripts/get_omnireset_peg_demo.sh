#!/usr/bin/env bash
# get_omnireset_peg_demo.sh
#
# Full pipeline:
#   [1/3] Download the pre-trained state expert checkpoint from HuggingFace
#   [2/3] Roll it out in Isaac Sim to collect (obs, next_obs) state transitions
#   [3/3] Convert to the MPAIL .pt format
#
# Usage:
#   bash scripts/get_omnireset_peg_demo.sh
#
# Optional env vars:
#   ENV_NAME   - conda env name (default: mpail2-omnireset)
#   NUM_ENVS   - parallel Isaac envs (default: 32)
#   NUM_STEPS  - total steps to collect (default: 5000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-mpail2-omnireset}"
NUM_ENVS="${NUM_ENVS:-32}"
NUM_STEPS="${NUM_STEPS:-5000}"

CKPT_URL="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts_finetuned/peg_state_rl_expert_finetuned_seed42.pt"
DEMO_DIR="${REPO_ROOT}/mpail2/train/isaac_franka/demos"
CKPT_PATH="${DEMO_DIR}/peg_expert_seed42.pt"
DEMO_RAW="${DEMO_DIR}/peg_state_raw.pt"
DEMO_MPAIL="${DEMO_DIR}/omnireset_peg_state.pt"

mkdir -p "${DEMO_DIR}"

# ── [1/3] Download expert checkpoint ────────────────────────────────────────
if [[ -f "${CKPT_PATH}" ]]; then
    echo "==> [1/3] Checkpoint already present: ${CKPT_PATH}"
else
    echo "==> [1/3] Downloading state expert checkpoint..."
    wget -O "${CKPT_PATH}" "${CKPT_URL}"
    echo "    Saved to: ${CKPT_PATH}"
fi

# ── [2/3] Collect state demo trajectories ───────────────────────────────────
echo "==> [2/3] Collecting state demos (${NUM_STEPS} steps, ${NUM_ENVS} envs)..."
echo "    This will open Isaac Sim — takes ~2-5 min to load."
OMNI_KIT_ACCEPT_EULA=YES MPLCONFIGDIR=/tmp/mpl \
conda run -n "${ENV_NAME}" python \
    "${SCRIPT_DIR}/collect_state_demos_omnireset.py" \
    --checkpoint "${CKPT_PATH}" \
    --output     "${DEMO_RAW}" \
    --num-envs   "${NUM_ENVS}" \
    --num-steps  "${NUM_STEPS}" \
    --headless

# ── [3/3] Convert to MPAIL format ───────────────────────────────────────────
echo "==> [3/3] Converting to MPAIL format..."
OMNI_KIT_ACCEPT_EULA=YES MPLCONFIGDIR=/tmp/mpl \
conda run -n "${ENV_NAME}" python \
    "${SCRIPT_DIR}/convert_omnireset_demo.py" \
    "${DEMO_RAW}" "${DEMO_MPAIL}"

echo ""
echo "Demo ready: ${DEMO_MPAIL}"
echo ""
echo "Next: start training with"
echo "  bash scripts/run_lab_omnireset.sh"
