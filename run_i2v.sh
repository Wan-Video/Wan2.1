#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export LITELINEAR_DISABLED="${LITELINEAR_DISABLED:-0}"
export WAN_I2V_SAMPLE_STEPS="${WAN_I2V_SAMPLE_STEPS:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
LOG_FILE="${WAN_I2V_LOG_FILE:-${REPO_ROOT}/run_i2v.log}"
export LITELINEAR_ONLINE_PATCH="${LITELINEAR_ONLINE_PATCH:-1}"
LITELINEAR_SOURCE_ROOT="${WAN_I2V_LITELINEAR_SOURCE_ROOT:-${REPO_ROOT}/LiteLinear}"
LITELINEAR_SOURCE_ACTIVE=0
case "${WAN_I2V_USE_LITELINEAR_SOURCE:-0}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    if [[ ! -d "${LITELINEAR_SOURCE_ROOT}" ]]; then
      echo "LiteLinear source directory not found: ${LITELINEAR_SOURCE_ROOT}"
      echo "Install the lite_linear wheel or set WAN_I2V_LITELINEAR_SOURCE_ROOT to a source checkout."
      exit 1
    fi
    export PYTHONPATH="${LITELINEAR_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    LITELINEAR_SOURCE_ACTIVE=1
    ;;
esac

# This is the same base prompt used in the Wan README. If prompt extension is
# enabled, Wan expands this text rather than replacing it from scratch.
DEFAULT_PROMPT="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

is_truthy() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN_DEFAULT="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN_DEFAULT="python3"
fi

PYTHON_BIN="${WAN_I2V_PYTHON:-${PYTHON_BIN_DEFAULT}}"
TASK="${WAN_I2V_TASK:-i2v-14B}"
SIZE="${WAN_I2V_SIZE:-1280*720}"
if [[ -n "${WAN_I2V_CKPT_DIR:-}" ]]; then
  CKPT_DIR="${WAN_I2V_CKPT_DIR}"
  CKPT_ROOT="$(dirname "${CKPT_DIR}")"
else
  if [[ -n "${HF_HOME:-}" ]]; then
    CKPT_ROOT="${HF_HOME}/Wan2.1"
  else
    CKPT_ROOT="${REPO_ROOT}"
  fi
  mkdir -p "${CKPT_ROOT}"
  CKPT_DIR="${CKPT_ROOT}/Wan2.1-I2V-14B-720P"
fi
LITELINEAR_PATCH_CONFIG_DEFAULT="${REPO_ROOT}/litelinear.config"
if [[ -z "${LITELINEAR_PATCH_CONFIG:-}" && -f "${LITELINEAR_PATCH_CONFIG_DEFAULT}" ]]; then
  export LITELINEAR_PATCH_CONFIG="${LITELINEAR_PATCH_CONFIG_DEFAULT}"
fi
IMAGE_PATH="${WAN_I2V_IMAGE:-${REPO_ROOT}/surf_cat_ref.jpeg}"
PROMPT="${WAN_I2V_PROMPT:-${DEFAULT_PROMPT}}"
FRAME_NUM="${WAN_I2V_FRAME_NUM:-49}"
OFFLOAD_MODEL="${WAN_I2V_OFFLOAD_MODEL:-False}"
SAMPLE_FPS="${WAN_I2V_SAMPLE_FPS:-24}"
USE_PROMPT_EXTEND="${WAN_I2V_USE_PROMPT_EXTEND:-0}"
PROMPT_EXTEND_METHOD="${WAN_I2V_PROMPT_EXTEND_METHOD:-local_qwen}"
PROMPT_EXTEND_MODEL="${WAN_I2V_PROMPT_EXTEND_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PROMPT_EXTEND_TARGET_LANG="${WAN_I2V_PROMPT_EXTEND_TARGET_LANG:-}"
FRAME_NUM_REQUESTED="${FRAME_NUM}"

if [[ ! "${FRAME_NUM}" =~ ^[0-9]+$ ]]; then
  echo "WAN_I2V_FRAME_NUM must be a positive integer, got: ${FRAME_NUM}"
  exit 1
fi

if (( FRAME_NUM < 1 )); then
  echo "WAN_I2V_FRAME_NUM must be >= 1, got: ${FRAME_NUM}"
  exit 1
fi

if (( (FRAME_NUM - 1) % 4 != 0 )); then
  FRAME_NUM=$(( FRAME_NUM + 4 - ((FRAME_NUM - 1) % 4) ))
  FRAME_NUM_ADJUSTMENT_NOTE="Adjusted frame_num from ${FRAME_NUM_REQUESTED} to ${FRAME_NUM} to satisfy Wan's 4n+1 requirement."
fi

if [[ ! -f "${REPO_ROOT}/generate.py" ]]; then
  echo "generate.py not found under ${REPO_ROOT}"
  exit 1
fi

if [[ ! -e "${IMAGE_PATH}" ]]; then
  echo "Input image not found: ${IMAGE_PATH}"
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}"
  if [[ -z "${WAN_I2V_CKPT_DIR:-}" ]]; then
    echo "Created checkpoint root: ${CKPT_ROOT}"
  fi
  echo "Set WAN_I2V_CKPT_DIR to your Wan2.1-I2V checkpoint directory."
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

cmd=(
  "${PYTHON_BIN}"
  "generate.py"
  "--task" "${TASK}"
  "--size" "${SIZE}"
  "--ckpt_dir" "${CKPT_DIR}"
  "--frame_num" "${FRAME_NUM}"
  "--offload_model" "${OFFLOAD_MODEL}"
  "--sample_fps" "${SAMPLE_FPS}"
  "--image" "${IMAGE_PATH}"
  "--prompt" "${PROMPT}"
)

if is_truthy "${USE_PROMPT_EXTEND}"; then
  cmd+=(
    "--use_prompt_extend"
    "--prompt_extend_method" "${PROMPT_EXTEND_METHOD}"
  )
  if [[ "${PROMPT_EXTEND_METHOD}" == "local_qwen" ]]; then
    cmd+=("--prompt_extend_model" "${PROMPT_EXTEND_MODEL}")
  fi
  if [[ -n "${PROMPT_EXTEND_TARGET_LANG}" ]]; then
    cmd+=("--prompt_extend_target_lang" "${PROMPT_EXTEND_TARGET_LANG}")
  fi
fi

if [[ -n "${WAN_I2V_SAVE_FILE:-}" ]]; then
  cmd+=("--save_file" "${WAN_I2V_SAVE_FILE}")
fi

if [[ -n "${WAN_I2V_BASE_SEED:-}" ]]; then
  cmd+=("--base_seed" "${WAN_I2V_BASE_SEED}")
fi

if [[ -n "${WAN_I2V_SAMPLE_STEPS:-40}" ]]; then
  cmd+=("--sample_steps" "${WAN_I2V_SAMPLE_STEPS}")
fi

if [[ -n "${WAN_I2V_SAMPLE_GUIDE_SCALE:-}" ]]; then
  cmd+=("--sample_guide_scale" "${WAN_I2V_SAMPLE_GUIDE_SCALE}")
fi

if [[ -n "${WAN_I2V_SAMPLE_SHIFT:-}" ]]; then
  cmd+=("--sample_shift" "${WAN_I2V_SAMPLE_SHIFT}")
fi

cmd+=("$@")

printf -v COMMAND_LINE '%q ' "${cmd[@]}"
COMMAND_LINE="${COMMAND_LINE% }"

echo
echo "========================================="
echo "run_i2v.sh ($(timestamp))"
echo "log_file=${LOG_FILE}"
echo "python=${PYTHON_BIN}"
echo "task=${TASK}"
echo "size=${SIZE}"
echo "ckpt_root=${CKPT_ROOT}"
echo "ckpt_dir=${CKPT_DIR}"
echo "frame_num=${FRAME_NUM}"
if [[ -n "${FRAME_NUM_ADJUSTMENT_NOTE:-}" ]]; then
  echo "${FRAME_NUM_ADJUSTMENT_NOTE}"
fi
if [[ -n "${LITELINEAR_PATCH_CONFIG:-}" ]]; then
  echo "litelinear_patch_config=${LITELINEAR_PATCH_CONFIG}"
fi
echo "litelinear_disabled=${LITELINEAR_DISABLED}"
echo "litelinear_online_patch=${LITELINEAR_ONLINE_PATCH}"
echo "litelinear_cache=${LITELINEAR_CACHE:-}"
echo "litelinear_source=${LITELINEAR_SOURCE_ACTIVE}"
if [[ "${LITELINEAR_SOURCE_ACTIVE}" -eq 1 ]]; then
  echo "litelinear_source_root=${LITELINEAR_SOURCE_ROOT}"
fi
echo "offload_model=${OFFLOAD_MODEL}"
echo "sample_fps=${SAMPLE_FPS}"
echo "image=${IMAGE_PATH}"
echo "prompt_extend=${USE_PROMPT_EXTEND}"
if is_truthy "${USE_PROMPT_EXTEND}"; then
  echo "prompt_extend_method=${PROMPT_EXTEND_METHOD}"
  if [[ "${PROMPT_EXTEND_METHOD}" == "local_qwen" ]]; then
    echo "prompt_extend_model=${PROMPT_EXTEND_MODEL}"
  fi
  if [[ -n "${PROMPT_EXTEND_TARGET_LANG}" ]]; then
    echo "prompt_extend_target_lang=${PROMPT_EXTEND_TARGET_LANG}"
  fi
fi
echo "command=${COMMAND_LINE}"
echo "========================================="

"${cmd[@]}"
