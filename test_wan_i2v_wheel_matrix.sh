#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
LITELINEAR_WHEEL_URL="${LITELINEAR_WHEEL_URL:-https://raw.githubusercontent.com/moonmath-ai/LiteLinear/3f3f09bf2dd1775dbf43f4ffe5de4a3cff9a25c7/install/lite_linear-0.2.0%2Bcu128-cp310-cp310-linux_x86_64.whl}"

PY_BIN="${WAN_WHEEL_MATRIX_PYTHON:-/root/miniconda3/envs/wan2114b-ll/bin/python}"
OUTPUT_ROOT="${WAN_WHEEL_MATRIX_OUTPUT_ROOT:-${REPO_ROOT}/benchmarks/wan_i2v_wheel_matrix_$(date -u +%Y%m%d_%H%M%S)}"
WHEELHOUSE="${OUTPUT_ROOT}/wheelhouse"
CACHE_BASE="${WAN_WHEEL_MATRIX_CACHE_BASE:-${OUTPUT_ROOT}/litelinear_cache}"
if [[ -z "${WAN_I2V_CKPT_DIR:-}" ]]; then
  if [[ -d "${REPO_ROOT}/Wan2.1-I2V-14B-720P" ]]; then
    WAN_I2V_CKPT_DIR="${REPO_ROOT}/Wan2.1-I2V-14B-720P"
  elif [[ -n "${HF_HOME:-}" && -d "${HF_HOME}/Wan2.1/Wan2.1-I2V-14B-720P" ]]; then
    WAN_I2V_CKPT_DIR="${HF_HOME}/Wan2.1/Wan2.1-I2V-14B-720P"
  elif [[ -f "/data/huggingface_cache/Wan2.1/config.json" && -f "/data/huggingface_cache/Wan2.1/diffusion_pytorch_model.safetensors.index.json" ]]; then
    WAN_I2V_CKPT_DIR="/data/huggingface_cache/Wan2.1"
  else
    WAN_I2V_CKPT_DIR="/data/huggingface_cache/Wan2.1/Wan2.1-I2V-14B-720P"
  fi
fi

FRAME_NUM="${WAN_WHEEL_MATRIX_FRAME_NUM:-9}"
SAMPLE_STEPS="${WAN_WHEEL_MATRIX_SAMPLE_STEPS:-10}"
SAMPLE_FPS="${WAN_WHEEL_MATRIX_SAMPLE_FPS:-16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

if [[ ! -x "${PY_BIN}" ]]; then
  echo "Python not executable: ${PY_BIN}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/run_i2v.sh" ]]; then
  echo "run_i2v.sh not found under ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${WAN_I2V_CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${WAN_I2V_CKPT_DIR}" >&2
  echo "Set WAN_I2V_CKPT_DIR to your Wan2.1-I2V-14B-720P directory." >&2
  exit 1
fi

mkdir -p "${WHEELHOUSE}" "${CACHE_BASE}"
MATRIX_LOG="${WAN_WHEEL_MATRIX_LOG_FILE:-${OUTPUT_ROOT}/matrix.log}"
touch "${MATRIX_LOG}"
exec > >(tee -a "${MATRIX_LOG}") 2>&1

echo "[wheel-matrix] repo_root=${REPO_ROOT}"
echo "[wheel-matrix] litelinear_wheel_url=${LITELINEAR_WHEEL_URL}"
echo "[wheel-matrix] python=${PY_BIN}"
echo "[wheel-matrix] ckpt_dir=${WAN_I2V_CKPT_DIR}"
echo "[wheel-matrix] output_root=${OUTPUT_ROOT}"
echo "[wheel-matrix] matrix_log=${MATRIX_LOG}"
echo "[wheel-matrix] frame_num=${FRAME_NUM} sample_steps=${SAMPLE_STEPS} sample_fps=${SAMPLE_FPS}"

echo "[wheel-matrix] downloading LiteLinear wheel"
WHEEL_PATH="$("${PY_BIN}" - "${LITELINEAR_WHEEL_URL}" "${WHEELHOUSE}" <<'PY'
import pathlib
import sys
import urllib.parse
import urllib.request
import zipfile

url = sys.argv[1]
wheelhouse = pathlib.Path(sys.argv[2])
wheelhouse.mkdir(parents=True, exist_ok=True)

filename = pathlib.Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
if not filename.endswith(".whl"):
    filename = "lite_linear-0.2.0+cu128-cp310-cp310-linux_x86_64.whl"
target = wheelhouse / filename

request = urllib.request.Request(url, headers={"User-Agent": "Wan2.1 wheel matrix"})
with urllib.request.urlopen(request) as response:
    target.write_bytes(response.read())

if not zipfile.is_zipfile(target):
    raise SystemExit(f"Downloaded LiteLinear wheel is not a valid wheel archive: {target}")
print(target)
PY
)"

echo "[wheel-matrix] installing ${WHEEL_PATH}"
"${PY_BIN}" -m pip install --force-reinstall --no-deps "${WHEEL_PATH}"

echo "[wheel-matrix] verifying wheel import is not from the repo tree"
PYTHONPATH="" REPO_ROOT="${REPO_ROOT}" "${PY_BIN}" <<'PY'
import os
import pathlib

import lite_linear
import lite_linear._cuda as litelinear_cuda

repo_root = pathlib.Path(os.environ["REPO_ROOT"]).resolve()
package_file = pathlib.Path(lite_linear.__file__).resolve()
cuda_file = pathlib.Path(litelinear_cuda.__file__).resolve()
print("lite_linear:", package_file)
print("lite_linear._cuda:", cuda_file)
if repo_root == package_file or repo_root in package_file.parents:
    raise SystemExit(f"lite_linear imported from repo tree: {package_file}")
if repo_root == cuda_file or repo_root in cuda_file.parents:
    raise SystemExit(f"lite_linear._cuda imported from repo tree: {cuda_file}")
PY

SUMMARY_TSV="${OUTPUT_ROOT}/matrix_summary.tsv"
printf "case\tdisabled\tonline_patch\tcache_state\texpected\tstatus\n" > "${SUMMARY_TSV}"

ensure_cache_present() {
  local cache_dir="${CACHE_BASE}/present"
  if [[ -d "${cache_dir}/patched_model_dirs" || -d "${cache_dir}/patched_models" ]]; then
    echo "${cache_dir}"
    return 0
  fi

  echo "[wheel-matrix] preparing cache-present state with one online-patch smoke run" >&2
  rm -rf "${cache_dir}"
  mkdir -p "${cache_dir}"
  local prep_dir="${OUTPUT_ROOT}/_prepare_cache_present"
  mkdir -p "${prep_dir}"
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    PYTHONPATH="" \
    WAN_I2V_CKPT_DIR="${WAN_I2V_CKPT_DIR}" \
    WAN_I2V_PYTHON="${PY_BIN}" \
    WAN_I2V_USE_LITELINEAR_SOURCE=0 \
    WAN_I2V_FRAME_NUM="${FRAME_NUM}" \
    WAN_I2V_SAMPLE_STEPS="${SAMPLE_STEPS}" \
    WAN_I2V_SAMPLE_FPS="${SAMPLE_FPS}" \
    WAN_I2V_LOG_FILE="${prep_dir}/run_i2v.log" \
    WAN_I2V_SAVE_FILE="${prep_dir}/out.mp4" \
    LITELINEAR_CACHE="${cache_dir}" \
    LITELINEAR_DISABLED=0 \
    LITELINEAR_ONLINE_PATCH=1 \
    "${REPO_ROOT}/run_i2v.sh" >&2
  echo "${cache_dir}"
}

run_case() {
  local disabled="$1"
  local online_patch="$2"
  local cache_state="$3"
  local expected="$4"

  local case_name="disabled${disabled}_online${online_patch}_cache${cache_state}"
  local case_dir="${OUTPUT_ROOT}/${case_name}"
  local cache_dir

  mkdir -p "${case_dir}"
  if [[ "${cache_state}" == "present" ]]; then
    cache_dir="$(ensure_cache_present)"
  else
    cache_dir="${CACHE_BASE}/${case_name}"
    rm -rf "${cache_dir}"
    mkdir -p "${cache_dir}"
  fi

  echo "[wheel-matrix] case=${case_name} expected=${expected}"
  set +e
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    PYTHONPATH="" \
    WAN_I2V_CKPT_DIR="${WAN_I2V_CKPT_DIR}" \
    WAN_I2V_PYTHON="${PY_BIN}" \
    WAN_I2V_USE_LITELINEAR_SOURCE=0 \
    WAN_I2V_FRAME_NUM="${FRAME_NUM}" \
    WAN_I2V_SAMPLE_STEPS="${SAMPLE_STEPS}" \
    WAN_I2V_SAMPLE_FPS="${SAMPLE_FPS}" \
    WAN_I2V_LOG_FILE="${case_dir}/run_i2v.log" \
    WAN_I2V_SAVE_FILE="${case_dir}/out.mp4" \
    LITELINEAR_CACHE="${cache_dir}" \
    LITELINEAR_DISABLED="${disabled}" \
    LITELINEAR_ONLINE_PATCH="${online_patch}" \
    "${REPO_ROOT}/run_i2v.sh"
  local rc=$?
  set -e

  local status="fail"
  if [[ "${rc}" -eq 0 ]]; then
    status="pass"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${case_name}" "${disabled}" "${online_patch}" "${cache_state}" "${expected}" "${status}" >> "${SUMMARY_TSV}"

  if [[ "${expected}" == "pass" && "${status}" != "pass" ]]; then
    echo "[wheel-matrix] unexpected failure in ${case_name}; see ${case_dir}/run_i2v.log" >&2
    return 1
  fi
  if [[ "${expected}" == "fail" && "${status}" != "fail" ]]; then
    echo "[wheel-matrix] unexpected success in ${case_name}; expected strict cached mode to fail without cache" >&2
    return 1
  fi
  return 0
}

# Matrix:
# - disabled=1 should use dense/original weights regardless of online-patch or cache.
# - disabled=0, online=1 should build/reuse patched cache and pass.
# - disabled=0, online=0 is strict cached mode: pass only when cache is present.
run_case 1 0 absent pass
run_case 1 0 present pass
run_case 1 1 absent pass
run_case 1 1 present pass
run_case 0 1 absent pass
run_case 0 1 present pass
run_case 0 0 absent fail
run_case 0 0 present pass

echo "[wheel-matrix] wrote ${SUMMARY_TSV}"
echo "[wheel-matrix] full log: ${MATRIX_LOG}"
cat "${SUMMARY_TSV}"
