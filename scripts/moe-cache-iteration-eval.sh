#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=${1:?Usage: bash scripts/moe-cache-iteration-eval.sh model.gguf [output-directory]}
RUN_DIR=${2:-"${ROOT_DIR}/moe-cache-eval/iteration-$(date +%Y%m%d-%H%M%S)"}
CMAKE_BIN=${CMAKE_BIN:-/opt/homebrew/bin/cmake}
CORE_RUNS=${CORE_RUNS:-3}
FEATURE_RUNS=${FEATURE_RUNS:-2}
BUDGET_RUNS=${BUDGET_RUNS:-2}
VALIDATION_RUNS=${VALIDATION_RUNS:-1}
PREFILL_EXTRA_RUNS=${PREFILL_EXTRA_RUNS:-3}

cd "${ROOT_DIR}"
[[ -f "${MODEL_PATH}" ]] || { printf 'Model not found: %s\n' "${MODEL_PATH}" >&2; exit 1; }
mkdir -p "${RUN_DIR}"
if [[ "${BUILD:-1}" != 0 ]]; then
    "${CMAKE_BIN}" -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
        -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF -DGGML_CCACHE=OFF
    "${CMAKE_BIN}" --build build --target llama-completion test-moe-cache test-moe-profile test-backend-ops -j 8
fi
if [[ "${RUN_TESTS:-1}" != 0 ]]; then
    build/bin/test-moe-cache > "${RUN_DIR}/test-moe-cache.log" 2>&1
    build/bin/test-moe-profile > "${RUN_DIR}/test-moe-profile.log" 2>&1
    build/bin/test-backend-ops -b MTL0 -o MUL_MAT_ID \
        -p 'type_a=(f32|q4_K|q5_K|q6_K),.*m=512,n=(1|32),' \
        > "${RUN_DIR}/test-backend-ops.log" 2>&1
fi

for phase in core prefill policies budget validation; do
    case "${phase}" in
        core) repetitions=${CORE_RUNS} ;;
        budget) repetitions=${BUDGET_RUNS} ;;
        validation) repetitions=${VALIDATION_RUNS} ;;
        *) repetitions=${FEATURE_RUNS} ;;
    esac
    python3 scripts/moe-cache-iteration-eval.py "${MODEL_PATH}" \
        --output "${RUN_DIR}" --phase "${phase}" --repetitions "${repetitions}"
done
python3 scripts/moe-cache-iteration-reference.py "${MODEL_PATH}" --output "${RUN_DIR}"
if [[ "${PREFILL_EXTRA_RUNS}" -gt 0 ]]; then
    python3 scripts/moe-cache-iteration-eval.py "${MODEL_PATH}" --output "${RUN_DIR}" \
        --phase prefill --repetitions "$((FEATURE_RUNS + PREFILL_EXTRA_RUNS))"
fi
printf 'Results: %s/report.md\n' "${RUN_DIR}"
