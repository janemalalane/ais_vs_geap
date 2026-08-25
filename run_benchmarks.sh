#!/usr/bin/env bash
# ==============================================================================
# Multi-Surface Gemini Performance Benchmark Runner
# ==============================================================================
# Dynamically resolves auth tokens, timestamps output report directories,
# and executes inference-perf across all benchmark configurations (AI Studio & GEAP).
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. Activate virtual environment if available
if [[ -d ".venv" && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# 2. Load environment variables from .env if present
if [[ -f ".env" ]]; then
    echo "[INFO] Loading configuration from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# 3. Generate run timestamp and output directories
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORTS_DIR="reports"
RUN_DIR="${REPORTS_DIR}/report_${TIMESTAMP}"
AI_STUDIO_OUTPUT_DIR="${RUN_DIR}/ai_studio"
GEAP_OUTPUT_DIR="${RUN_DIR}/geap"

mkdir -p "${AI_STUDIO_OUTPUT_DIR}" "${GEAP_OUTPUT_DIR}"

echo "======================================================================"
echo "          🚀 GEMINI MULTI-SURFACE BENCHMARK HARNESS RUNNER"
echo "======================================================================"
echo "Run Timestamp       : ${TIMESTAMP}"
echo "Run Directory       : ${RUN_DIR}/"
echo "AI Studio Reports   : ${AI_STUDIO_OUTPUT_DIR}/"
echo "GEAP Reports        : ${GEAP_OUTPUT_DIR}/"
echo "======================================================================"

# 4. Function to refresh gcloud OAuth access token for GEAP
refresh_auth_token() {
    echo "[AUTH] Refreshing access token via gcloud CLI..."
    local token=""
    if command -v gcloud &> /dev/null; then
        token=$(gcloud auth print-access-token 2>/dev/null || true)
    fi

    if [[ -n "${token}" ]]; then
        AUTH_TOKEN="${token}"
        export AUTH_TOKEN
        echo "[AUTH] Fresh access token acquired from gcloud (${AUTH_TOKEN:0:15}...)"
        return 0
    elif [[ -n "${AUTH_TOKEN:-}" ]]; then
        echo "[AUTH] Using existing AUTH_TOKEN from environment/.env (${AUTH_TOKEN:0:15}...)"
        return 0
    else
        echo "[ERROR] Failed to obtain access token from 'gcloud auth print-access-token' and AUTH_TOKEN is not set." >&2
        echo "[ERROR] Please refresh your gcloud credentials or update AUTH_TOKEN in .env." >&2
        exit 1
    fi
}

# 5. Check AI Studio API Key
if [[ -z "${API_KEY:-}" ]]; then
    echo "[WARNING] API_KEY is not set in environment or .env. AI Studio benchmarks may fail."
else
    echo "[AUTH] Google AI Studio API_KEY is set (${API_KEY:0:10}...)"
fi

# Initial auth token retrieval
refresh_auth_token

# 6. Verify Dual Proxy Gateway is running
PROXY_URL="http://127.0.0.1:8000"
PROXY_STARTED_BY_SCRIPT=0

check_proxy() {
    python3 -c "import urllib.request; urllib.request.urlopen('${PROXY_URL}/docs', timeout=2)" &>/dev/null
}

if ! check_proxy; then
    uvicorn dual_proxy:app --host 127.0.0.1 --port 8000 --workers 2 --backlog 8192 --limit-concurrency 10000 > proxy.log 2>&1 &
    PROXY_PID=$!
    PROXY_STARTED_BY_SCRIPT=1

    # Wait for proxy to initialize
    echo -n "[PROXY] Waiting for proxy to be ready"
    for i in {1..10}; do
        if check_proxy; then
            echo " [OK]"
            break
        fi
        echo -n "."
        sleep 1
    done

    if ! check_proxy; then
        echo -e "\n[ERROR] Failed to start dual proxy. Check proxy.log for details." >&2
        exit 1
    fi
else
    echo "[PROXY] Dual proxy is already running at ${PROXY_URL}."
fi

# Cleanup proxy if it was started by this script
cleanup() {
    if [[ ${PROXY_STARTED_BY_SCRIPT} -eq 1 && -n "${PROXY_PID:-}" ]]; then
        echo -e "\n[CLEANUP] Stopping background dual proxy (PID: ${PROXY_PID})..."
        kill "${PROXY_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 7. Collect configurations from configs/
FILTER_ARG="${1:-all}"
if [[ "${FILTER_ARG}" == "text" ]]; then
    CONFIG_FILES=($(find configs -maxdepth 1 -name "config_*.yaml" ! -name "*image*" | sort))
elif [[ "${FILTER_ARG}" == "image" ]]; then
    CONFIG_FILES=($(find configs -maxdepth 1 -name "*image*.yaml" | sort))
elif [[ "${FILTER_ARG}" != "all" ]]; then
    CONFIG_FILES=($(find configs -maxdepth 1 -name "${FILTER_ARG}" | sort))
else
    CONFIG_FILES=($(find configs -maxdepth 1 -name "config_*.yaml" | sort))
fi

if [[ ${#CONFIG_FILES[@]} -eq 0 ]]; then
    echo "[ERROR] No configuration files found in configs/ matching 'config_*.yaml'." >&2
    exit 1
fi

echo -e "\n[CONFIGS] Found ${#CONFIG_FILES[@]} benchmark configuration(s) to execute:"
for cfg in "${CONFIG_FILES[@]}"; do
    echo "  - ${cfg}"
done
echo ""

# 8. Execute benchmarks sequentially (one by one to prevent host contention and OOM)
run_single_benchmark() {
    local config_file="$1"
    local config_name=$(basename "${config_file}")
    local log_file="/tmp/inference_perf_${config_name%.*}.log"
    
    local surface=""
    local target_path=""
    local token_or_key=""
    
    if [[ "${config_name}" == *"ai_studio"* ]]; then
        surface="Google AI Studio"
        target_path="${AI_STUDIO_OUTPUT_DIR}"
        token_or_key="${API_KEY:-}"
    elif [[ "${config_name}" == *"geap"* ]]; then
        surface="Gemini Enterprise Agent Platform (GEAP)"
        target_path="${GEAP_OUTPUT_DIR}"
        token_or_key="${AUTH_TOKEN}"
    else
        surface="Custom"
        target_path="${RUN_DIR}/custom"
        token_or_key="${API_KEY:-${AUTH_TOKEN:-}}"
    fi
    
    echo "----------------------------------------------------------------------"
    echo "[LAUNCH] Started ${config_name} (${surface})"
    echo "----------------------------------------------------------------------"
    if inference-perf \
        -c "${config_file}" \
        --storage.local_storage.path "${target_path}" \
        --server.api_key "${token_or_key}" > "${log_file}" 2>&1; then
        echo "[SUCCESS] Finished ${config_name}"
        return 0
    else
        local status=$?
        echo "[FAILED] Error in ${config_name} (exit code ${status}). See ${log_file}" >&2
        return ${status}
    fi
}

echo "======================================================================"
echo " [SEQUENTIAL] Running ${#CONFIG_FILES[@]} benchmark(s) sequentially..."
echo "======================================================================"

failed_count=0
current_idx=1
total_configs=${#CONFIG_FILES[@]}

for config_file in "${CONFIG_FILES[@]}"; do
    echo ""
    echo "[PROGRESS] (${current_idx}/${total_configs}) Executing $(basename "${config_file}")..."
    if ! run_single_benchmark "${config_file}"; then
        failed_count=$((failed_count + 1))
    fi
    current_idx=$((current_idx + 1))
done

if [[ ${failed_count} -gt 0 ]]; then
    echo -e "\n[WARNING] ${failed_count} benchmark(s) finished with errors. Inspect logs in ${RUN_DIR}/."
fi

echo "======================================================================"
echo "          🎉 ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo "Run Directory              : ${RUN_DIR}/"
echo "AI Studio metrics saved to : ${AI_STUDIO_OUTPUT_DIR}/"
echo "GEAP metrics saved to      : ${GEAP_OUTPUT_DIR}/"
echo "======================================================================"

# 9. Optionally synthesize unified comparison report if generate_report.py exists
if [[ -f "gemini_report_synthesis/generate_report.py" ]]; then
    echo "\n[REPORT] Synthesizing unified comparison report using Gemini..."
    python3 gemini_report_synthesis/generate_report.py \
        --run-dir "${RUN_DIR}" || {
            echo "[WARNING] Report synthesis encountered an issue. Metrics remain preserved in output directories."
        }
fi

echo "\nDone!"
