#!/bin/bash
# research_sweep.sh
# Automates the research comparison script across multiple depths
export OMP_NUM_THREADS=8

export NANOCHAT_BASE_DIR="out"
mkdir -p $NANOCHAT_BASE_DIR
set -euo pipefail

# Default to 8-way launch (for full 8xGPU nodes), but allow override via env
# and cap to locally visible GPUs when fewer are allocated.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ "${AVAILABLE_GPUS}" =~ ^[0-9]+$ ]] && [ "$AVAILABLE_GPUS" -gt 0 ] && [ "$NPROC_PER_NODE" -gt "$AVAILABLE_GPUS" ]; then
    echo "Requested NPROC_PER_NODE=${NPROC_PER_NODE}, but only ${AVAILABLE_GPUS} GPU(s) visible. Capping."
    NPROC_PER_NODE="${AVAILABLE_GPUS}"
fi
echo "Using NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NANOCHAT_SKIP_ENV_SETUP=${NANOCHAT_SKIP_ENV_SETUP:-0}"

echo "===== GPU Info ====="
nvidia-smi || true

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/research_sweep.sh [flags] <depth1> [depth2] ..."
    echo "Flags:"
    echo "  --fp8               Enable FP8 training"
    echo "  --no-compile        Disable torch.compile"
    echo "  --use-onecycle 0|1  Enable OneCycleLR scheduler (default: 1)"
    echo "  --target-tokens N   Explicit token budget per run (default: auto-estimated)"
    echo "  --max-shards N      Step through up to N data shards"
    echo "  --models M1,M2,...  Comma-separated list of models to run (e.g. 'base,remixed-linear')"
    echo "  --device-batch-size N  Override per-device batch size"
    echo "  --total-batch-size N   Override total batch size"
    echo "  --log-every N       Logging frequency"
    echo "  --eval-every N      Evaluation frequency (-1 = at end)"
    echo "  --save-every N      Checkpoint frequency"
    echo "  --core-metric-every N Core metric frequency"
    echo "  --skip-core         Completely disable CORE metric evaluation"
    echo "  --no-mu-p           Disable mu-P LR scaling (default: True for research)"
    echo ""
    echo "Example: ./scripts/research_sweep.sh --models base,remixed-linear 12"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_OUT_DIR="out/research_sweep_${TIMESTAMP}"
ROOT_OUT_DIR_OVERRIDE=""

EXTRA_ARGS=()
MAX_SHARDS=170

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fp8)
            EXTRA_ARGS+=("--fp8")
            shift
            ;;
        --no-compile)
            EXTRA_ARGS+=("--no-compile")
            shift
            ;;
        --skip-core)
            EXTRA_ARGS+=("--skip-core")
            shift
            ;;
        --no-mu-p)
            EXTRA_ARGS+=("--no-mu-p")
            shift
            ;;
        --models)
            EXTRA_ARGS+=("--models" "$2")
            shift 2
            ;;
        --device-batch-size)
            EXTRA_ARGS+=("--device-batch-size" "$2")
            shift 2
            ;;
        --total-batch-size)
            EXTRA_ARGS+=("--total-batch-size" "$2")
            shift 2
            ;;
        --log-every)
            EXTRA_ARGS+=("--log-every" "$2")
            shift 2
            ;;
        --eval-every)
            EXTRA_ARGS+=("--eval-every" "$2")
            shift 2
            ;;
        --save-every)
            EXTRA_ARGS+=("--save-every" "$2")
            shift 2
            ;;
        --core-metric-every)
            EXTRA_ARGS+=("--core-metric-every" "$2")
            shift 2
            ;;
        --max-shards)
            MAX_SHARDS=$2
            EXTRA_ARGS+=("--max-shards" "$2")
            shift 2
            ;;
        --target-tokens)
            EXTRA_ARGS+=("--target-tokens" "$2")
            shift 2
            ;;
        --target-param-data-ratio)
            EXTRA_ARGS+=("--target-param-data-ratio" "$2")
            shift 2
            ;;
        --sequence-len)
            EXTRA_ARGS+=("--sequence-len" "$2")
            shift 2
            ;;
        --mu-p-mode)
            EXTRA_ARGS+=("--mu-p-mode" "$2")
            shift 2
            ;;
        --router-context-window)
            EXTRA_ARGS+=("--router-context-window" "$2")
            shift 2
            ;;
        --remix-use-basis-gate)
            EXTRA_ARGS+=("--remix-use-basis-gate" "$2")
            shift 2
            ;;
        --remix-use-output-gate)
            EXTRA_ARGS+=("--remix-use-output-gate" "$2")
            shift 2
            ;;
        --remix-use-context)
            EXTRA_ARGS+=("--remix-use-context" "$2")
            shift 2
            ;;
        --remix-basis-gate-mode)
            EXTRA_ARGS+=("--remix-basis-gate-mode" "$2")
            shift 2
            ;;
        --remix-shared-context-gates)
            EXTRA_ARGS+=("--remix-shared-context-gates" "$2")
            shift 2
            ;;
        --remix-use-dual-gate)
            EXTRA_ARGS+=("--remix-use-dual-gate" "$2")
            shift 2
            ;;
        --remix-basis-scale-factor)
            EXTRA_ARGS+=("--remix-basis-scale-factor" "$2")
            shift 2
            ;;
        --remix-output-gate-rank)
            EXTRA_ARGS+=("--remix-output-gate-rank" "$2")
            shift 2
            ;;
        --research-dim)
            EXTRA_ARGS+=("--research-dim" "$2")
            shift 2
            ;;
        --remix-basis-size)
            EXTRA_ARGS+=("--remix-basis-size" "$2")
            shift 2
            ;;
        --cclblock-modulation)
            EXTRA_ARGS+=("--cclblock-modulation" "$2")
            shift 2
            ;;
        --cclblock-orth-lambda)
            EXTRA_ARGS+=("--cclblock-orth-lambda" "$2")
            shift 2
            ;;
        --cclblock-context-stream)
            EXTRA_ARGS+=("--cclblock-context-stream" "$2")
            shift 2
            ;;
        --cclblock-ema-factor)
            EXTRA_ARGS+=("--cclblock-ema-factor" "$2")
            shift 2
            ;;
        --cclblock-stale-ctx-lag)
            EXTRA_ARGS+=("--cclblock-stale-ctx-lag" "$2")
            shift 2
            ;;
        --cclblock-sparse-gate-k)
            EXTRA_ARGS+=("--cclblock-sparse-gate-k" "$2")
            shift 2
            ;;
        --cclblock-gate-temperature)
            EXTRA_ARGS+=("--cclblock-gate-temperature" "$2")
            shift 2
            ;;
        --cclblock-context-bank-size)
            EXTRA_ARGS+=("--cclblock-context-bank-size" "$2")
            shift 2
            ;;
        --cclblock-per-head-ctx)
            EXTRA_ARGS+=("--cclblock-per-head-ctx" "$2")
            shift 2
            ;;
        --cclblock-context-source)
            EXTRA_ARGS+=("--cclblock-context-source" "$2")
            shift 2
            ;;
        --cclblock-chunk-size)
            EXTRA_ARGS+=("--cclblock-chunk-size" "$2")
            shift 2
            ;;
        --cclblock-aux-objective)
            EXTRA_ARGS+=("--cclblock-aux-objective" "$2")
            shift 2
            ;;
        --cclblock-aux-lambda)
            EXTRA_ARGS+=("--cclblock-aux-lambda" "$2")
            shift 2
            ;;
        --cclblock-boundary-token-id)
            EXTRA_ARGS+=("--cclblock-boundary-token-id" "$2")
            shift 2
            ;;
        --use-ral)
            EXTRA_ARGS+=("--use-ral" "$2")
            shift 2
            ;;
        --ral-rank)
            EXTRA_ARGS+=("--ral-rank" "$2")
            shift 2
            ;;
        --cclblock-film-gate)
            EXTRA_ARGS+=("--cclblock-film-gate" "$2")
            shift 2
            ;;
        --cclblock-attn-shadow-dim)
            EXTRA_ARGS+=("--cclblock-attn-shadow-dim" "$2")
            shift 2
            ;;
        --cclblock-dynamic-ratio)
            EXTRA_ARGS+=("--cclblock-dynamic-ratio" "$2")
            shift 2
            ;;
        --cclblock-gate-rank)
            EXTRA_ARGS+=("--cclblock-gate-rank" "$2")
            shift 2
            ;;
        --cclblock-num-regimes)
            EXTRA_ARGS+=("--cclblock-num-regimes" "$2")
            shift 2
            ;;
        --cclblock-regime-temperature)
            EXTRA_ARGS+=("--cclblock-regime-temperature" "$2")
            shift 2
            ;;
        --model-dim)
            EXTRA_ARGS+=("--model-dim" "$2")
            shift 2
            ;;
        --aspect-ratio)
            EXTRA_ARGS+=("--aspect-ratio" "$2")
            shift 2
            ;;
        --p20-mone-narrow|--p20-mone-frozen)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --p22-n-templates|--p22-template-routing-learned|--p22-template-topk|--p22-attn-moe-route|--remix-gate-lr-scale)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --p23-tiny-expert|--p23-n-experts|--p23-topk|--p23-learned-route|--p23-std-moe-experts|--p23-std-moe-topk|--p23-std-moe-aux-weight|--p23-lokr|--p23-lokr-rank|--p23-use-shared-block-router|--p23-linear-moe-experts|--p23-linear-moe-topk|--p24-use-sliced-weight|--p24-sliced-weight-reduction-scale|--p24-sliced-weight-min-select|--p24-sliced-weight-scope|--p24-sliced-weight-balance-coeff|--p24-quantile-route|--p24-use-folded-mod|--p24-folded-mod-reduction-scale|--p24-folded-mod-min-dim|--p24-folded-mod-scope|--p24-folded-mod-gate-act|--p24-use-sequence-gated-linear|--p24-sequence-gated-scope|--p24-sequence-gated-act)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --p26-output-gated-linear|--remix-basis-gate-rank|--remix-basis-gate-mode)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --p28-shared-basis|--p28-chunk-routing-size|--p28-global-template-bank|--p28-attn-proj-templates|--p28-attn-qk-templates|--target-active-params)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;

        --warmup-ratio|--warmdown-ratio|--final-lr-frac)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --cclblock-poly-order|--cclblock-lie-generators|--cclblock-grassmann-bank-size|--cclblock-tucker-rank|--cclblock-tucker-modes|--cclblock-svs-rank|--cclblock-svs-eps|--cclblock-vq-codes|--cclblock-vq-temperature|--cclblock-dcu-warmup-steps|--cclblock-fsi-rotations|--cclblock-fsi-selector-dim|--cclblock-aesp-strata|--cclblock-aesp-delta-rank|--cclblock-ckr-branches|--cclblock-ckr-kernel-size|--cclblock-ckr-pos-channels|--cclblock-ckr-dual-optim|--cclblock-ckr-content-bias|--cclblock-giad-rank|--cclblock-psg-kernel-size|--cclblock-ss-dynamic-ratio|--cclblock-ss-branches|--cclblock-ss-kernel-size|--cclblock-lokr-branches|--cclblock-lokr-rank|--cclblock-ckr-temp-start|--cclblock-ckr-temp-end|--cclblock-com-kernel-size|--cclblock-ckr-ortho-init|--cclblock-ckr-branch-dropout|--cclblock-ckr-diversity-lambda|--cclblock-pgr-kernel-size|--cclblock-cil-kernel-size|--cclblock-prb-kernel-size|--modulation-diagnostics|--p18-layer-drop|--p18-dynamic-activation|--p18-mixture-norm|--p18-aux-sim-lambda|--p18-gradient-penalty|--p18-per-channel-scale|--p19-residual-gate|--p19-head-importance|--p19-residual-mix-groups|--p19-attn-logit-bias|--p19-residual-decay|--p19-grad-equilibrium|--p19-spectral-reparam|--p19-weight-anticollapse|--p19-ve-bias|--p19-weight-noise|--p20-hrcs-scale|--p20-lswr-scale|--p20-lswr-planes|--p20-lrcfb-branches|--p20-lrcfb-narrow|--p20-lrcfb-learned|--p20-lrcfb-topk|--p20-dgcr-branches|--p20-dgcr-aux-weight|--p20-mone-experts|--p20-mone-topk|--p20-ncea-branches|--p20-ncea-eps|--p20-adwi|--p20-pwu-branches|--p20-pwu-phase|--p20-fsvd-gate|--p20-wbfc-clusters|--p20-wbfc-active|--p21-per-experts|--p21-per-topk|--p21-per-learned|--p21-per-attn|--p22-attn-moe-route|--p23-tiny-expert|--p23-n-experts|--p23-topk|--p23-learned-route|--p23-std-moe-experts|--p23-std-moe-topk|--p23-std-moe-aux-weight|--p23-lokr|--p23-lokr-rank|--p23-use-shared-block-router|--p23-linear-moe-experts|--p23-linear-moe-topk|--p23-quantile-route|--p24-use-sliced-weight|--p24-sliced-weight-reduction-scale|--p24-sliced-weight-min-select|--p24-sliced-weight-scope|--p24-sliced-weight-balance-coeff|--p24-quantile-route|--p24-use-folded-mod|--p24-folded-mod-reduction-scale|--p24-folded-mod-scope|--p24-folded-mod-gate-act|--p24-use-sequence-gated-linear|--p24-sequence-gated-scope|--p24-sequence-gated-act)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --use-onecycle)
            EXTRA_ARGS+=("--use-onecycle" "$2")
            shift 2
            ;;
        --use-chunked-remix)
            EXTRA_ARGS+=("--use-chunked-remix" "$2")
            shift 2
            ;;
        --tokenizer-dir)
            TOKENIZER_DIR_FLAG="$2"
            EXTRA_ARGS+=("--tokenizer-dir" "$2")
            shift 2
            ;;
        --data-dir)
            DATA_DIR_FLAG="$2"
            EXTRA_ARGS+=("--data-dir" "$2")
            shift 2
            ;;
        --out-dir)
            ROOT_OUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --remix-disable-ln-basis)
            EXTRA_ARGS+=("--remix-disable-ln-basis" "$2")
            shift 2
            ;;
        --dense-intermediate-ln)
            EXTRA_ARGS+=("--dense-intermediate-ln" "$2")
            shift 2
            ;;
        --use-mst|--mst-n-subs|--mst-sub-dim|--mst-head-dim|--mst-input-mode|--mst-rotated-slice-learned|\
        --mst-routing-mode|--mst-routing-topk|--mst-routing-aux-weight|--mst-diversity-weight|\
        --mst-ffn-mode|--mst-transition-mode|--mst-final-mode|--mst-final-topk|\
        --mst-ffn-shared-up|--mst-ffn-inner-dim|--mst-sub-dropout|\
        --mst-transition-every|--mst-ffa-temperature|--mst-global-residual|\
        --mst-hybrid-dense|--mst-cross-sub-kv|\
        --mst-sub-aux-weight|--mst-progressive-merge|--mst-multi-scale-windows|\
        --mst-delta-residual|--mst-sub-layers|\
        --mst-grad-equalize|--mst-block-diagonal-muon|--mst-transition-width-mult|\
        --mst-sub-lr-scale|--mst-shared-expert|--mst-router-entropy-weight|\
        --mst-shared-kv-attn|--mst-contrastive-diversity-weight|\
        --mst-transition-nonlinear|--mst-transition-gated|--mst-transition-mlp|\
        --mst-cross-sub-gate|--mst-hyper-connect|--mst-cross-kv-inject|\
        --mst-slice-transition|--mst-lookback-layers|--mst-bilinear-transition|\
        --mst-cross-sub-qmod|--mst-feature-cycle|--mst-mean-transition|\
        --mst-global-residual)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --use-eet|--eet-frozen-kv|--eet-router-type|--eet-router-hidden|\
        --eet-freq-prior-alpha|--eet-pos-prior-beta|--eet-domain-prior|\
        --eet-warmup-frac|--eet-explore-frac|--eet-reconstruct-lambda|\
        --eet-efficiency-lambda-start|--eet-efficiency-lambda-end|\
        --eet-translator-rank|--eet-max-frozen-kv-frac|\
        --eet-exit-threshold|--eet-min-exit-layer|\
        --eet-loss-variant|--eet-topk-vocab|--eet-entropy-lambda|\
        --eet-surprise-lambda|--eet-adv-lambda|--eet-adv-entropy-lambda|\
        --eet-quality-lambda|--eet-quality-entropy-bonus|\
        --eet-gumbel-temp-start|--eet-gumbel-temp-end|--eet-gumbel-hard|--eet-commitment-beta|--eet-global-router|\
        --eet-freq-efficiency-alpha|--eet-diversity-lambda|--eet-ce-guided-lambda|--eet-depth-weight-type|--eet-depth-weight-max|\
        --eet-use-override|--eet-override-prob-start|--eet-override-prob-end|--eet-reenter-final|--eet-compute-skip|--eet-target-active-frac|--eet-capacity-schedule|--eet-exit-fracs|--eet-capacity-alignment-lambda|--eet-router-task-grad|--eet-reinforce-interval|--eet-reinforce-lambda|--eet-exit-adapter-rank|--eet-router-after-block|--eet-ffn-skip|--eet-ffn-target-frac|--eet-ffn-full-attn|--eet-router-lr-mult|--eet-model-lr-mult|--eet-depth-affine|--eet-capacity-anneal-frac|--eet-learned-schedule|--eet-departure-summary|--eet-route-consistency-lambda|--eet-dense-distill-interval|--eet-dense-distill-lambda|--eet-depth-lr-scale|--eet-depth-grad-scale|--eet-detach-aux-from-backbone|--eet-detach-exit-from-backbone)
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            # Stop parsing flags when we hit a depth
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                break
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

# Check for depths again after potential shift
if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/research_sweep.sh [flags] <depth1> [depth2] ..."
    exit 1
fi

# If caller passed --out-dir, use that stable directory instead of the timestamped default.
# This is what allows sweep scripts (p29_sweep.sh etc.) to resume across re-runs.
if [[ -n "$ROOT_OUT_DIR_OVERRIDE" ]]; then
    ROOT_OUT_DIR="$ROOT_OUT_DIR_OVERRIDE"
fi

echo "Starting Research Sweep. Output directory: ${ROOT_OUT_DIR}"
mkdir -p "${ROOT_OUT_DIR}"

# ── 1. Install / sync environment ─────────────────────────────────────────────
# UV_SYSTEM_PYTHON=1  → use the calling shell's Python / site-packages as-is.
# NANOCHAT_SKIP_ENV_SETUP=1 → skip all uv setup (venv must already exist).

if [[ "${UV_SYSTEM_PYTHON:-0}" == "1" ]]; then
    echo "UV_SYSTEM_PYTHON=1: using system Python, skipping venv creation."
    echo "Validating system environment dependencies via uv..."
    uv pip install -e .[gpu] --system
elif [[ "${NANOCHAT_SKIP_ENV_SETUP:-0}" == "1" ]]; then
    echo "Skipping environment setup (NANOCHAT_SKIP_ENV_SETUP=1)."
    if [[ ! -d ".venv" ]]; then
        echo "Error: .venv not found. Unset NANOCHAT_SKIP_ENV_SETUP or create .venv first."
        exit 1
    fi
    source .venv/bin/activate
else
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate
fi
# ── 4. Resolve torchrun (prefers venv, falls back to system) ──────────────────
if command -v torchrun &> /dev/null; then
    RUNNER="torchrun --standalone --nproc_per_node=${NPROC_PER_NODE}"
else
    echo "Warning: torchrun not found, falling back to python -m torch.distributed.run"
    RUNNER="python -m torch.distributed.run --standalone --nproc_per_node=${NPROC_PER_NODE}"
fi

python -m nanochat.report reset

# ── 5. One-time data + tokenizer setup ──────────────────────────────────────────
# Resolve directories for existence checks
CHECK_DATA_DIR="${DATA_DIR_FLAG:-$NANOCHAT_BASE_DIR/base_data_climbmix}"
DATA_OPT=""
[ -n "${DATA_DIR_FLAG:-}" ] && DATA_OPT="--data-dir $DATA_DIR_FLAG"

TOK_OPT=""
[ -n "${TOKENIZER_DIR_FLAG:-}" ] && TOK_OPT="--tokenizer-dir $TOKENIZER_DIR_FLAG"
TOKENIZER_CHECK_DIR="${TOKENIZER_DIR_FLAG:-$NANOCHAT_BASE_DIR/tokenizer}"

# 1. Dataset existence check
# We check for the last requested shard to avoid redundant "skipping" logs
LAST_SHARD_IDX=$((MAX_SHARDS - 1))
LAST_SHARD_FILE=$(printf "shard_%05d.parquet" $LAST_SHARD_IDX)
VAL_SHARD_FILE="shard_06542.parquet"

DATASET_DOWNLOAD_PID=""
if [ -f "$CHECK_DATA_DIR/$LAST_SHARD_FILE" ] && [ -f "$CHECK_DATA_DIR/$VAL_SHARD_FILE" ]; then
    echo "Dataset (up to $MAX_SHARDS shards) already exists in $CHECK_DATA_DIR, skipping download."
else
    echo "Dataset incomplete. Downloading $MAX_SHARDS shards to $CHECK_DATA_DIR..."
    mkdir -p "$CHECK_DATA_DIR"
    # Download first 8 shards synchronously so we have something to start with
    python -m nanochat.dataset -n 8 $DATA_OPT
    # Download the rest in background, quieted to a log file
    python -m nanochat.dataset -n $MAX_SHARDS $DATA_OPT > "${ROOT_OUT_DIR}/dataset_download.log" 2>&1 &
    DATASET_DOWNLOAD_PID=$!
fi

# 2. Tokenizer existence check
if [ ! -f "$TOKENIZER_CHECK_DIR/tokenizer.pkl" ]; then
    echo "Tokenizer not found in $TOKENIZER_CHECK_DIR, training..."
    python -m scripts.tok_train $TOK_OPT $DATA_OPT
    python -m scripts.tok_eval $TOK_OPT $DATA_OPT
else
    echo "Tokenizer already exists in $TOKENIZER_CHECK_DIR, skipping training."
fi

if [ -n "$DATASET_DOWNLOAD_PID" ]; then
    echo "Waiting for background dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi


for DEPTH in "$@"; do
    echo "================================================================"
    echo "Processing Depth: ${DEPTH}"
    echo "================================================================"
    
    RUN_DIR="${ROOT_OUT_DIR}/depth_${DEPTH}"
    mkdir -p "${RUN_DIR}"

    # research_compare orchestrates multiple training subprocesses itself.
    # Running it under torchrun causes nested distributed launches and rank/env collisions.
    python -u -m scripts.research_compare --depth "${DEPTH}" --run-dir "${RUN_DIR}" "${EXTRA_ARGS[@]}"
    
    if [ $? -ne 0 ]; then
        echo "Error: Sweep failed for depth ${DEPTH}. Check logs in ${RUN_DIR}."
        exit 1
    fi
done

python -m nanochat.report generate
echo "================================================================"
echo "Sweep Complete! Results saved to ${ROOT_OUT_DIR}"
echo "================================================================"
