#!/usr/bin/env bash
# ============================================================================
# EET P01 Sweep — Early Exit Transformer Foundational Experiments
# ============================================================================
# Phase 1 of the EET research programme. Establishes:
#   - Dense baseline control (no early exit)
#   - Option A (masked attention) vs Option B (frozen KV injection)
#   - Frequency and POS prior ablations
#   - Router architecture comparison (linear, mlp1, mlp2)
#   - Full pipeline with three-phase training
#
# Token budget: uses standard Chinchilla-ratio (10.5×) calculated from the
# base dense GPT at the specified depth — same as MST sweeps.
#
# Usage:
#   bash scripts/eet_p01_sweep.sh [--force] [DEPTH]
#   bash scripts/eet_p01_sweep.sh 8
#   bash scripts/eet_p01_sweep.sh --force 12
#
# Default depth: 8
# ============================================================================

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────────────────
FORCE=0
DEPTH=8
for arg in "$@"; do
    case $arg in
        --force) FORCE=1 ;;
        *)       DEPTH=$arg ;;
    esac
done

# ── Output directory ─────────────────────────────────────────────────────────
EET_OUT_BASE="${EET_OUT_BASE:-out/eet_p01}"
LOGFILE="${SWEEP_LOG:-${EET_OUT_BASE}/sweep_eet_p01_d${DEPTH}.log}"
STATE_FILE="${EET_OUT_BASE}/sweep_state_d${DEPTH}.json"
mkdir -p "$EET_OUT_BASE"

echo "═══════════════════════════════════════════════════════════════"
echo "  EET P01: Early Exit Transformer — Foundational Experiments"
echo "  Depth:       ${DEPTH}"
echo "  Output:      ${EET_OUT_BASE}"
echo "  State:       ${STATE_FILE}"
echo "  Log:         ${LOGFILE}"
echo "═══════════════════════════════════════════════════════════════"

# ── State management (JSON) ──────────────────────────────────────────────────
init_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo '{"completed":{},"started":{}}' > "$STATE_FILE"
    fi
}

check_completed() {
    local tag="$1"
    [ "$FORCE" -eq 1 ] && return 1
    python3 -c "
import json, sys
state = json.load(open('$STATE_FILE'))
sys.exit(0 if '$tag' in state.get('completed', {}) else 1)
" 2>/dev/null
}

mark_started() {
    local tag="$1"
    local run_dir="$2"
    python3 -c "
import json, datetime
state = json.load(open('$STATE_FILE'))
state.setdefault('started', {})['$tag'] = {
    'run_dir': '$run_dir',
    'started_at': datetime.datetime.now().isoformat()
}
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

mark_completed() {
    local tag="$1"
    python3 -c "
import json, datetime
state = json.load(open('$STATE_FILE'))
state.setdefault('completed', {})['$tag'] = {
    'completed_at': datetime.datetime.now().isoformat()
}
state.get('started', {}).pop('$tag', None)
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
"
}

print_header() {
    local id="$1" tag="$2" desc="$3"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$id] $tag"
    echo "  $desc"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

init_state

# ── Common flags ─────────────────────────────────────────────────────────────
# Uses the standard base dense model's Chinchilla-ratio token count.
# --target-param-data-ratio 10.5 with --target-tokens 0 makes base_train.py
# compute num_iterations from the base GPT param count at this depth.
EET_COMMON="--models base \
  --device-batch-size ${DEVICE_BATCH_SIZE:-128} --total-batch-size -1 \
  --use-onecycle 0 --log-every 20 --skip-core \
  --data-dir ${DATA_DIR:-data} --tokenizer-dir ${TOKENIZER_DIR:-tokenizer} \
  --sequence-len 2048 \
  --target-param-data-ratio 10.5 \
  --warmup-ratio 0.005 \
  --warmdown-ratio 0.65 \
  --final-lr-frac 0.05 \
  --research-dim -1 \
  --target-tokens -1 \
  --target-active-params 0 \
  --save-every 200 \
  --eval-every -1"

# Add optional env-based flags
[ -n "${MAX_SHARDS:-}" ]    && EET_COMMON="$EET_COMMON --max-shards $MAX_SHARDS"
[ "${USE_FP8:-0}" = "1" ]   && EET_COMMON="$EET_COMMON --fp8"

run_experiment() {
    local tag="$1"
    shift
    local desc="$1"
    shift

    if check_completed "$tag"; then
        echo "⏭  Skipping $tag (already completed)"
        return 0
    fi

    print_header "$tag" "$tag" "$desc"
    local run_dir="${EET_OUT_BASE}/${tag}"

    # --force: clean old run directory to prevent checkpoint resumption
    if [ "$FORCE" -eq 1 ] && [ -d "$run_dir" ]; then
        echo "🗑  --force: removing old run directory: $run_dir"
        rm -rf "$run_dir"
    fi

    mark_started "$tag" "$run_dir"

    if bash scripts/research_sweep.sh $EET_COMMON \
      --out-dir "$run_dir" \
      "$@" \
      $DEPTH 2>&1 | tee -a "$LOGFILE"; then
        echo "✅  $tag done"
        mark_completed "$tag"
    else
        echo "❌  $tag FAILED — will retry next run"
    fi
}

# ============================================================================
# EET Phase 1 Experiments
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Depth: ${DEPTH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# EET_P1_0: Dense baseline (control — standard GPT, no early exit)
#run_experiment "EET_P1_0_DENSE_D${DEPTH}" \
#    "Dense baseline (standard GPT, no early exit — control)" \
#    --use-eet 0
#
# EET_P1_1: Masked Attention (Option A — exited tokens masked out of attention)
#run_experiment "EET_P1_1_MASKED_D${DEPTH}" \
#    "Option A: masked attention for exited tokens (mlp2 router)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2

# EET_P1_2: Frozen KV Injection (Option B — frozen K/V remain available)
#run_experiment "EET_P1_2_FROZEN_KV_D${DEPTH}" \
#    "Option B: frozen KV injection for exited tokens (mlp2 router)" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp2

# EET_P1_12: Entropy + Surprise Loss (runs with hard routing in Phase 2)
#run_experiment "EET_P1_12_ENTROPY_SURPRISE_D${DEPTH}" \
#    "Entropy + surprise loss variant (running with hard routing)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.00 --eet-explore-frac 0.5 \
#    --eet-loss-variant entropy_surprise \
#    --eet-commitment-beta 0.1 \
#    --eet-global-router 1 \
#    --eet-entropy-lambda 0.3 --eet-surprise-lambda 0.1 \
#    --eet-freq-efficiency-alpha 2.0 --eet-diversity-lambda 0.1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1

# EET_P1_19: CE-Guided Routing Loss (compile-friendly, zero memory overhead)
#run_experiment "EET_P1_19_CE_GUIDED_D${DEPTH}" \
#    "CE-guided routing loss variant (running with soft/hard routing, zero memory)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 

# EET_P1_21: CE-Guided + Loss scaling by exit depth (Option 1: Linear, max=2.5)
#run_experiment "EET_P1_21_CEG_LINEAR_D${DEPTH}" \
#    "CE-guided routing with linear loss scaling by exit depth (max=2.5)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 --eet-min-exit-layer 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type linear --eet-depth-weight-max 2.5
#
# EET_P1_22: CE-Guided + Loss scaling by exit depth (Option 2: EMA Inverse Freq) - CLEAN BASELINE
#run_experiment "EET_P1_22_CEG_EMA_D${DEPTH}" \
#    "CE-guided routing with EMA inverse frequency loss scaling by exit depth" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0
#
# EET_P1_22_AFFINE: Cheap Exit-Depth Alignment
#run_experiment "EET_P1_22_CEG_EMA_AFFINE_D${DEPTH}" \
#    "EET_P1_22 Baseline + Cheap Exit-Depth Alignment (depth-affine=1)" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0 \
#    --eet-depth-affine 1

# EET_P1_22_ANNEAL: Target Capacity Annealing
#run_experiment "EET_P1_22_CEG_EMA_ANNEAL_D${DEPTH}" \
#    "EET_P1_22 Baseline + Target Capacity Annealing (anneal-frac=0.15)" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0 \
#    --eet-capacity-anneal-frac 0.15

# EET_P1_22_SCHED: Learned prior scheduling logits
#run_experiment "EET_P1_22_CEG_EMA_SCHED_D${DEPTH}" \
#    "EET_P1_22 Baseline + Learned Prior Exit Scheduling Logits" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0 \
#    --eet-learned-schedule 1

# EET_P1_22_DEPART: Departure Summary state injection
#run_experiment "EET_P1_22_CEG_EMA_DEPART_D${DEPTH}" \
#    "EET_P1_22 Baseline + Departure Summary Context Injection" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0 \
#    --eet-departure-summary 1
#
# EET_P1_22_CONSIST: EMA Route Consistency Loss
#run_experiment "EET_P1_22_CEG_EMA_CONSIST_D${DEPTH}" \
#    "EET_P1_22 Baseline + EMA Route Consistency Loss (lambda=0.1)" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0 \
#    --eet-route-consistency-lambda 0.1
#
# EET_P1_22_DISTILL: Concurrent Dense Distillation
#run_experiment "EET_P1_22_CEG_EMA_DISTILL_D${DEPTH}" \
#    "EET_P1_22 Baseline + Concurrent Dense Distillation (interval=2, lambda=0.5)" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0 \
#    --eet-dense-distill-interval 2 --eet-dense-distill-lambda 0.5

# EET_P1_22_ALL: All 6 features combined (Kitchen Sink)
#run_experiment "EET_P1_22_CEG_EMA_ALL_D${DEPTH}" \
#    "EET_P1_22 Baseline + All 6 Novel EET Improvements Combined" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
#    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
#    --eet-global-router 1 --eet-router-task-grad 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
#    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
#    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
#    --eet-capacity-alignment-lambda 1.0 \
#    --eet-depth-affine 1 \
#    --eet-capacity-anneal-frac 0.15 \
#    --eet-learned-schedule 1 \
#    --eet-departure-summary 1 \
#    --eet-route-consistency-lambda 0.1 \
#    --eet-dense-distill-interval 2 --eet-dense-distill-lambda 0.5
# EET_P1_22: CE-Guided + Loss scaling by exit depth (Option 2: EMA Inverse Freq)
#run_experiment "EET_P1_22_CEG_EMA_REENTRY_D${DEPTH}" \
#    "CE-guided routing with EMA inverse frequency loss and renetry scaling by exit depth" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 1 \
#    --eet-router-type mlp1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 --eet-exit-fracs "0.15, 0.25, 0.20, 0.15, 0.15, 0.10" \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10

# EET_P1_23: CE-Guided + Loss scaling by exit depth (Option 3: Square Root)
#run_experiment "EET_P1_23_CEG_SQRT_D${DEPTH}" \
#    "CE-guided routing with square root loss scaling by exit depth" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type sqrt

# EET_P1_24: CE-Guided + Stochastic Depth Override (Linear + Override start=0.5, end=0.1)
#run_experiment "EET_P1_24_CEG_OVERRIDE_D${DEPTH}" \
#    "CE-guided routing with linear loss scaling and stochastic depth override" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type linear --eet-depth-weight-max 2.5 \
#    --eet-use-override 1 --eet-override-prob-start 0.5 --eet-override-prob-end 0.1


#run_experiment "EET_P1_19_CE_GUIDED_D${DEPTH}" \
#run_experiment "EET_P1_19_CE_GUIDED_D${DEPTH}" \
#    "CE-guided routing loss variant (running with soft/hard routing, zero memory)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-commitment-beta 0.1 \
#    --eet-quality-entropy-bonus 0.1 \
#    --eet-global-router 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-freq-efficiency-alpha 2.0 --eet-diversity-lambda 0.1 \
#    --eet-gumbel-temp-start 2.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1

# EET_P1_10: Variant A — REINFORCE Quality Loss + Entropy Bonus
#run_experiment "EET_P1_10_VARIANT_A_D${DEPTH}" \
#    "Variant A: REINFORCE quality loss + entropy bonus (per-token exit differentiation)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 1.0 \
#    --eet-loss-variant quality \
#    --eet-quality-lambda 1.0 \
#    --eet-commitment-beta 0.1 \
#    --eet-quality-entropy-bonus 0.1 \
#    --eet-global-router 1 \
#    --eet-freq-efficiency-alpha 2.0 --eet-diversity-lambda 0.1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1

# EET_P1_13: Gumbel-Softmax routing with temperature annealing and commitment loss
#run_experiment "EET_P1_13_GUMBEL_D_HARD${DEPTH}" \
#    "Gumbel-Softmax routing with annealing (5.0->0.1) and hard routing and commitment loss (beta=0.1)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.50 \
#    --eet-loss-variant quality \
#    --eet-quality-lambda 1.0 \
#    --eet-quality-entropy-bonus 0.1 \
#    --eet-gumbel-temp-start 5.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-commitment-beta 0.1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1
#
# EET_P1_15: Gumbel-Softmax routing with temperature annealing and soft routing and commitment loss
#run_experiment "EET_P1_13_GUMBEL_D_SOFT${DEPTH}" \
#    "Gumbel-Softmax routing with annealing (5.0->0.1) and soft routing and commitment loss (beta=0.1)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.50 \
#    --eet-loss-variant quality \
#    --eet-quality-lambda 1.0 \
#    --eet-quality-entropy-bonus 0.1 \
#    --eet-gumbel-temp-start 5.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 0 \
#    --eet-commitment-beta 0.1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1
#
#
#
# EET_P1_15: Upfront Global Exit Router with Gumbel-Softmax + Quality Loss + Commitment Loss
#run_experiment "EET_P1_15_GLOBAL_ROUTER_GUMBEL_D${DEPTH}" \
#    "Upfront Global Exit Router with Gumbel-Softmax (5.0->0.1) and commitment loss (beta=0.1)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.50 \
#    --eet-loss-variant quality \
#    --eet-quality-lambda 1.0 \
#    --eet-quality-entropy-bonus 0.1 \
#    --eet-gumbel-temp-start 5.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-commitment-beta 0.1 \
#    --eet-global-router 1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1

# EET_P1_14: Per-Token Layer-Weighted Loss with commitment loss
#run_experiment "EET_P1_14_LAYER_WEIGHTED_D${DEPTH}" \
#    "Per-Token Layer-Weighted Loss with commitment loss (beta=0.1, co-adaptive training)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.50 \
#    --eet-loss-variant layer_weighted \
#    --eet-commitment-beta 0.1 \
#    --eet-quality-entropy-bonus 0.1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1
#
# EET_P1_16: Upfront Global Exit Router with Per-Token Layer-Weighted Loss + Commitment Loss
#run_experiment "EET_P1_16_GLOBAL_ROUTER_LW_D${DEPTH}" \
#    "Upfront Global Exit Router with Per-Token Layer-Weighted Loss and commitment loss (beta=0.1)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.50 \
#    --eet-loss-variant layer_weighted \
#    --eet-commitment-beta 0.1 \
#    --eet-global-router 1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1
#

# EET_P1_17: Gumbel-Softmax + Freq-Scaled Efficiency + Diversity Pressure
# Tests whether per-token frequency-weighted efficiency and exit diversity
# breaks the uniform exit depth equilibrium that plagued all previous variants.
#run_experiment "EET_P1_17_GUMBEL_FREQDIV_D${DEPTH}" \
#    "Gumbel-Softmax with freq-scaled efficiency (α=2.0) + diversity pressure (λ=0.1)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.50 \
#    --eet-loss-variant quality \
#    --eet-quality-lambda 1.0 \
#    --eet-quality-entropy-bonus 0.1 \
#    --eet-gumbel-temp-start 5.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-commitment-beta 0.1 \
#    --eet-freq-efficiency-alpha 2.0 --eet-diversity-lambda 0.1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1
#
# EET_P1_18: Layer-Weighted + Freq-Scaled Efficiency + Diversity Pressure
#run_experiment "EET_P1_18_LW_FREQDIV_D${DEPTH}" \
#    "Layer-Weighted with freq-scaled efficiency (α=2.0) + diversity pressure (λ=0.1)" \
#    --use-eet 1 --eet-frozen-kv 0 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.50 \
#    --eet-loss-variant layer_weighted \
#    --eet-commitment-beta 0.1 \
#    --eet-freq-efficiency-alpha 2.0 --eet-diversity-lambda 0.1 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1

# EET_P1_11: Variant B — Adversarial + Entropy stabilizer (uses hard routing, deprecated)
# NOTE: adversarial variant uses hard routing in Phase 2 — router gets no gradient.
# Kept for reference; use quality variant instead.
#run_experiment "EET_P1_11_VARIANT_B_D${DEPTH}" \
#    "Variant B: adversarial gap + entropy stabilizer (no reconstruction/translators)" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.15 \
#    --eet-loss-variant adversarial \
#    --eet-adv-lambda 1.0 --eet-adv-entropy-lambda 0.2 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1



# EET_P1_3: Frozen KV + Frequency Prior (α=0.1)
#run_experiment "EET_P1_3_FREQ_PRIOR_D${DEPTH}" \
#    "Frozen KV + frequency prior (α=0.1)" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1
#
# EET_P1_4: Frozen KV + POS Prior (β=0.1)
#run_experiment "EET_P1_4_POS_PRIOR_D${DEPTH}" \
#    "Frozen KV + POS prior (β=0.1)" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp2 \
#    --eet-pos-prior-beta 0.1
#
# EET_P1_5: Frozen KV + Freq + POS Combined
#run_experiment "EET_P1_5_COMBINED_PRIORS_D${DEPTH}" \
#    "Frozen KV + frequency (α=0.1) + POS (β=0.1) priors" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1
#
# EET_P1_6: Router Architecture Ablation — Linear
#run_experiment "EET_P1_6_ROUTER_LINEAR_D${DEPTH}" \
#    "Frozen KV + combined priors + LINEAR router" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type linear \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1
#
# EET_P1_7: Router Architecture Ablation — MLP1
#run_experiment "EET_P1_7_ROUTER_MLP1_D${DEPTH}" \
#    "Frozen KV + combined priors + MLP1 router" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp1 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1
#
# EET_P1_8: Full Pipeline — All priors + 3-phase training with exploration loss
#run_experiment "EET_P1_8_FULL_PIPELINE_D${DEPTH}" \
#    "Full pipeline: frozen KV + freq+POS priors + 3-phase training" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.15 \
#    --eet-reconstruct-lambda 1.0 \
#    --eet-efficiency-lambda-start 0.01 --eet-efficiency-lambda-end 0.1
#
# EET_P1_9: Full Pipeline with aggressive efficiency pressure
#run_experiment "EET_P1_9_AGGRESSIVE_D${DEPTH}" \
#    "Full pipeline with stronger efficiency pressure (λ_e: 0.05→0.3)" \
#    --use-eet 1 --eet-frozen-kv 1 \
#    --eet-router-type mlp2 \
#    --eet-freq-prior-alpha 0.1 --eet-pos-prior-beta 0.1 \
#    --eet-warmup-frac 0.02 --eet-explore-frac 0.15 \
#    --eet-reconstruct-lambda 1.0 \
#    --eet-efficiency-lambda-start 0.05 --eet-efficiency-lambda-end 0.3

# ── New Attention and Reentry Variants ───────────────────────────────────────

# EET_P1_25: Option A — Pure MoD Attention Skipping (no reentry)
#run_experiment "EET_P1_25_OPTION_A_D${DEPTH}" \
#    "Option A: pure MoD attention skipping for exited tokens (mlp2 router)" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
#    --eet-router-type mlp2 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema
#
# EET_P1_26: Option B — Frozen KV Injection (no reentry)
#run_experiment "EET_P1_26_OPTION_B_D${DEPTH}" \
#    "Option B: frozen KV injection for exited tokens (mlp2 router)" \
#    --use-eet 1 --eet-frozen-kv 1 --eet-reenter-final 0 \
#    --eet-router-type mlp2 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema
#
# EET_P1_27: Layer 8 Reentry (Option A + Final Layer Reentry)
#run_experiment "EET_P1_27_REENTER_L8_D${DEPTH}" \
#    "Option A + Layer 8 Reentry: exited tokens re-enter and are processed by the final layer" \
#    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 1 \
#    --eet-router-type mlp2 \
#    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
#    --eet-loss-variant ce_guided \
#    --eet-global-router 1 \
#    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 \
#    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
#    --eet-depth-weight-type ema
#

# ============================================================================
# DIAGNOSTIC: Dense vs EET Comparison
# ============================================================================
# Trains a dense baseline and the clean EET baseline, then runs the
# diagnostic comparison script automatically.

run_experiment "EET_P1_DIAG_DENSE_D${DEPTH}" \
    "Diagnostic: Dense baseline (control)" \
    --use-eet 0

run_experiment "EET_P1_DIAG_EET_D${DEPTH}" \
    "Diagnostic: EET baseline (P1_22 config)" \
    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
    --eet-router-type mlp1 \
    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
    --eet-global-router 1 --eet-router-task-grad 1 \
    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1 \
    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
    --eet-capacity-alignment-lambda 1.0

# ============================================================================
# DIAGNOSTIC-GUIDED FIXES
# ============================================================================

# Option A: Depth-Scaled LR — deeper layers get higher LR to compensate for
# fewer surviving tokens providing gradient signal
run_experiment "EET_P1_DEPTH_LR_D${DEPTH}" \
    "Diagnostic fix: depth-scaled LR (Option A)" \
    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
    --eet-router-type mlp1 \
    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
    --eet-global-router 1 --eet-router-task-grad 1 \
    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
    --eet-capacity-alignment-lambda 1.0 \
    --eet-depth-lr-scale 1

# Option B: Depth Gradient Scaling — scale per-token CE loss by inverse of
# active fraction at the token's exit layer
run_experiment "EET_P1_GRAD_SCALE_D${DEPTH}" \
    "Diagnostic fix: depth gradient scaling (Option B)" \
    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
    --eet-router-type mlp1 \
    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
    --eet-global-router 1 --eet-router-task-grad 1 \
    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
    --eet-capacity-alignment-lambda 1.0 \
    --eet-depth-grad-scale 1

# Detach Aux: Stop CE-guided and surprise loss from flowing gradients into
# transformer blocks — only the router params get trained by aux losses
run_experiment "EET_P1_DETACH_AUX_D${DEPTH}" \
    "Diagnostic fix: detach aux losses from backbone" \
    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
    --eet-router-type mlp1 \
    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
    --eet-global-router 1 --eet-router-task-grad 1 \
    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
    --eet-capacity-alignment-lambda 1.0 \
    --eet-detach-aux-from-backbone 1

# ============================================================================
# BACKBONE GRADIENT ISOLATION — Detach Exit Representations
# ============================================================================
# The main CE loss for early-exit tokens sends conflicting gradients into the
# backbone (early layers pulled toward "prediction-ready" AND "good intermediate
# features" simultaneously). Detaching exit representations isolates the backbone
# so it only trains from final-layer tokens (clean dense-like objective).

# Detach exits at 10% active (aggressive exits, maximum compute savings)
run_experiment "EET_P1_DETACH_EXIT_D${DEPTH}" \
    "Detach exit from backbone (10% active)" \
    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
    --eet-router-type mlp1 \
    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
    --eet-global-router 1 --eet-router-task-grad 1 \
    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.10 \
    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
    --eet-capacity-alignment-lambda 1.0 \
    --eet-detach-exit-from-backbone 1

# Detach exits at 30% active (more backbone signal, still good compute savings)
run_experiment "EET_P1_DETACH_EXIT_30_D${DEPTH}" \
    "Detach exit from backbone (30% active)" \
    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
    --eet-router-type mlp1 \
    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
    --eet-global-router 1 --eet-router-task-grad 1 \
    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.30 \
    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
    --eet-capacity-alignment-lambda 1.0 \
    --eet-detach-exit-from-backbone 1

# Detach exits at 50% active (most backbone signal, moderate compute savings)
run_experiment "EET_P1_DETACH_EXIT_50_D${DEPTH}" \
    "Detach exit from backbone (50% active)" \
    --use-eet 1 --eet-frozen-kv 0 --eet-reenter-final 0 \
    --eet-router-type mlp1 \
    --eet-warmup-frac 0.0 --eet-explore-frac 0.0 \
    --eet-exit-adapter-rank 0 --eet-router-after-block 0 \
    --eet-loss-variant ce_guided --eet-capacity-schedule bell \
    --eet-global-router 1 --eet-router-task-grad 1 \
    --eet-ce-guided-lambda 1.0 --eet-surprise-lambda 0.1 --eet-min-exit-layer 1\
    --eet-gumbel-temp-start 1.0 --eet-gumbel-temp-end 0.1 --eet-gumbel-hard 1 \
    --eet-depth-weight-type ema --eet-compute-skip 1 --eet-target-active-frac 0.50 \
    --eet-reinforce-interval 0 --eet-reinforce-lambda 0.0 \
    --eet-ffn-skip 0 --eet-ffn-target-frac 0.00 --eet-model-lr-mult 1.0 --eet-router-lr-mult 1.0\
    --eet-capacity-alignment-lambda 1.0 \
    --eet-detach-exit-from-backbone 1

# Run diagnostic comparison if both completed
DENSE_CKPT="${EET_OUT_BASE}/EET_P1_DIAG_DENSE_D${DEPTH}/depth_${DEPTH}/ckpt_base/base"
EET_CKPT="${EET_OUT_BASE}/EET_P1_DIAG_EET_D${DEPTH}/depth_${DEPTH}/ckpt_base/base"
DIAG_OUT="${EET_OUT_BASE}/diagnostic_D${DEPTH}"

if [ -d "$DENSE_CKPT" ] && [ -d "$EET_CKPT" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running EET vs Dense Diagnostic Comparison"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python -m scripts.eet_dense_diagnostic \
        --dense-ckpt-dir "$DENSE_CKPT" \
        --eet-ckpt-dir "$EET_CKPT" \
        --output-dir "$DIAG_OUT" \
        --data-dir "${DATA_DIR:-data}" \
        --tokenizer-dir "${TOKENIZER_DIR:-tokenizer}" \
        --num-batches 50 \
        --num-grad-batches 5 \
        --device-batch-size 8 2>&1 | tee -a "$LOGFILE"
    echo "✅  Diagnostic comparison complete. Results in $DIAG_OUT"
else
    echo "⚠️  Skipping diagnostic: one or both checkpoints not found."
    echo "    Dense: $DENSE_CKPT"
    echo "    EET:   $EET_CKPT"
fi


echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  EET P01 Sweep Complete (depth=${DEPTH})"
echo "  Results: ${EET_OUT_BASE}"
echo "═══════════════════════════════════════════════════════════════"
