import argparse
import sys
import os
import json
import glob
import shutil
import datetime
from pathlib import Path
import subprocess

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts._sweep_utils import resolve_runner, estimate_tokens_from_base, model_dims, check_and_prepare_env
from nanochat.checkpoint_manager import find_last_step
class ChunkedRemixConfig:
    def to_cli_args(self, model_dim): return []
    def summary(self): return "mock"



RUNNER = resolve_runner()

# ---------------------------------------------------------------------------
# Persistent sweep state helpers
# sweep_state.json lives inside --run-dir and survives re-runs.
# Format:
# {
#   "completed":  { "model_name": {"val_bpb": 1.23, "checkpoint": "...", "ckpt_dir": "..."} },
#   "unfinished": { "model_name": {"ckpt_dir": "...", "started_at": "..."} }
# }
# ---------------------------------------------------------------------------

def _state_path(run_dir_path: Path) -> Path:
    return run_dir_path / "sweep_state.json"

def load_sweep_state(run_dir_path: Path) -> dict:
    p = _state_path(run_dir_path)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": {}, "unfinished": {}}

def save_sweep_state(run_dir_path: Path, state: dict) -> None:
    p = _state_path(run_dir_path)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


def run_training_sweep(args):
    # Ensure environment is ready
    check_and_prepare_env(args)
    
    depth = args.depth
    run_dir = args.run_dir
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Load (or create) persistent state for this run_dir
    state = load_sweep_state(run_dir_path)
    if state["unfinished"]:
        print("="*64)
        print("Unfinished experiments found from a previous run:")
        for name, info in state["unfinished"].items():
            print(f"  [{name}] ckpt_dir = {info.get('ckpt_dir', '?')}")
        print("These will be resumed from their last checkpoint.")
        print("="*64)
    
    if args.target_tokens > 0:
        target_tokens = args.target_tokens
    elif args.target_tokens == 0:
        target_tokens = estimate_tokens_from_base(depth, tokenizer_dir=args.tokenizer_dir)
    else:
        target_tokens = -1

    print("=" * 64)
    print(f"Starting Sweep for Depth {depth}")
    print(f"Target Tokens: {'Auto (per-model param count)' if target_tokens == -1 else f'{target_tokens:,}'}")
    print("=" * 64)

    
    aspect_ratio, head_dim, model_dim, target_dim = model_dims(depth, aspect_ratio=args.aspect_ratio)
    if args.research_dim > 0:
        print(f"  Overriding default target_dim ({target_dim}) with --research-dim {args.research_dim}")
        target_dim = args.research_dim
    elif args.research_dim == -1:
        print(f"  Overriding default target_dim ({target_dim}) with full model_dim {model_dim}")
        target_dim = model_dim
    if args.model_dim > 0:
        model_dim = args.model_dim
    max_seq_len = args.sequence_len
    
    device_batch_size = args.device_batch_size if args.device_batch_size > 0 else {4: 8, 8: 32, 16: 16, 24: 8}.get(depth, 16)
    total_batch_size = args.total_batch_size if args.total_batch_size > 0 else 262144
    eval_every = args.eval_every
    log_every = args.log_every
    
    warm_up_ratio = args.warmup_ratio
    adam_beta2 = 0.99
    
    # Common kwargs for all models
    common_args = [
        "--depth", str(depth),
        "--aspect-ratio", str(aspect_ratio),
        "--head-dim", str(head_dim),
        "--model-dim", str(model_dim),
        "--max-seq-len", str(max_seq_len),
        "--device-batch-size", str(device_batch_size),
        "--total-batch-size", str(total_batch_size), # standard for reference
        "--target-tokens", str(target_tokens),
        *(["--target-param-data-ratio", str(args.target_param_data_ratio)] if args.target_param_data_ratio > 0 else []),
        "--eval-every", str(eval_every),        
        "--log-every", str(log_every),
        "--core-metric-every", "0" if args.skip_core else str(args.core_metric_every),
        "--save-every", str(args.save_every),
        "--warmup-ratio", str(warm_up_ratio),
        "--warmdown-ratio", str(getattr(args, 'warmdown_ratio', 0.5)),
        "--final-lr-frac", str(getattr(args, 'final_lr_frac', 0.0)),    # Safer for research models
        "--adam-beta2", str(adam_beta2),     # Matches notebook
        "--research-warmup-ratio", str(args.research_warmup_ratio),
        "--use-onecycle", str(args.use_onecycle),
        # EET: Early Exit Transformer
        "--use-eet", str(getattr(args, 'use_eet', 0)),
        "--eet-frozen-kv", str(getattr(args, 'eet_frozen_kv', 1)),
        "--eet-reenter-final", str(getattr(args, 'eet_reenter_final', 0)),
        "--eet-compute-skip", str(getattr(args, 'eet_compute_skip', 0)),
        "--eet-target-active-frac", str(getattr(args, 'eet_target_active_frac', 0.125)),
        "--eet-capacity-schedule", str(getattr(args, 'eet_capacity_schedule', 'bell')),
        "--eet-exit-fracs", str(getattr(args, 'eet_exit_fracs', '')),
        "--eet-router-type", str(getattr(args, 'eet_router_type', 'mlp2')),
        "--eet-router-hidden", str(getattr(args, 'eet_router_hidden', 0)),
        "--eet-freq-prior-alpha", str(getattr(args, 'eet_freq_prior_alpha', 0.0)),
        "--eet-pos-prior-beta", str(getattr(args, 'eet_pos_prior_beta', 0.0)),
        "--eet-domain-prior", str(getattr(args, 'eet_domain_prior', 0)),
        "--eet-warmup-frac", str(getattr(args, 'eet_warmup_frac', 0.02)),
        "--eet-explore-frac", str(getattr(args, 'eet_explore_frac', 0.15)),
        "--eet-reconstruct-lambda", str(getattr(args, 'eet_reconstruct_lambda', 1.0)),
        "--eet-efficiency-lambda-start", str(getattr(args, 'eet_efficiency_lambda_start', 0.01)),
        "--eet-efficiency-lambda-end", str(getattr(args, 'eet_efficiency_lambda_end', 0.1)),
        "--eet-translator-rank", str(getattr(args, 'eet_translator_rank', 0)),
        "--eet-max-frozen-kv-frac", str(getattr(args, 'eet_max_frozen_kv_frac', 0.75)),
        "--eet-exit-threshold", str(getattr(args, 'eet_exit_threshold', 0.5)),
        "--eet-min-exit-layer", str(getattr(args, 'eet_min_exit_layer', 1)),
        "--eet-loss-variant", str(getattr(args, 'eet_loss_variant', 'reconstruct')),
        "--eet-topk-vocab", str(getattr(args, 'eet_topk_vocab', 512)),
        "--eet-entropy-lambda", str(getattr(args, 'eet_entropy_lambda', 0.3)),
        "--eet-surprise-lambda", str(getattr(args, 'eet_surprise_lambda', 0.1)),
        "--eet-adv-lambda", str(getattr(args, 'eet_adv_lambda', 1.0)),
        "--eet-adv-entropy-lambda", str(getattr(args, 'eet_adv_entropy_lambda', 0.2)),
        "--eet-quality-lambda", str(getattr(args, 'eet_quality_lambda', 1.0)),
        "--eet-quality-entropy-bonus", str(getattr(args, 'eet_quality_entropy_bonus', 0.1)),
        "--eet-gumbel-temp-start", str(getattr(args, 'eet_gumbel_temp_start', 0.0)),
        "--eet-gumbel-temp-end", str(getattr(args, 'eet_gumbel_temp_end', 0.1)),
        "--eet-gumbel-hard", str(getattr(args, 'eet_gumbel_hard', 1)),
        "--eet-commitment-beta", str(getattr(args, 'eet_commitment_beta', 0.1)),
        "--eet-global-router", str(getattr(args, 'eet_global_router', 0)),
        "--eet-freq-efficiency-alpha", str(getattr(args, 'eet_freq_efficiency_alpha', 0.0)),
        "--eet-diversity-lambda", str(getattr(args, 'eet_diversity_lambda', 0.0)),
        "--eet-ce-guided-lambda", str(getattr(args, 'eet_ce_guided_lambda', 1.0)),
        "--eet-router-lr-mult", str(getattr(args, 'eet_router_lr_mult', 5.0)),
        "--eet-model-lr-mult", str(getattr(args, 'eet_model_lr_mult', 1.0)),
        "--eet-depth-weight-type", str(getattr(args, 'eet_depth_weight_type', 'none')),
        "--eet-depth-weight-max", str(getattr(args, 'eet_depth_weight_max', 2.5)),
        "--eet-use-override", str(getattr(args, 'eet_use_override', 0)),
        "--eet-override-prob-start", str(getattr(args, 'eet_override_prob_start', 0.5)),
        "--eet-override-prob-end", str(getattr(args, 'eet_override_prob_end', 0.1)),
        "--eet-capacity-alignment-lambda", str(getattr(args, 'eet_capacity_alignment_lambda', 0.0)),
        "--eet-router-task-grad", str(getattr(args, 'eet_router_task_grad', 1)),
        "--eet-reinforce-interval", str(getattr(args, 'eet_reinforce_interval', 0)),
        "--eet-reinforce-lambda", str(getattr(args, 'eet_reinforce_lambda', 0.1)),
        "--eet-exit-adapter-rank", str(getattr(args, 'eet_exit_adapter_rank', 0)),
        "--eet-router-after-block", str(getattr(args, 'eet_router_after_block', 0)),
        "--eet-ffn-skip", str(getattr(args, 'eet_ffn_skip', 0)),
        "--eet-ffn-target-frac", str(getattr(args, 'eet_ffn_target_frac', 0.50)),
        "--eet-ffn-full-attn", str(getattr(args, 'eet_ffn_full_attn', 1)),
        "--eet-depth-affine", str(getattr(args, 'eet_depth_affine', 0)),
        "--eet-capacity-anneal-frac", str(getattr(args, 'eet_capacity_anneal_frac', 0.0)),
        "--eet-learned-schedule", str(getattr(args, 'eet_learned_schedule', 0)),
        "--eet-departure-summary", str(getattr(args, 'eet_departure_summary', 0)),
        "--eet-route-consistency-lambda", str(getattr(args, 'eet_route_consistency_lambda', 0.0)),
        "--eet-dense-distill-interval", str(getattr(args, 'eet_dense_distill_interval', 0)),
        "--eet-dense-distill-lambda", str(getattr(args, 'eet_dense_distill_lambda', 0.5)),
        "--eet-depth-lr-scale", str(getattr(args, 'eet_depth_lr_scale', 0)),
        "--eet-depth-grad-scale", str(getattr(args, 'eet_depth_grad_scale', 0)),
        "--eet-detach-aux-from-backbone", str(getattr(args, 'eet_detach_aux_from_backbone', 0)),
        "--eet-detach-exit-from-backbone", str(getattr(args, 'eet_detach_exit_from_backbone', 0)),
    ]
    if args.compile:
        common_args.append("--compile")
    else:
        common_args.append("--no-compile")
    if getattr(args, "fp8", False):
        common_args.append("--fp8")
    if getattr(args, "tokenizer_dir", None):
        common_args.extend(["--tokenizer-dir", args.tokenizer_dir])
    if getattr(args, "data_dir", None):
        common_args.extend(["--data-dir", args.data_dir])
    if getattr(args, "max_shards", -1) != -1:
        common_args.extend(["--max-shards", str(args.max_shards)])




    
    # --- Optimal LR Configurations (from actual_lr_research_sweep) ---
    BEST_LRS = {
        "moe_no_perm": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
        "moe_perm": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
        "remixed-linear": {
            "embedding_lr":   0.104074,
            "unembedding_lr": 0.0245175,
            "matrix_lr":      0.0329274,
            "scalar_lr":      0.152507,
        },
    }
    # ---------------------------------------------------------------

    models = {
        "base": ["--use-eet", "0"],
        "eet": ["--use-eet", "1", "--eet-compute-skip", "1"],
    }
    
    # Filter models if requested
    target_models = args.models.split(",") if args.models != "all" else models.keys()
    filtered_models = {k: v for k, v in models.items() if k in target_models}
    if not filtered_models:
        print(f"No matching models found in {list(models.keys())} for selection '{args.models}'")
        return

    results = {}
    
    for model_name, extra_args in filtered_models.items():
        print(f"\n--- Training {model_name} ---")
        
        # Check for resumption. We resolve a fallback ckpt_dir in the current run environment
        # in case absolute paths stored in the sweep state from a previous run/different machine are invalid.
        fallback_ckpt_dir = (run_dir_path / f"ckpt_{model_name}").resolve()

        saved_ckpt_dir = None
        if model_name in state.get("unfinished", {}):
            saved_ckpt_dir = state["unfinished"][model_name].get("ckpt_dir")
        elif model_name in state.get("completed", {}):
            saved_ckpt_dir = state["completed"][model_name].get("ckpt_dir")

        if saved_ckpt_dir:
            temp_ckpt_dir = Path(saved_ckpt_dir)
            temp_actual = temp_ckpt_dir / model_name
            # Verify if this saved checkpoint directory exists and actually contains model checkpoint files
            if temp_actual.exists() and glob.glob(str(temp_actual / "model_*.pt")):
                ckpt_dir = temp_ckpt_dir
            else:
                ckpt_dir = fallback_ckpt_dir
        else:
            ckpt_dir = fallback_ckpt_dir

        # ChunkedRemixConfig: when --use-chunked-remix 1, inject canonical config defaults
        # as a *prefix* before common_args so explicit sweep flags (which come later) win.
        chunked_prefix: list[str] = []
        if getattr(args, 'use_chunked_remix', False) and model_name == "remixed-linear":
            _cfg = ChunkedRemixConfig()
            chunked_prefix = _cfg.to_cli_args(model_dim=model_dim)
            print(f"  [ChunkedRemixConfig] {_cfg.summary()}")

        train_cmd_args = chunked_prefix + common_args + extra_args + [
            "--checkpoints-dir", str(ckpt_dir),
            "--model-tag", model_name
        ]
        
        # Handle mu-P scaling based on the new mode system
        if args.mu_p_mode == "disable":
            if model_name != "base":
                train_cmd_args.append("--disable-mu-p")
        elif args.mu_p_mode == "base_only":
            if model_name != "base":
                # Force research models to use the exact same multiplier base would've used
                base_multiplier = (model_dim / 768) ** -0.5
                train_cmd_args.extend(["--mu-p-scale-override", str(base_multiplier)])
        # if "enable", neither flag is passed; models calculate their own inherently

        # ── Record as unfinished BEFORE launching so a crash is visible ──
        actual_model_ckpt_dir = ckpt_dir / model_name
        state.setdefault("unfinished", {})[model_name] = {
            "ckpt_dir": str(ckpt_dir),
            "actual_model_ckpt_dir": str(actual_model_ckpt_dir),
            "started_at": datetime.datetime.now().isoformat(),
        }
        save_sweep_state(run_dir_path, state)

        # Check for resumption
        try:
            last_step = find_last_step(str(actual_model_ckpt_dir))
            print(f"\n  ┌─────────────────────────────────────────────────────┐")
            print(f"  │  ⏩  RESUMING [{model_name}] from step {last_step:,}")
            print(f"  │     {str(actual_model_ckpt_dir)}")
            print(f"  └─────────────────────────────────────────────────────┘\n")
            train_cmd_args.extend(["--resume-from-step", str(last_step)])
        except FileNotFoundError:
            print(f"\n  ┌─────────────────────────────────────────────────────┐")
            print(f"  │  🆕  STARTING FRESH: [{model_name}]")
            print(f"  │     No checkpoints found — training from scratch.")
            print(f"  └─────────────────────────────────────────────────────┘\n")
        

        # Need to preserve environment variables, especially LD_LIBRARY_PATH for cusparseLt
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Each model is trained as a proper DDP job via torchrun.
        cmd = RUNNER + ["-m", "scripts.base_train"] + train_cmd_args
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # We stream stdout so user isn't stuck waiting blindly
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(line, end="", flush=True)
            process.communicate()
            
            if process.returncode != 0:
                print(f"Error training {model_name}. Marking as failed and continuing to next model.")
                results[f"{model_name}"] = "FAILED"
                continue
                
            # Extract final checkpoint val_bpb
            # Checkpoint format is usually checkpoints_dir/model/state_*.pt etc
            # Base train saves it with step index. We find the largest one.
            model_ckpt_dir = ckpt_dir / model_name
            if model_ckpt_dir.exists():
                meta_files = glob.glob(str(model_ckpt_dir / "meta_*.json"))
                if meta_files:
                    meta_files.sort()
                    last_meta = meta_files[-1]
                    try:
                        with open(last_meta, "r") as f:
                            meta_data = json.load(f)
                        if "val_bpb" in meta_data and meta_data["val_bpb"] is not None:
                            val_bpb = float(meta_data["val_bpb"])
                            results[model_name] = {"val_bpb": val_bpb, "checkpoint": last_meta}
                            # ── Promote to completed in state ──
                            state.setdefault("completed", {})[model_name] = {
                                "val_bpb": val_bpb,
                                "checkpoint": last_meta,
                                "ckpt_dir": str(ckpt_dir),
                            }
                            state.get("unfinished", {}).pop(model_name, None)
                            save_sweep_state(run_dir_path, state)
                            print(f"Final Validation BPB for {model_name}: {val_bpb:.4f}")
                        else:
                            print(f"No val_bpb found in {last_meta}")
                    except Exception as e:
                        print(f"Failed to load metadata {last_meta}: {e}")
                else:
                    print(f"No meta_*.json files found in {model_ckpt_dir}")
            else:
                 print(f"Checkpoint directory {model_ckpt_dir} does not exist.")
                 
        except Exception as e:
            print(f"Exception during {model_name}: {e}")
            
    # --- Fail fast if any model errored ---
    failed_models = [n for n, v in results.items() if v == "FAILED"]

    # --- Generate Report and Plot ---
    if not results:
        print("No results collected to plot.")
        if failed_models:
            print(f"Failed models: {failed_models}")
            sys.exit(1)
        return
        
    print("\n--- Generating Report ---")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    # Filter out failed runs (stored as string "FAILED", not a result dict)
    names = [n for n in names if isinstance(results[n], dict)]
    if not names:
        print("No successful runs to plot.")
        if failed_models:
            print(f"\n[ERROR] The following models FAILED: {failed_models}")
            sys.exit(1)
        return
    # Ensure all collected BPBs are floats for math
    bpbs = [float(results[n]["val_bpb"]) for n in names]

    bars = plt.bar(names, bpbs, color=sns.color_palette("husl", len(names)))
    
    plt.title(f"Validation BPB Comparison at Depth {depth} ({target_tokens:,} tokens)", fontsize=14)
    plt.ylabel("Validation Bits Per Byte (lower is better)", fontsize=12)
    plt.ylim(float(min(bpbs)) * 0.95, float(max(bpbs)) * 1.05) # Zoom in for better contrast

    # Add exact values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=10)
        
    plt.tight_layout()
    plot_path = run_dir_path / f"comparison_depth_{depth}.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Save TSV data
    tsv_path = run_dir_path / f"results_depth_{depth}.tsv"
    with open(tsv_path, "w") as f:
        f.write("model_name\tval_bpb\n")
        for name, data in results.items():
            if isinstance(data, dict):
                f.write(f"{name}\t{data['val_bpb']}\n")
            else:
                f.write(f"{name}\tFAILED\n")
    print(f"Saved TSV data to {tsv_path}")

    if failed_models:
        print(f"\n[ERROR] The following models FAILED: {failed_models}")
        sys.exit(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--aspect-ratio", type=int, default=0, help="model_dim = depth * aspect_ratio (0 = use defaults)")
    parser.add_argument("--model-dim", type=int, default=0, help="Explicit model_dim override for base_train.py")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 training (Blackwell optimization)")
    parser.add_argument("--tokenizer-dir", type=str, default=None, help="explicit tokenizer directory")
    parser.add_argument("--data-dir", type=str, default=None, help="explicit data directory")
    parser.add_argument("--max-shards", type=int, default=-1, help="maximum number of dataset shards to use")
    parser.add_argument("--target-tokens", type=int, default=-1, help="explicit number of tokens to train for per model")
    parser.add_argument("--target-param-data-ratio", type=float, default=-1.0, help="Chinchilla token:param ratio (e.g. 20.0); -1 = use base_train.py default (10.5)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="enable/disable torch.compile")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="base warmup ratio passed to all runs")
    parser.add_argument("--warmdown-ratio", type=float, default=0.7, help="ratio of iterations for LR warmdown (rest is constant LR)")
    parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of peak LR (eta_min)")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to run (e.g. 'base,remixed-linear'), or 'all'")
    parser.add_argument("--research-warmup-ratio", type=float, default=0.05, help="research-branch warmup ratio for OneCycle")
    parser.add_argument("--use-onecycle", type=int, default=1, choices=[0, 1], help="research branches: 1=OneCycle, 0=use base schedule")
    
    # New flags for run configuration
    parser.add_argument("--device-batch-size", type=int, default=-1, help="override per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=-1, help="override total batch size")
    parser.add_argument("--log-every", type=int, default=1, help="logging frequency")
    parser.add_argument("--eval-every", type=int, default=-1, help="evaluation frequency (-1 = at end)")
    parser.add_argument("--save-every", type=int, default=-1, help="checkpoint frequency")
    parser.add_argument("--core-metric-every", type=int, default=-1, help="core metric frequency")
    parser.add_argument("--skip-core", action="store_true", help="completely disable CORE metric evaluation")
    parser.add_argument("--mu-p-mode", type=str, default="base_only", choices=["disable", "base_only", "enable"], help="mu-P scaling logic")
    parser.add_argument("--sequence-len", type=int, default=2048, help="override max sequence length")
    parser.add_argument("--router-context-window", type=int, default=-1, help="override sliding window size for contextual router (-1 for full sequence)")
    # Research dimension override
    parser.add_argument("--research-dim", type=int, default=0, help="override default 1/8th model_dim for research branches (MoE/Remix)")
    parser.add_argument("--remix-basis-size", type=int, default=0, help="explicit basis_size for remixed-linear (0 = auto via scale_basis_size; set to model_dim for full-rank)")
    # Remixed-linear components
    parser.add_argument("--remix-use-basis-gate", type=int, default=1, choices=[0, 1], help="enable basis gating in remixed linear (1/0)")
    parser.add_argument("--remix-use-output-gate", type=int, default=1, choices=[0, 1], help="enable output gating in remixed linear (1/0)")
    parser.add_argument("--remix-use-context", type=int, default=1, choices=[0, 1], help="enable context modulation in remixed linear (1/0)")
    parser.add_argument("--remix-basis-gate-mode", type=str, default="mlp", choices=["mlp", "linear", "centered", "attn", "none", "random", "lowrank"], help="basis gate architecture")
    parser.add_argument("--p22-n-templates", type=int, default=1, help="22: number of template_mixing matrices (1=standard, K>1=MoE routing)")
    parser.add_argument("--p22-template-routing-learned", type=int, default=0, choices=[0, 1], help="22: learned template routing (0=frozen, 1=learned)")
    parser.add_argument("--p22-template-topk", type=int, default=0, help="22: hard top-k for legacy template bank")
    parser.add_argument("--p22-attn-moe-route", type=str, default="none", choices=["none", "sequence", "token"], help="22: MoE routing for attention Q/K/V/Proj")
    parser.add_argument("--remix-gate-lr-scale", type=float, default=0.3, help="remix: learning rate scale for gate parameters")
    parser.add_argument("--p26-output-gated-linear", type=int, default=0, choices=[0, 1], help="26: use OutputGatedLinear (single W + low-rank output gate, no factorization)")
    parser.add_argument("--p28-shared-basis", type=int, default=0, choices=[0, 1], help="28C: share single W_b projection across all attn Q/K/V/O per block")
    parser.add_argument("--p28-chunk-routing-size", type=int, default=0, help="28D: amortize template routing over N-token chunks (0=per-token)")
    parser.add_argument("--p28-global-template-bank", type=str, default="none", choices=["none", "ffn", "all"], help="28E/F: cross-layer global template bank mode")
    parser.add_argument("--p28-attn-proj-templates", type=int, default=0, help="28C2: override n_templates for attn c_proj (0=default)")
    parser.add_argument("--p28-attn-qk-templates",   type=int, default=0, help="28C3: override n_templates for attn c_q/c_k (0=default)")
    parser.add_argument("--target-active-params",     type=int, default=0, choices=[0, 1], help="use active params for target_tokens budget")
    parser.add_argument("--remix-basis-gate-rank", type=int, default=8, help="rank for lowrank basis gate mode")
    # CCL block modulation
    parser.add_argument("--cclblock-modulation", type=str, default="weight",
                        choices=["weight", "normalization", "householder", "spectral", "ocd", "lie", "polynomial", "grassmann", "decoupled", "tucker", "svs", "vq", "dcu", "fsi", "aesp", "ckr", "ckr_ffn", "com", "giad", "psg", "splitstream", "lokr", "pgr", "cil", "prb", "arg", "kfl"],
                        help="CCL block strategy: 'weight' (RemixedLinear+SelectiveContextStream) "
                             "or 'normalization' (CCLBlock with AdaRMSNorm)")
    parser.add_argument("--cclblock-orth-lambda", type=float, default=0.0,
                        help="OCD overlap penalty weight (0 disables)")
    parser.add_argument("--cclblock-context-stream", type=str, default="local", 
                        choices=["local", "shifted", "ema", "selective", "multiscale", "ssm", "boundary", "chunk", "predictive_chunk", "evidence_ssm", "dacs", "prefix", "warmup_ema", "dacs_ema", "decay_prefix"],
                        help="Context stream type")
    parser.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                        help="EMA factor for the legacy EMAContextStream")
    parser.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                        help="Design C stale context lag (0=disabled, k>=1 = context from k blocks ago)")
    # Novel ablation designs
    parser.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                        help="Design 3: sparse top-k basis gate (0=off, N=top-N)")
    parser.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                        help="Design 6: gate temperature (<1=sharper, >1=softer)")
    parser.add_argument("--cclblock-context-bank-size", type=int, default=0,
                        help="Design 4: context prototype bank size (0=off, e.g. 16)")
    parser.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                        help="Design 7: separate attn/ffn context projections (0=off, 1=on)")
    parser.add_argument("--cclblock-context-source", type=str, default="norm_x",
                        choices=["norm_x", "attn_heads", "attn_geometry"],
                        help="Design 2: context source ('norm_x'=residual, 'attn_heads'=query vectors)")
    # Phase 8
    parser.add_argument("--cclblock-chunk-size", type=int, default=0)
    parser.add_argument("--cclblock-aux-objective", type=str, default="none", choices=["none", "boundary", "entropy"])
    parser.add_argument("--cclblock-aux-lambda", type=float, default=0.1)
    parser.add_argument("--cclblock-boundary-token-id", type=int, default=198)
    # Phase 9
    parser.add_argument("--use-ral", type=int, default=0, choices=[0, 1])
    parser.add_argument("--ral-rank", type=int, default=32)
    parser.add_argument("--cclblock-film-gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-attn-shadow-dim", type=int, default=0)
    parser.add_argument("--cclblock-dynamic-ratio", type=float, default=0.25)
    parser.add_argument("--cclblock-gate-rank", type=int, default=8)
    parser.add_argument("--cclblock-num-regimes", type=int, default=8)
    parser.add_argument("--cclblock-regime-temperature", type=float, default=1.0)
    parser.add_argument("--cclblock-poly-order", type=int, default=2)
    parser.add_argument("--cclblock-lie-generators", type=int, default=4)
    parser.add_argument("--cclblock-grassmann-bank-size", type=int, default=4)
    parser.add_argument("--cclblock-tucker-rank", type=int, default=32)
    parser.add_argument("--cclblock-tucker-modes", type=int, default=8)
    parser.add_argument("--cclblock-svs-rank", type=int, default=64)
    parser.add_argument("--cclblock-svs-eps", type=float, default=0.1)
    parser.add_argument("--cclblock-vq-codes", type=int, default=8)
    parser.add_argument("--cclblock-vq-temperature", type=float, default=1.0)
    parser.add_argument("--cclblock-dcu-warmup-steps", type=int, default=0)
    # Phase 12: FSI/AESP/CKR
    parser.add_argument("--cclblock-fsi-rotations", type=int, default=8)
    parser.add_argument("--cclblock-fsi-selector-dim", type=int, default=64)
    parser.add_argument("--cclblock-aesp-strata", type=int, default=4)
    parser.add_argument("--cclblock-aesp-delta-rank", type=int, default=4)
    parser.add_argument("--cclblock-ckr-branches", type=int, default=4)
    parser.add_argument("--cclblock-ckr-kernel-size", type=int, default=64)
    # Phase 13: CKR enhancements
    parser.add_argument("--cclblock-ckr-pos-channels", type=int, default=1)
    parser.add_argument("--cclblock-ckr-dual-optim", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-ckr-content-bias", type=float, default=0.0)
    # Phase 14: GIAD/PSG/SplitStream
    parser.add_argument("--cclblock-giad-rank", type=int, default=32)
    parser.add_argument("--cclblock-psg-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-ss-dynamic-ratio", type=float, default=0.25)
    parser.add_argument("--cclblock-ss-branches", type=int, default=2)
    parser.add_argument("--cclblock-ss-kernel-size", type=int, default=64)
    # Phase 15: LoKR
    parser.add_argument("--cclblock-lokr-branches", type=int, default=8)
    parser.add_argument("--cclblock-lokr-rank", type=int, default=16)
    # Phase 16: CKR-Anneal / COM
    parser.add_argument("--cclblock-ckr-temp-start", type=float, default=2.0)
    parser.add_argument("--cclblock-ckr-temp-end", type=float, default=0.3)
    parser.add_argument("--cclblock-com-kernel-size", type=int, default=32)
    # Phase 17
    parser.add_argument("--cclblock-ckr-ortho-init", type=int, default=0, choices=[0, 1])
    parser.add_argument("--cclblock-ckr-branch-dropout", type=float, default=0.0)
    parser.add_argument("--cclblock-ckr-diversity-lambda", type=float, default=0.0)
    parser.add_argument("--cclblock-pgr-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-cil-kernel-size", type=int, default=64)
    parser.add_argument("--cclblock-prb-kernel-size", type=int, default=64)
    parser.add_argument("--modulation-diagnostics", type=int, default=0, choices=[0, 1])
    # Phase 18
    parser.add_argument("--p18-layer-drop", type=float, default=0.0)
    parser.add_argument("--p18-dynamic-activation", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p18-mixture-norm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p18-aux-sim-lambda", type=float, default=0.0)
    parser.add_argument("--p18-gradient-penalty", type=float, default=0.0)
    parser.add_argument("--p18-per-channel-scale", type=int, default=0, choices=[0, 1])
    # Phase 19
    parser.add_argument("--p19-residual-gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-head-importance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-residual-mix-groups", type=int, default=0)
    parser.add_argument("--p19-attn-logit-bias", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-residual-decay", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-grad-equilibrium", type=float, default=0.0)
    parser.add_argument("--p19-spectral-reparam", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--p19-weight-anticollapse", type=float, default=0.0)
    parser.add_argument("--p19-ve-bias", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p19-weight-noise", type=float, default=0.0)
    # Phase 20
    parser.add_argument("--p20-hrcs-scale", type=int, default=0)
    parser.add_argument("--p20-lswr-scale", type=int, default=0)
    parser.add_argument("--p20-lswr-planes", type=int, default=8)
    parser.add_argument("--p20-lrcfb-branches", type=int, default=0)
    parser.add_argument("--p20-lrcfb-narrow", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-lrcfb-learned", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-lrcfb-topk", type=int, default=0)
    parser.add_argument("--p20-dgcr-branches", type=int, default=0)
    parser.add_argument("--p20-dgcr-aux-weight", type=float, default=0.01)
    parser.add_argument("--p20-mone-experts", type=int, default=0)
    parser.add_argument("--p20-mone-topk", type=int, default=0)
    parser.add_argument("--p20-mone-narrow", type=int, default=1, choices=[0, 1])
    parser.add_argument("--p20-mone-frozen", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-ncea-branches", type=int, default=0)
    parser.add_argument("--p20-ncea-eps", type=float, default=0.1)
    parser.add_argument("--p20-adwi", type=int, default=0, choices=[0, 1])
    # Phase 2 proposals
    parser.add_argument("--p20-pwu-branches", type=int, default=0)
    parser.add_argument("--p20-pwu-phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--p20-fsvd-gate", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p20-wbfc-clusters", type=int, default=0)
    parser.add_argument("--p20-wbfc-active", type=int, default=0)
    # Phase 21
    parser.add_argument("--p21-per-experts", type=int, default=0)
    parser.add_argument("--p21-per-topk", type=int, default=0)
    parser.add_argument("--p21-per-learned", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p21-per-attn", type=int, default=0, choices=[0, 1])
    # Phase 23: Tiny Experts RemixedLinear + Standard MoE baseline
    parser.add_argument("--p23-tiny-expert", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-n-experts", type=int, default=64)
    parser.add_argument("--p23-topk", type=int, default=16)
    parser.add_argument("--p23-learned-route", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-std-moe-experts", type=int, default=0)
    parser.add_argument("--p23-std-moe-topk", type=int, default=-1)
    parser.add_argument("--p23-std-moe-aux-weight", type=float, default=0.01)
    parser.add_argument("--p23-lokr", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-lokr-rank", type=int, default=4)
    parser.add_argument("--p23-use-shared-block-router", type=int, default=0, choices=[0, 1])
    parser.add_argument("--p23-linear-moe-experts", type=int, default=0, help="23: enable weight-space LinearMoE with K experts (0=off)")
    parser.add_argument("--p23-linear-moe-topk", type=int, default=0, help="23: top-k selected experts in LinearMoE (0=soft all-expert blend)")
    parser.add_argument("--p23-quantile-route", type=int, default=0, choices=[0, 1, 2], help="23: 1=EMA quantile routing, 2=Causal Expert Cross-Attention")
    def _add_unique(*opt, **kwargs):
        # Guard against accidental duplicate definitions across merged branches.
        if any(o in parser._option_string_actions for o in opt):
            return
        parser.add_argument(*opt, **kwargs)

    _add_unique("--p24-use-sliced-weight", type=int, default=0, choices=[0, 1])
    _add_unique("--p24-sliced-weight-reduction-scale", type=int, default=8)
    _add_unique("--p24-sliced-weight-min-select", type=int, default=128)
    _add_unique("--p24-sliced-weight-scope", type=str, default="per_token", choices=["per_token", "per_block", "global"])
    _add_unique("--p24-sliced-weight-balance-coeff", type=float, default=0.01)
    _add_unique("--p24-quantile-route", type=int, default=0, choices=[0, 1, 2])
    _add_unique("--p24-use-folded-mod", type=int, default=0, choices=[0, 1])
    _add_unique("--p24-folded-mod-reduction-scale", type=int, default=8)
    _add_unique("--p24-folded-mod-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"])
    _add_unique("--p24-folded-mod-gate-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"])
    _add_unique("--p24-folded-mod-min-dim", type=int, default=128, help="floor on folded_dim (0=no floor, 128=match min_select default)")
    _add_unique("--p24-use-sequence-gated-linear", type=int, default=0, choices=[0, 1])
    _add_unique("--p24-sequence-gated-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"])
    _add_unique("--p24-sequence-gated-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"])
    parser.add_argument("--remix-shared-context-gates", type=int, default=0, choices=[0, 1], help="23: batch context gates")
    parser.add_argument("--remix-use-dual-gate", type=int, default=0, choices=[0, 1], help="25: use DualGateLinear instead of RemixedLinear")
    parser.add_argument("--remix-basis-scale-factor", type=int, default=4, help="basis compression: 4=C//4, 1=full rank")
    parser.add_argument("--remix-output-gate-rank", type=int, default=16, help="output gate rank")
    parser.add_argument("--use-chunked-remix", type=int, default=0, choices=[0, 1],
                        help="1 = activate ChunkedRemixConfig canonical P29 defaults for remixed-linear runs; "
                             "individual flags in the sweep still override on top")
    # Phase 30: LayerNorm ablation
    parser.add_argument("--remix-disable-ln-basis", type=int, default=0, choices=[0, 1], help="30B: disable intermediate LN in RemixedLinear")
    parser.add_argument("--dense-intermediate-ln", type=int, default=0, choices=[0, 1], help="30A: add intermediate LN to dense projections")
    # MST: Modular Sub-Transformer
    parser.add_argument("--use-mst", type=int, default=0, choices=[0, 1], help="MST: enable Modular Sub-Transformer mode")
    parser.add_argument("--mst-n-subs", type=int, default=8, help="MST: number of sub-transformers")
    parser.add_argument("--mst-sub-dim", type=int, default=64, help="MST: dimension per sub-transformer")
    parser.add_argument("--mst-head-dim", type=int, default=0, help="MST: attention head_dim (0=auto d//n_head)")
    parser.add_argument("--mst-input-mode", type=str, default="fixed_slice",
                        choices=["fixed_slice", "learned_proj", "rotated_slice", "per_sub_embed", "stem"])
    parser.add_argument("--mst-rotated-slice-learned", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-routing-mode", type=str, default="soft_weighted",
                        choices=["soft_weighted", "topk_hard", "sequence_path"])
    parser.add_argument("--mst-routing-topk", type=int, default=4)
    parser.add_argument("--mst-routing-aux-weight", type=float, default=0.01)
    parser.add_argument("--mst-diversity-weight", type=float, default=0.0)
    parser.add_argument("--mst-ffn-mode", type=str, default="standard", choices=["standard", "no_downproj", "linear"])
    parser.add_argument("--mst-transition-mode", type=str, default="parallel",
                        choices=["parallel", "aggregate_distribute", "cross_attend", "concat_proj", "free_for_all", "micro_attention", "micro_attention_shared_kv"])
    parser.add_argument("--mst-final-mode", type=str, default="aggregate_proj",
                        choices=["aggregate_proj", "weighted_logits", "concat_proj"])
    parser.add_argument("--mst-final-topk", type=int, default=-1)
    parser.add_argument("--mst-ffn-shared-up", type=int, default=0)
    parser.add_argument("--mst-ffn-inner-dim", type=int, default=0)
    parser.add_argument("--mst-sub-dropout", type=float, default=0.0)
    parser.add_argument("--mst-transition-every", type=int, default=1)
    parser.add_argument("--mst-ffa-temperature", type=float, default=1.0)
    parser.add_argument("--mst-global-residual", type=int, default=0)
    parser.add_argument("--mst-hybrid-dense", type=int, default=0)
    parser.add_argument("--mst-cross-sub-kv", type=int, default=0)
    # Stage 5 features
    parser.add_argument("--mst-sub-aux-weight", type=float, default=0.0, help="H3: per-sub auxiliary prediction loss weight")
    parser.add_argument("--mst-progressive-merge", type=int, default=0, choices=[0, 1], help="N1: pyramid sub-merging")
    parser.add_argument("--mst-multi-scale-windows", type=int, default=0, choices=[0, 1], help="W1: per-sub multi-scale windows")
    parser.add_argument("--mst-delta-residual", type=int, default=0, choices=[0, 1], help="DR1: delta residual mode")
    parser.add_argument("--mst-sub-layers", type=int, default=1, help="SL1: layers per sub-transformer")
    # Stage 7 features (P07)
    parser.add_argument("--mst-grad-equalize", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-block-diagonal-muon", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-transition-width-mult", type=float, default=1.0)
    parser.add_argument("--mst-sub-lr-scale", type=float, default=1.0)
    parser.add_argument("--mst-shared-expert", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-router-entropy-weight", type=float, default=0.0)
    parser.add_argument("--mst-shared-kv-attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-contrastive-diversity-weight", type=float, default=0.0)
    # MST Stage 8: Transition expressivity
    parser.add_argument("--mst-transition-nonlinear", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-transition-gated", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-transition-mlp", type=int, default=0, choices=[0, 1])
    # MST Stage 9: Cross-sub expressivity
    parser.add_argument("--mst-cross-sub-gate", type=int, default=0)
    parser.add_argument("--mst-hyper-connect", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-cross-kv-inject", type=int, default=0, choices=[0, 1])
    # MST Stage 10: Structural transition improvements
    parser.add_argument("--mst-slice-transition", type=int, default=0)
    parser.add_argument("--mst-lookback-layers", type=int, default=0)
    parser.add_argument("--mst-bilinear-transition", type=int, default=0, choices=[0, 1])
    # MST Stage 11: Attention bottleneck + structural improvements
    parser.add_argument("--mst-cross-sub-qmod", type=int, default=0)
    parser.add_argument("--mst-feature-cycle", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mst-mean-transition", type=int, default=0, choices=[0, 1])
    # EET: Early Exit Transformer
    parser.add_argument("--use-eet", type=int, default=0, choices=[0, 1], help="EET: enable Early Exit Transformer")
    parser.add_argument("--eet-frozen-kv", type=int, default=1, choices=[0, 1], help="EET: frozen KV injection (1) or masked attention (0)")
    parser.add_argument("--eet-reenter-final", type=int, default=0, choices=[0, 1], help="EET: force exited tokens to re-enter and be processed by the final layer (1) or not (0)")
    parser.add_argument("--eet-compute-skip", type=int, default=0, choices=[0, 1], help="EET: enable compute-level skipping of intermediate blocks (1/0)")
    parser.add_argument("--eet-target-active-frac", type=float, default=0.125, help="EET: target active token fraction at the deepest routable layer")
    parser.add_argument("--eet-capacity-schedule", type=str, default="bell", choices=["uniform", "linear", "geometric", "bell"])
    parser.add_argument("--eet-exit-fracs", type=str, default="")
    parser.add_argument("--eet-router-type", type=str, default="mlp2", choices=["linear", "mlp1", "mlp2", "attention", "attn"])
    parser.add_argument("--eet-router-hidden", type=int, default=0)
    parser.add_argument("--eet-freq-prior-alpha", type=float, default=0.0)
    parser.add_argument("--eet-pos-prior-beta", type=float, default=0.0)
    parser.add_argument("--eet-domain-prior", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-warmup-frac", type=float, default=0.02)
    parser.add_argument("--eet-explore-frac", type=float, default=0.15)
    parser.add_argument("--eet-reconstruct-lambda", type=float, default=1.0)
    parser.add_argument("--eet-efficiency-lambda-start", type=float, default=0.01)
    parser.add_argument("--eet-efficiency-lambda-end", type=float, default=0.1)
    parser.add_argument("--eet-translator-rank", type=int, default=0)
    parser.add_argument("--eet-max-frozen-kv-frac", type=float, default=0.75)
    parser.add_argument("--eet-exit-threshold", type=float, default=0.5)
    parser.add_argument("--eet-min-exit-layer", type=int, default=1)
    parser.add_argument("--eet-loss-variant", type=str, default="reconstruct")
    parser.add_argument("--eet-topk-vocab", type=int, default=512)
    parser.add_argument("--eet-entropy-lambda", type=float, default=0.3)
    parser.add_argument("--eet-surprise-lambda", type=float, default=0.1)
    parser.add_argument("--eet-adv-lambda", type=float, default=1.0)
    parser.add_argument("--eet-adv-entropy-lambda", type=float, default=0.2)
    parser.add_argument("--eet-quality-lambda", type=float, default=1.0)
    parser.add_argument("--eet-quality-entropy-bonus", type=float, default=0.1)
    parser.add_argument("--eet-gumbel-temp-start", type=float, default=0.0)
    parser.add_argument("--eet-gumbel-temp-end", type=float, default=0.1)
    parser.add_argument("--eet-gumbel-hard", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eet-commitment-beta", type=float, default=0.1)
    parser.add_argument("--eet-global-router", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-freq-efficiency-alpha", type=float, default=0.0)
    parser.add_argument("--eet-diversity-lambda", type=float, default=0.0)
    parser.add_argument("--eet-ce-guided-lambda", type=float, default=1.0)
    parser.add_argument("--eet-router-lr-mult", type=float, default=5.0)
    parser.add_argument("--eet-model-lr-mult", type=float, default=1.0)
    parser.add_argument("--eet-depth-weight-type", type=str, default="none", choices=["none", "linear", "ema", "sqrt"])
    parser.add_argument("--eet-depth-weight-max", type=float, default=2.5)
    parser.add_argument("--eet-use-override", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-override-prob-start", type=float, default=0.5)
    parser.add_argument("--eet-override-prob-end", type=float, default=0.1)
    parser.add_argument("--eet-capacity-alignment-lambda", type=float, default=0.0)
    parser.add_argument("--eet-router-task-grad", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eet-reinforce-interval", type=int, default=0)
    parser.add_argument("--eet-reinforce-lambda", type=float, default=0.1)
    parser.add_argument("--eet-exit-adapter-rank", type=int, default=0)
    parser.add_argument("--eet-router-after-block", type=int, default=0)
    parser.add_argument("--eet-ffn-skip", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-ffn-target-frac", type=float, default=0.50)
    parser.add_argument("--eet-ffn-full-attn", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eet-depth-affine", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-capacity-anneal-frac", type=float, default=0.0)
    parser.add_argument("--eet-learned-schedule", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-departure-summary", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-route-consistency-lambda", type=float, default=0.0)
    parser.add_argument("--eet-dense-distill-interval", type=int, default=0)
    parser.add_argument("--eet-dense-distill-lambda", type=float, default=0.5)
    parser.add_argument("--eet-depth-lr-scale", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-depth-grad-scale", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-detach-aux-from-backbone", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eet-detach-exit-from-backbone", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
    
    run_training_sweep(args)
