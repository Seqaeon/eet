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

RUNNER = resolve_runner()

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
    print(f"Starting EET Sweep for Depth {depth}")
    print(f"Target Tokens: {'Auto (per-model param count)' if target_tokens == -1 else f'{target_tokens:,}'}")
    print("=" * 64)
    
    aspect_ratio, head_dim, model_dim, target_dim = model_dims(depth, aspect_ratio=args.aspect_ratio)
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
        "--total-batch-size", str(total_batch_size),
        "--target-tokens", str(target_tokens),
        *(["--target-param-data-ratio", str(args.target_param_data_ratio)] if args.target_param_data_ratio > 0 else []),
        "--eval-every", str(eval_every),        
        "--log-every", str(log_every),
        "--save-every", str(args.save_every),
        "--warmup-ratio", str(warm_up_ratio),
        "--warmdown-ratio", str(getattr(args, 'warmdown_ratio', 0.5)),
        "--final-lr-frac", str(getattr(args, 'final_lr_frac', 0.05)),
        "--adam-beta2", str(adam_beta2),
        
        # EET parameters forwarding
        "--use-eet", str(args.use_eet),
        "--eet-frozen-kv", str(args.eet_frozen_kv),
        "--eet-router-type", str(args.eet_router_type),
        "--eet-router-hidden", str(args.eet_router_hidden),
        "--eet-freq-prior-alpha", str(args.eet_freq_prior_alpha),
        "--eet-pos-prior-beta", str(args.eet_pos_prior_beta),
        "--eet-domain-prior", str(args.eet_domain_prior),
        "--eet-warmup-frac", str(args.eet_warmup_frac),
        "--eet-explore-frac", str(args.eet_explore_frac),
        "--eet-reconstruct-lambda", str(args.eet_reconstruct_lambda),
        "--eet-efficiency-lambda-start", str(args.eet_efficiency_lambda_start),
        "--eet-efficiency-lambda-end", str(args.eet_efficiency_lambda_end),
        "--eet-translator-rank", str(args.eet_translator_rank),
        "--eet-max-frozen-kv-frac", str(args.eet_max_frozen_kv_frac),
        "--eet-exit-threshold", str(args.eet_exit_threshold),
        "--eet-min-exit-layer", str(args.eet_min_exit_layer),
        "--eet-loss-variant", str(args.eet_loss_variant),
        "--eet-topk-vocab", str(args.eet_topk_vocab),
        "--eet-entropy-lambda", str(args.eet_entropy_lambda),
        "--eet-surprise-lambda", str(args.eet_surprise_lambda),
        "--eet-adv-lambda", str(args.eet_adv_lambda),
        "--eet-adv-entropy-lambda", str(args.eet_adv_entropy_lambda),
        "--eet-quality-lambda", str(args.eet_quality_lambda),
        "--eet-quality-entropy-bonus", str(args.eet_quality_entropy_bonus),
        "--eet-gumbel-temp-start", str(args.eet_gumbel_temp_start),
        "--eet-gumbel-temp-end", str(args.eet_gumbel_temp_end),
        "--eet-gumbel-hard", str(args.eet_gumbel_hard),
        "--eet-commitment-beta", str(args.eet_commitment_beta),
        "--eet-global-router", str(args.eet_global_router),
        "--eet-freq-efficiency-alpha", str(args.eet_freq_efficiency_alpha),
        "--eet-diversity-lambda", str(args.eet_diversity_lambda),
        "--eet-ce-guided-lambda", str(args.eet_ce_guided_lambda),
        "--eet-router-lr-mult", str(args.eet_router_lr_mult),
        "--eet-model-lr-mult", str(args.eet_model_lr_mult),
        "--eet-depth-weight-type", str(args.eet_depth_weight_type),
        "--eet-depth-weight-max", str(args.eet_depth_weight_max),
        "--eet-use-override", str(args.eet_use_override),
        "--eet-override-prob-start", str(args.eet_override_prob_start),
        "--eet-override-prob-end", str(args.eet_override_prob_end),
        "--eet-reenter-final", str(args.eet_reenter_final),
        "--eet-compute-skip", str(args.eet_compute_skip),
        "--eet-target-active-frac", str(args.eet_target_active_frac),
        "--eet-capacity-schedule", str(args.eet_capacity_schedule),
        "--eet-exit-fracs", str(args.eet_exit_fracs),
        "--eet-capacity-alignment-lambda", str(args.eet_capacity_alignment_lambda),
        "--eet-router-task-grad", str(args.eet_router_task_grad),
        "--eet-reinforce-interval", str(args.eet_reinforce_interval),
        "--eet-reinforce-lambda", str(args.eet_reinforce_lambda),
        "--eet-exit-adapter-rank", str(args.eet_exit_adapter_rank),
        "--eet-router-after-block", str(args.eet_router_after_block),
        "--eet-ffn-skip", str(args.eet_ffn_skip),
        "--eet-ffn-target-frac", str(args.eet_ffn_target_frac),
        "--eet-ffn-full-attn", str(args.eet_ffn_full_attn),
        "--eet-depth-affine", str(args.eet_depth_affine),
        "--eet-capacity-anneal-frac", str(args.eet_capacity_anneal_frac),
        "--eet-learned-schedule", str(args.eet_learned_schedule),
        "--eet-departure-summary", str(args.eet_departure_summary),
        "--eet-route-consistency-lambda", str(args.eet_route_consistency_lambda),
        "--eet-dense-distill-interval", str(args.eet_dense_distill_interval),
        "--eet-dense-distill-lambda", str(args.eet_dense_distill_lambda),
        "--eet-depth-lr-scale", str(args.eet_depth_lr_scale),
        "--eet-depth-grad-scale", str(args.eet_depth_grad_scale),
        "--eet-detach-aux-from-backbone", str(args.eet_detach_aux_from_backbone),
        "--eet-detach-exit-from-backbone", str(args.eet_detach_exit_from_backbone),
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

    # Setup the sweep models
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
        
        fallback_ckpt_dir = (run_dir_path / f"ckpt_{model_name}").resolve()

        saved_ckpt_dir = None
        if model_name in state.get("unfinished", {}):
            saved_ckpt_dir = state["unfinished"][model_name].get("ckpt_dir")
        elif model_name in state.get("completed", {}):
            saved_ckpt_dir = state["completed"][model_name].get("ckpt_dir")

        if saved_ckpt_dir:
            temp_ckpt_dir = Path(saved_ckpt_dir)
            temp_actual = temp_ckpt_dir / model_name
            if temp_actual.exists() and glob.glob(str(temp_actual / "model_*.pt")):
                ckpt_dir = temp_ckpt_dir
            else:
                ckpt_dir = fallback_ckpt_dir
        else:
            ckpt_dir = fallback_ckpt_dir

        train_cmd_args = common_args + extra_args + [
            "--checkpoints-dir", str(ckpt_dir),
            "--model-tag", model_name
        ]
        
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
            print(f"  │  │  🆕  STARTING FRESH: [{model_name}]")
            print(f"  │     No checkpoints found — training from scratch.")
            print(f"  └─────────────────────────────────────────────────────┘\n")
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        cmd = RUNNER + ["-m", "scripts.base_train"] + train_cmd_args
        print(f"Running: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    print(line, end="", flush=True)
            process.communicate()
            
            if process.returncode != 0:
                print(f"Error training {model_name}. Marking as failed and continuing to next model.")
                results[f"{model_name}"] = "FAILED"
                continue
                
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
            
    failed_models = [n for n, v in results.items() if v == "FAILED"]

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
    names = [n for n in names if isinstance(results[n], dict)]
    if not names:
        print("No successful runs to plot.")
        if failed_models:
            print(f"\n[ERROR] The following models FAILED: {failed_models}")
            sys.exit(1)
        return
    bpbs = [float(results[n]["val_bpb"]) for n in names]

    bars = plt.bar(names, bpbs, color=sns.color_palette("husl", len(names)))
    
    plt.title(f"Validation BPB Comparison at Depth {depth} ({target_tokens:,} tokens)", fontsize=14)
    plt.ylabel("Validation Bits Per Byte (lower is better)", fontsize=12)
    plt.ylim(float(min(bpbs)) * 0.95, float(max(bpbs)) * 1.05)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', fontsize=10)
        
    plt.tight_layout()
    plot_path = run_dir_path / f"comparison_depth_{depth}.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    
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
    parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of peak LR (eta_min)")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to run (e.g. 'base,eet'), or 'all'")
    
    # New flags for run configuration
    parser.add_argument("--device-batch-size", type=int, default=-1, help="override per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=-1, help="override total batch size")
    parser.add_argument("--log-every", type=int, default=1, help="logging frequency")
    parser.add_argument("--eval-every", type=int, default=-1, help="evaluation frequency (-1 = at end)")
    parser.add_argument("--save-every", type=int, default=-1, help="checkpoint frequency")
    parser.add_argument("--sequence-len", type=int, default=2048, help="override max sequence length")
    
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
