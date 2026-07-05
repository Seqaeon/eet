"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import torch
import torch.nn.functional as F
torch.set_num_threads(8)
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000
torch._dynamo.config.optimize_ddp = False
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized, wrap_model
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3, HAS_FA4
from scripts.base_eval import evaluate_core
print_banner()


def check_router_learned(exit_probs, token_ids, orig_model):
    """
    Run this at end of Phase 2 before committing to Phase 3.
    Check if exit_probs correlates with token frequency as TRS predicted.
    Uses orig_model._freq_prior.freq_bias as the authoritative frequency source
    (the exact signal the router saw during training).
    """
    import numpy as np
    from scipy import stats
    
    n_exits = exit_probs.size(-1)
    
    # Per-exit-slot probability stats (diagnose collapse)
    probs_cpu = exit_probs.detach().cpu().float()
    print0(f'[EET DIAGNOSTIC] Exit probability distribution across {n_exits} slots:')
    for slot in range(n_exits):
        slot_probs = probs_cpu[:, :, slot].numpy().ravel()
        label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
        print0(f'  {label}: mean={slot_probs.mean():.6f}  std={slot_probs.std():.6f}  '
               f'min={slot_probs.min():.6f}  max={slot_probs.max():.6f}')
    
    # Expected exit layer under soft distribution
    layer_indices = torch.arange(n_exits, device=exit_probs.device).float()
    expected_exit = (exit_probs * layer_indices).sum(dim=-1)  # (B, T)
    
    flat_ids  = token_ids.view(-1)
    flat_exit = expected_exit.view(-1).detach().cpu().numpy()
    
    print0(f'[EET DIAGNOSTIC] Mean expected exit layer: {flat_exit.mean():.4f}')
    print0(f'[EET DIAGNOSTIC] Std expected exit layer:  {flat_exit.std():.6f}')
    
    # Guard against constant router output (std=0 → Spearman undefined)
    if flat_exit.std() < 1e-8:
        print0('[EET DIAGNOSTIC] ⚠ Router output is CONSTANT across all tokens — no structure learned.')
        print0('[EET DIAGNOSTIC] Router has NOT learned structure yet. Extend Phase 2 or check gradients.')
        return 0.0
    
    # Build frequency signal — prefer _freq_prior.freq_bias (authoritative: exact signal the router saw)
    freq_prior = getattr(orig_model, '_freq_prior', None)
    if freq_prior is not None and hasattr(freq_prior, 'freq_bias'):
        freq_bias = freq_prior.freq_bias  # (vocab_size,), already log-normalized to [0,1]
        log_freq  = freq_bias[flat_ids.clamp(0, freq_bias.size(0) - 1)].cpu().float().numpy()
        freq_src  = '_freq_prior.freq_bias'
    else:
        # Fallback: load raw counts from disk and apply log1p
        import os
        from nanochat.common import get_base_dir
        cache_path = os.path.join(get_base_dir(), 'tokenizer', 'freq_table.pt')
        if os.path.exists(cache_path):
            ft = torch.load(cache_path, map_location='cpu')
            flat_ids_np = flat_ids.cpu().numpy()
            log_freq   = np.log1p(np.array([ft[i].item() for i in flat_ids_np]))
            freq_src   = f'freq_table.pt ({cache_path})'
        else:
            print0('[EET DIAGNOSTIC] ⚠ No frequency source available (no _freq_prior and no freq_table.pt). Skipping correlation.')
            return 0.0
    
    print0(f'[EET DIAGNOSTIC] Frequency source: {freq_src}')
    print0(f'[EET DIAGNOSTIC] log_freq — mean={log_freq.mean():.6f}  std={log_freq.std():.6f}'
           f'  min={log_freq.min():.6f}  max={log_freq.max():.6f}')
    
    if log_freq.std() < 1e-8:
        print0('[EET DIAGNOSTIC] ⚠ Frequency signal is constant (freq_table was saved with uniform fallback).')
        print0('[EET DIAGNOSTIC]   Run FrequencyPrior with access to training data shards to rebuild freq_table.pt.')
        return 0.0
    
    rho, p = stats.spearmanr(log_freq, flat_exit)
    print0(f'[EET DIAGNOSTIC] Spearman ρ (freq vs expected exit layer): {rho:.4f} (p={p:.4f})')
    
    if rho < -0.1 and p < 0.05:
        print0('[EET DIAGNOSTIC] ✓ Router has learned frequency-depth structure. Safe to enter Phase 3.')
    else:
        print0('[EET DIAGNOSTIC] ✗ Router has NOT learned structure yet. Extend Phase 2 or check gradients.')
    
    return rho

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--data-dir", type=str, default=None, help="dataset parquet directory (default: nanochat.dataset.DATA_DIR)")
parser.add_argument("--checkpoints-dir", type=str, default=None, help="base checkpoint root directory (default: <base_dir>/base_checkpoints)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--parallel", type=str, default="ddp", choices=["ddp", "dp"], help="ddp: DistributedDataParallel (via torchrun), dp: nn.DataParallel (for Kaggle/notebooks)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--model-dim", type=int, default=0, help="explicit model_dim override (0 = use aspect-ratio)")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")

# ── EET: Early Exit Transformer ──
parser.add_argument("--use-eet", type=int, default=0, choices=[0, 1], help="EET: enable Early Exit Transformer mode")
parser.add_argument("--eet-frozen-kv", type=int, default=1, choices=[0, 1], help="EET: 1=frozen KV injection (Option B), 0=masked attention (Option A)")
parser.add_argument("--eet-router-type", type=str, default="mlp2", choices=["linear", "mlp1", "mlp2", "attention", "attn"], help="EET: exit router architecture")
parser.add_argument("--eet-router-hidden", type=int, default=0, help="EET: router MLP hidden dim (0=n_embd//4)")
parser.add_argument("--eet-freq-prior-alpha", type=float, default=0.0, help="EET: frequency prior weight (0=disabled)")
parser.add_argument("--eet-pos-prior-beta", type=float, default=0.0, help="EET: POS prior weight (0=disabled)")
parser.add_argument("--eet-domain-prior", type=int, default=0, choices=[0, 1], help="EET: enable domain-conditioned routing")
parser.add_argument("--eet-warmup-frac", type=float, default=0.02, help="EET: Phase 1 dense warmup fraction")
parser.add_argument("--eet-explore-frac", type=float, default=0.15, help="EET: Phase 2 exploration fraction")
parser.add_argument("--eet-reconstruct-lambda", type=float, default=1.0, help="EET: reconstruction loss weight (λ_r)")
parser.add_argument("--eet-efficiency-lambda-start", type=float, default=0.01, help="EET: initial efficiency loss weight")
parser.add_argument("--eet-efficiency-lambda-end", type=float, default=0.1, help="EET: final efficiency loss weight")
parser.add_argument("--eet-translator-rank", type=int, default=0, help="EET: TunedLens translator rank (0=full)")
parser.add_argument("--eet-max-frozen-kv-frac", type=float, default=0.75, help="EET: max fraction of tokens that can exit")
parser.add_argument("--eet-exit-threshold", type=float, default=0.5, help="EET: sigmoid threshold for exit decision")
parser.add_argument("--eet-min-exit-layer", type=int, default=1, help="EET: earliest layer a token can exit at")
parser.add_argument("--eet-loss-variant", type=str, default="reconstruct", choices=["reconstruct", "entropy_surprise", "adversarial", "quality", "layer_weighted", "ce_guided"], help="EET: loss variant to use for early exit training")
parser.add_argument("--eet-topk-vocab", type=int, default=512, help="EET: top-k vocabulary size for cheap entropy calculation")
parser.add_argument("--eet-entropy-lambda", type=float, default=0.3, help="EET: entropy pressure weight (λ_ent) for entropy_surprise loss")
parser.add_argument("--eet-surprise-lambda", type=float, default=0.1, help="EET: surprise pressure weight (λ_sur) for entropy_surprise loss")
parser.add_argument("--eet-adv-lambda", type=float, default=1.0, help="EET: adversarial loss weight (λ_adv) for adversarial loss")
parser.add_argument("--eet-adv-entropy-lambda", type=float, default=0.2, help="EET: adversarial entropy pressure weight (λ_adv_ent) for adversarial loss")
parser.add_argument("--eet-quality-lambda", type=float, default=1.0, help="EET: REINFORCE quality loss weight for quality variant")
parser.add_argument("--eet-quality-entropy-bonus", type=float, default=0.1, help="EET: entropy bonus coefficient to prevent exit distribution collapse")
parser.add_argument("--eet-gumbel-temp-start", type=float, default=0.0, help="EET: Gumbel-Softmax starting temperature (0=disabled)")
parser.add_argument("--eet-gumbel-temp-end", type=float, default=0.1, help="EET: Gumbel-Softmax ending temperature")
parser.add_argument("--eet-gumbel-hard", type=int, default=1, choices=[0, 1], help="EET: enable Straight-Through Estimator for Gumbel")
parser.add_argument("--eet-commitment-beta", type=float, default=0.1, help="EET: commitment loss weight beta (0=disabled)")
parser.add_argument("--eet-global-router", type=int, default=0, choices=[0, 1], help="EET: use an upfront single global exit router predicting exit layer distribution")
parser.add_argument("--eet-freq-efficiency-alpha", type=float, default=0.0, help="EET: per-token frequency-scaled efficiency loss (0=uniform, >0=frequent tokens penalized more for late exits)")
parser.add_argument("--eet-diversity-lambda", type=float, default=0.0, help="EET: exit diversity pressure - penalizes uniform exit depth across tokens (0=disabled)")
parser.add_argument("--eet-ce-guided-lambda", type=float, default=1.0, help="EET: CE-guided routing loss weight (loss_variant='ce_guided')")
parser.add_argument("--eet-router-lr-mult", type=float, default=5.0, help="EET: Router LR multiplier relative to gate_lr (default 5.0). Higher = faster router learning.")
parser.add_argument("--eet-model-lr-mult", type=float, default=1.0, help="EET: Backbone LR multiplier (default 1.0). Scales all non-router LRs when EET is active.")
parser.add_argument("--eet-depth-weight-type", type=str, default="none", choices=["none", "linear", "ema", "sqrt"], help="EET: Token-wise CE loss weighting by exit depth")
parser.add_argument("--eet-depth-weight-max", type=float, default=2.5, help="EET: Maximum weighting factor for deep tokens in linear strategy")
parser.add_argument("--eet-use-override", type=int, default=0, choices=[0, 1], help="EET: 1 = enable stochastic depth override, 0 = disabled")
parser.add_argument("--eet-override-prob-start", type=float, default=0.5, help="EET: initial override probability during training")
parser.add_argument("--eet-override-prob-end", type=float, default=0.1, help="EET: minimum/terminal override probability during training")
parser.add_argument("--eet-reenter-final", type=int, default=0, choices=[0, 1], help="EET: force exited tokens to re-enter and be processed by the final layer (1/0)")
parser.add_argument("--eet-compute-skip", type=int, default=0, choices=[0, 1], help="EET: enable compute-level skipping of intermediate blocks (1/0)")
parser.add_argument("--eet-target-active-frac", type=float, default=0.125, help="EET: target active token fraction at the deepest routable layer")
parser.add_argument("--eet-capacity-schedule", type=str, default="bell", choices=["uniform", "linear", "geometric", "bell"], help="EET: capacity schedule for compute skip")
parser.add_argument("--eet-exit-fracs", type=str, default="", help="EET: comma-separated list of float exit fractions (sums to ~1.0) overriding schedule")
parser.add_argument("--eet-capacity-alignment-lambda", type=float, default=0.0, help="EET: weight for load-balancing/capacity alignment loss (0=disabled)")
parser.add_argument("--eet-router-task-grad", type=int, default=1, choices=[0, 1], help="EET: allow task loss gradients to propagate to router through continue weights (1/0)")
parser.add_argument("--eet-reinforce-interval", type=int, default=0, help="EET: two-pass REINFORCE every N steps (0=disabled)")
parser.add_argument("--eet-reinforce-lambda", type=float, default=0.1, help="EET: REINFORCE loss weight")
parser.add_argument("--eet-exit-adapter-rank", type=int, default=0, help="EET: per-exit low-rank adapter rank (0=disabled)")
parser.add_argument("--eet-router-after-block", type=int, default=0, help="EET: run global router after this block index (0=use raw embedding)")
parser.add_argument("--eet-ffn-skip", type=int, default=0, choices=[0, 1], help="EET A³D: skip FFN only, preserve attention at all layers")
parser.add_argument("--eet-ffn-target-frac", type=float, default=0.50, help="EET A³D: fraction of tokens that get FFN at each layer")
parser.add_argument("--eet-ffn-full-attn", type=int, default=1, choices=[0, 1], help="EET A³D: 1=attention on all T tokens, 0=gather for attention too")
parser.add_argument("--eet-depth-affine", type=int, default=0, choices=[0, 1], help="EET: apply learned (γ, β) depth-conditional affine alignment before LM head")
parser.add_argument("--eet-capacity-anneal-frac", type=float, default=0.0, help="EET: fraction of training steps to anneal target active frac from 0.5 to configured value")
parser.add_argument("--eet-learned-schedule", type=int, default=0, choices=[0, 1], help="EET: learn per-exit scheduling prior logits instead of fixed schedule")
parser.add_argument("--eet-departure-summary", type=int, default=0, choices=[0, 1], help="EET: inject mean state of exiting tokens into continuing active tokens")
parser.add_argument("--eet-route-consistency-lambda", type=float, default=0.0, help="EET: weight for EMA-based routing consistency loss")
parser.add_argument("--eet-dense-distill-interval", type=int, default=0, help="EET: concurrent dense distillation step interval (0=disabled)")
parser.add_argument("--eet-dense-distill-lambda", type=float, default=0.5, help="EET: concurrent dense distillation KL loss weight")
parser.add_argument("--eet-depth-lr-scale", type=int, default=0, choices=[0, 1], help="EET: per-layer LR scaling by inverse surviving fraction (Option A)")
parser.add_argument("--eet-depth-grad-scale", type=int, default=0, choices=[0, 1], help="EET: scale per-token CE by inverse active fraction at exit depth (Option B)")
parser.add_argument("--eet-detach-aux-from-backbone", type=int, default=0, choices=[0, 1], help="EET: detach aux losses (CE-guided, surprise) from backbone gradients")
parser.add_argument("--eet-detach-exit-from-backbone", type=int, default=0, choices=[0, 1], help="EET: detach exiting token representations from backbone — backbone only trains from final-layer tokens")

# Standard training parameters
parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient norm clip threshold (-1 to disable)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-tokens", type=int, default=-1, help="explicit number of tokens to train for (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.8, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--disable-mu-p", action="store_true", help="disable μP-style LR scaling")
parser.add_argument("--mu-p-scale-override", type=float, default=-1.0, help="force a specific mu-P scale")
parser.add_argument("--warmup-ratio", type=float, default=0.005, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = only at end, 0 = disable)")
parser.add_argument("--log-every", type=int, default=1, help="print step log to console every N steps")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = only at end, 0 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="enable/disable torch.compile")
parser.add_argument("--tokenizer-dir", type=str, default=None, help="explicit tokenizer directory (overrides default)")
parser.add_argument("--max-shards", type=int, default=-1, help="maximum number of dataset shards to use (-1 = all)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument("--early-stop-tokens", type=int, default=-1, help="terminate training after this many tokens without affecting the LR schedule (-1 = disabled)")
parser.add_argument("--step-loss-file", type=str, default="", help="optional JSONL file to write per-step training loss for external sweep plotting")
args = parser.parse_args()

if args.data_dir is not None:
    os.environ["NANOCHAT_DATA_DIR"] = args.data_dir
user_config = vars(args).copy()  # for logging

# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# nn.DataParallel optimization (multi-GPU without torchrun)
is_dp = args.parallel == "dp" and device_type == "cuda" and torch.cuda.device_count() > 1
if is_dp:
    ddp_world_size = torch.cuda.device_count()
    ddp_rank = 0
    ddp_local_rank = 0
    print0(f"✓ Using nn.DataParallel (detected {ddp_world_size} GPUs)")
else:
    if args.parallel == "dp":
        print0(f"i DataParallel requested but suppressed: device_type={device_type}, gpu_count={torch.cuda.device_count()}")

master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = True
wandb_run = DummyWandb()

# Flash Attention backend status
from nanochat.flash_attention import USE_FA4, USE_FA3, _BACKEND as _FA_BACKEND
if USE_FA4:
    print0(f"✓ Using Flash Attention 4 (Blackwell GPU detected) — fastest possible attention.")
elif USE_FA3:
    major, _ = torch.cuda.get_device_capability() if device_type == 'cuda' else (0, 0)
    hw = "Blackwell" if major >= 10 else "Hopper"
    print0(f"✓ Using Flash Attention 3 ({hw} GPU detected) — efficient and fast.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3/4 not available")
    print0("WARNING: Falling back to PyTorch SDPA — training will be less efficient.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
tokenizer = get_tokenizer(tokenizer_dir=args.tokenizer_dir)
token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    if getattr(args, 'model_dim', 0) > 0:
        base_model_dim = args.model_dim
    else:
        base_dim = depth * args.aspect_ratio
        base_model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    base_num_heads = base_model_dim // args.head_dim

    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=base_num_heads, n_kv_head=base_num_heads, n_embd=base_model_dim,
        window_pattern=args.window_pattern,
        # EET: Early Exit Transformer
        use_eet=bool(getattr(args, 'use_eet', 0)),
        eet_frozen_kv=bool(getattr(args, 'eet_frozen_kv', 1)),
        eet_reenter_final=bool(getattr(args, 'eet_reenter_final', 0)),
        eet_router_type=getattr(args, 'eet_router_type', 'mlp2'),
        eet_router_hidden=getattr(args, 'eet_router_hidden', 0),
        eet_freq_prior_alpha=getattr(args, 'eet_freq_prior_alpha', 0.0),
        eet_pos_prior_beta=getattr(args, 'eet_pos_prior_beta', 0.0),
        eet_domain_prior=bool(getattr(args, 'eet_domain_prior', 0)),
        eet_warmup_frac=getattr(args, 'eet_warmup_frac', 0.02),
        eet_explore_frac=getattr(args, 'eet_explore_frac', 0.15),
        eet_reconstruct_lambda=getattr(args, 'eet_reconstruct_lambda', 1.0),
        eet_efficiency_lambda_start=getattr(args, 'eet_efficiency_lambda_start', 0.01),
        eet_efficiency_lambda_end=getattr(args, 'eet_efficiency_lambda_end', 0.1),
        eet_translator_rank=getattr(args, 'eet_translator_rank', 0),
        eet_max_frozen_kv_frac=getattr(args, 'eet_max_frozen_kv_frac', 0.75),
        eet_exit_threshold=getattr(args, 'eet_exit_threshold', 0.5),
        eet_min_exit_layer=getattr(args, 'eet_min_exit_layer', 1),
        eet_loss_variant=getattr(args, 'eet_loss_variant', 'reconstruct'),
        eet_topk_vocab=int(getattr(args, 'eet_topk_vocab', 512)),
        eet_entropy_lambda=float(getattr(args, 'eet_entropy_lambda', 0.3)),
        eet_surprise_lambda=float(getattr(args, 'eet_surprise_lambda', 0.1)),
        eet_adv_lambda=float(getattr(args, 'eet_adv_lambda', 1.0)),
        eet_adv_entropy_lambda=float(getattr(args, 'eet_adv_entropy_lambda', 0.2)),
        eet_quality_lambda=float(getattr(args, 'eet_quality_lambda', 1.0)),
        eet_quality_entropy_bonus=float(getattr(args, 'eet_quality_entropy_bonus', 0.1)),
        eet_gumbel_temp_start=float(getattr(args, 'eet_gumbel_temp_start', 0.0)),
        eet_gumbel_temp_end=float(getattr(args, 'eet_gumbel_temp_end', 0.1)),
        eet_gumbel_hard=bool(getattr(args, 'eet_gumbel_hard', 1)),
        eet_commitment_beta=float(getattr(args, 'eet_commitment_beta', 0.1)),
        eet_global_router=bool(getattr(args, 'eet_global_router', 0)),
        eet_freq_efficiency_alpha=float(getattr(args, 'eet_freq_efficiency_alpha', 0.0)),
        eet_diversity_lambda=float(getattr(args, 'eet_diversity_lambda', 0.0)),
        eet_ce_guided_lambda=float(getattr(args, 'eet_ce_guided_lambda', 1.0)),
        eet_router_lr_mult=float(getattr(args, 'eet_router_lr_mult', 5.0)),
        eet_model_lr_mult=float(getattr(args, 'eet_model_lr_mult', 1.0)),
        eet_depth_weight_type=str(getattr(args, 'eet_depth_weight_type', 'none')),
        eet_depth_weight_max=float(getattr(args, 'eet_depth_weight_max', 2.5)),
        eet_use_override=int(getattr(args, 'eet_use_override', 0)),
        eet_override_prob_start=float(getattr(args, 'eet_override_prob_start', 0.5)),
        eet_override_prob_end=float(getattr(args, 'eet_override_prob_end', 0.1)),
        eet_compute_skip=bool(getattr(args, 'eet_compute_skip', 0)),
        eet_target_active_frac=float(getattr(args, 'eet_target_active_frac', 0.125)),
        eet_capacity_schedule=getattr(args, 'eet_capacity_schedule', 'bell'),
        eet_exit_fracs=[float(x.strip()) for x in getattr(args, 'eet_exit_fracs', '').split(',') if x.strip()] if getattr(args, 'eet_exit_fracs', '') else None,
        eet_capacity_alignment_lambda=float(getattr(args, 'eet_capacity_alignment_lambda', 0.0)),
        eet_router_task_grad=bool(getattr(args, 'eet_router_task_grad', 1)),
        eet_reinforce_interval=int(getattr(args, 'eet_reinforce_interval', 0)),
        eet_reinforce_lambda=float(getattr(args, 'eet_reinforce_lambda', 0.1)),
        eet_exit_adapter_rank=int(getattr(args, 'eet_exit_adapter_rank', 0)),
        eet_router_after_block=int(getattr(args, 'eet_router_after_block', 0)),
        eet_ffn_skip=bool(getattr(args, 'eet_ffn_skip', 0)),
        eet_ffn_target_frac=float(getattr(args, 'eet_ffn_target_frac', 0.50)),
        eet_ffn_full_attn=bool(getattr(args, 'eet_ffn_full_attn', 1)),
        eet_depth_affine=bool(getattr(args, 'eet_depth_affine', 0)),
        eet_capacity_anneal_frac=float(getattr(args, 'eet_capacity_anneal_frac', 0.0)),
        eet_learned_schedule=bool(getattr(args, 'eet_learned_schedule', 0)),
        eet_departure_summary=bool(getattr(args, 'eet_departure_summary', 0)),
        eet_route_consistency_lambda=float(getattr(args, 'eet_route_consistency_lambda', 0.0)),
        eet_dense_distill_interval=int(getattr(args, 'eet_dense_distill_interval', 0)),
        eet_dense_distill_lambda=float(getattr(args, 'eet_dense_distill_lambda', 0.5)),
        eet_depth_lr_scale=bool(int(getattr(args, 'eet_depth_lr_scale', 0))),
        eet_depth_grad_scale=bool(int(getattr(args, 'eet_depth_grad_scale', 0))),
        eet_detach_aux_from_backbone=bool(int(getattr(args, 'eet_detach_aux_from_backbone', 0))),
        eet_detach_exit_from_backbone=bool(int(getattr(args, 'eet_detach_exit_from_backbone', 0))),
    )
    config._tokenizer_dir = getattr(args, 'tokenizer_dir', None)

    with torch.device("meta"):
        if config.use_eet:
            from nanochat.eet import EarlyExitGPT
            model_meta = EarlyExitGPT(config)
        else:
            model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)
model.init_weights()

# Checkpoints config
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
if args.checkpoints_dir:
    checkpoints_root = os.path.abspath(args.checkpoints_dir)
else:
    checkpoints_root = os.path.join(get_base_dir(), "base_checkpoints")

checkpoint_dir = os.path.abspath(os.path.join(checkpoints_root, output_dirname))
print0(f"Checkpoints directory: {checkpoint_dir}")

if args.step_loss_file and master_process:
    step_loss_dir = os.path.dirname(os.path.abspath(args.step_loss_file))
    if step_loss_dir:
        os.makedirs(step_loss_dir, exist_ok=True)
    with open(args.step_loss_file, "w", encoding="utf-8"):
        pass

eet_ever_routed = False
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data
    eet_ever_routed = meta_data.get("eet_ever_routed", False)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        args.fp8 = False
    else:
        major, minor = torch.cuda.get_device_capability()
        if major < 8 or (major == 8 and minor < 9):
            print0(f"Warning: FP8 training requires compute capability >= 8.9 (e.g. H100, L4, 4090), but detected {major}.{minor}. Disabling FP8.")
            args.fp8 = False

if args.fp8:
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn

    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
    num_skipped = num_linear - num_fp8
    print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

@contextmanager
def disable_fp8(model):
    import torch.nn as nn
    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# Disable requires_grad for EET parameters during Phase 1 warmup
if model_config.use_eet:
    for param in model.eet_routers.parameters():
        param.requires_grad = False
    for param in model.eet_translators.parameters():
        param.requires_grad = False

orig_model = model
model = wrap_model(model, parallel_type=args.parallel, compile=args.compile, device=device)

# Scaling laws
param_counts = orig_model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token, num_active_flops_per_token, num_active_params = orig_model.estimate_flops()
print0(f"Estimated FLOPs per token (total):  {num_flops_per_token:e}")
print0(f"Estimated FLOPs per token (active): {num_active_flops_per_token:e}")
print0(f"Estimated active params:            {num_active_params:,}")

def get_scaling_params(m):
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params

num_scaling_params = get_scaling_params(orig_model)
if args.target_tokens > 0:
    target_tokens = args.target_tokens
else:
    active_scaling_params = num_scaling_params
    target_tokens = int(args.target_param_data_ratio * active_scaling_params)

d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# Initialize Optimizer
optimizer = orig_model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    disable_mu_p=args.disable_mu_p,
    mu_p_scale_override=args.mu_p_scale_override,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

# Initialize DataLoaders
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size * (ddp_world_size if is_dp else 1),
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    data_dir=args.data_dir,
    max_shards=args.max_shards,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer,
    args.device_batch_size * (ddp_world_size if is_dp else 1),
    args.max_seq_len,
    split="val",
    device=device,
    data_dir=args.data_dir,
    max_shards=args.max_shards,
)
x, y, dataloader_state_dict = next(train_loader)

assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
else:
    num_iterations = target_tokens // total_batch_size

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")

def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# Training loop state
mfu = 0.0
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    min_val_loss = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
    last_periodic_ckpt_step = -1
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    min_val_loss = loop_state.get("min_val_loss", float("inf"))
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    last_periodic_ckpt_step = step

effective_device_batch_size = args.device_batch_size * (ddp_world_size if is_dp else 1)
tokens_per_fwdbwd = effective_device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * (1 if is_dp else ddp_world_size)
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
EMA_BETA = 0.9

# Pre-compilation warmup
if not resuming and device_type == "cuda":
    print0("Running pre-compilation warmup (1 dummy forward+backward to init lazy allocations)...")
    torch.cuda.reset_peak_memory_stats()
    _wx = torch.zeros_like(x)
    _wy = torch.zeros_like(y)
    if model_config.use_eet:
        _wloss = model(_wx, _wy, eet_do_route=False)
    else:
        _wloss = model(_wx, _wy)
    if is_dp:
        _wloss = _wloss.mean()
    (_wloss / grad_accum_steps).backward()
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy, _wloss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Go!
while True:
    last_step = step == num_iterations

    if args.early_stop_tokens > 0 and step * total_batch_size >= args.early_stop_tokens:
        print0(f"[early stop] Reached {step * total_batch_size:,} tokens. Initiating final eval/save.")
        last_step = True

    flops_so_far = num_flops_per_token * total_batch_size * step

    do_eval = (args.eval_every > 0 and (last_step or step % args.eval_every == 0)) or (last_step and args.eval_every == -1)
    if do_eval:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            eval_kwargs = {}
            if args.use_eet:
                if eet_ever_routed:
                    eval_kwargs['eet_do_route'] = True
                    eval_kwargs['eet_phase'] = 3
                else:
                    eval_kwargs['eet_do_route'] = False
                    eval_kwargs['eet_phase'] = 1
            val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes, **eval_kwargs)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f} | val_loss: {val_loss:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        if val_loss < min_val_loss:
            min_val_loss = val_loss

        # periodic core eval (omitted to save time in standard logs, but can run at end)
        if args.core_metric_every > 0 and (step % args.core_metric_every == 0 or last_step):
            model.eval()
            print0(f"Step {step:05d} | Evaluating CORE metric...")
            results = evaluate_core(orig_model, device=device, max_examples_per_task=args.core_metric_max_per_task, verbose=False)
            print0(f"Step {step:05d} | CORE metric estimate: {results['core_metric']:.4f}")

        # periodic model sampling
        if args.sample_every > 0 and step % args.sample_every == 0 and master_process:
            model.eval()
            print0(f"Step {step:05d} | Sampling from model...")
            with torch.no_grad():
                # Take first 8 tokens of the first val batch as prompt
                _prompt = x[0, :8].tolist()
                _prompt_str = tokenizer.decode(_prompt)
                print0(f"Prompt: {_prompt_str!r}")
                # Simple greedy generation
                _gen = _prompt
                _x_gen = torch.tensor([_gen], device=device)
                for _ in range(32):
                    with disable_fp8(orig_model):
                        _logits = orig_model(_x_gen)
                    _next_tok = _logits[0, -1].argmax().item()
                    _gen.append(_next_tok)
                    _x_gen = torch.tensor([_gen], device=device)
                _gen_str = tokenizer.decode(_gen)
                print0(f"Generated: {_gen_str!r}")

        model.train()

    # save checkpoints
    do_save = (args.save_every > 0 and step > 0 and step % args.save_every == 0) or last_step
    if do_save and master_process:
        checkpoint_metadata = {
            "step": step,
            "val_bpb": val_bpb,
            "dataloader_state_dict": dataloader_state_dict,
            "eet_ever_routed": eet_ever_routed,
            "loop_state": {
                "min_val_bpb": min_val_bpb,
                "min_val_loss": min_val_loss,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
            }
        }
        save_checkpoint(checkpoint_dir, step, orig_model.state_dict(), optimizer.state_dict(), checkpoint_metadata)

    if last_step:
        debiased_at_step = smooth_train_loss / (1 - EMA_BETA**max(step, 1))
        print0(f"step {step:05d}/{num_iterations:05d} (final) | loss: {debiased_at_step:.6f} | early_stop: {int(args.early_stop_tokens > 0)}")
        break

    # -------------------------------------------------------------------------
    # single training step
    synchronize()
    t0 = time.time()
    
    if model_config.use_eet:
        from nanochat.eet import EETPhaseScheduler
        _eet_sched = EETPhaseScheduler(
            num_iterations,
            warmup_frac=model_config.eet_warmup_frac,
            explore_frac=model_config.eet_explore_frac,
            reconstruct_lambda=model_config.eet_reconstruct_lambda,
            efficiency_lambda_start=model_config.eet_efficiency_lambda_start,
            efficiency_lambda_end=model_config.eet_efficiency_lambda_end,
        )
        if _eet_sched.explore_end < num_iterations and _eet_sched.explore_end > _eet_sched.warmup_end and step == _eet_sched.explore_end:
            print0(f"\n[EET DIAGNOSTIC] Step {step:05d}: Running router structure check before entering Phase 3...")
            orig_model.train()
            with torch.no_grad():
                _ = orig_model(x, eet_do_route=True, eet_phase=2, eet_lambda_r=0.0, eet_lambda_e=0.0)
            
            if hasattr(orig_model, '_last_exit_probs'):
                try:
                    check_router_learned(orig_model._last_exit_probs, x, orig_model)
                except Exception as e:
                    print0(f"[EET DIAGNOSTIC] Warning: correlation check failed with error: {e}")
            else:
                print0("[EET DIAGNOSTIC] Warning: exit probabilities not captured during forward pass.")
            orig_model.train()

    for micro_step in range(grad_accum_steps):
        if model_config.use_eet:
            from nanochat.eet import EETPhaseScheduler
            _eet_phase_info = EETPhaseScheduler(
                num_iterations,
                warmup_frac=model_config.eet_warmup_frac,
                explore_frac=model_config.eet_explore_frac,
                reconstruct_lambda=model_config.eet_reconstruct_lambda,
                efficiency_lambda_start=model_config.eet_efficiency_lambda_start,
                efficiency_lambda_end=model_config.eet_efficiency_lambda_end,
            ).get_phase(step)

            eet_do_route = _eet_phase_info['do_route']
            eet_phase = _eet_phase_info['phase']
            if eet_do_route:
                eet_ever_routed = True

            if eet_phase == 1:
                loss = model(x, y)
            else:
                if (hasattr(orig_model, 'eet_current_phase') and 
                    orig_model.eet_current_phase == 1):
                    orig_model.eet_current_phase = eet_phase
                    orig_model.eet_phase_tracker[0] = eet_phase
                    print0(f"[EET] Transitioning from Phase 1 (Dense Warmup) to Phase {eet_phase} (Routing active).")

                for param in orig_model.eet_routers.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                for param in orig_model.eet_translators.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True

                _anneal_frac = getattr(model_config, 'eet_capacity_anneal_frac', 0.0)
                if _anneal_frac > 0.0:
                    progress = step / max(num_iterations, 1)
                    if progress < _anneal_frac:
                        t = progress / _anneal_frac
                        t_discrete = round(t * 10) / 10.0
                        base_target_frac = getattr(args, 'eet_target_active_frac', 0.125)
                        model_config.eet_target_active_frac = 0.5 + t_discrete * (base_target_frac - 0.5)
                    else:
                        model_config.eet_target_active_frac = getattr(args, 'eet_target_active_frac', 0.125)

                eet_gumbel_temp_tensor = torch.tensor(1.0, device=x.device, dtype=torch.float32)
                if model_config.eet_gumbel_temp_start > 0.0:
                    t_start = model_config.eet_gumbel_temp_start
                    t_end = model_config.eet_gumbel_temp_end
                    progress = min(max(step / max(num_iterations, 1), 0.0), 1.0)
                    temp_val = t_start * ((t_end / t_start) ** progress)
                    eet_gumbel_temp_tensor = torch.tensor(temp_val, device=x.device, dtype=torch.float32)

                eet_step_tensor = torch.tensor(step, device=x.device, dtype=torch.float32)
                eet_total_steps_tensor = torch.tensor(num_iterations, device=x.device, dtype=torch.float32)

                _reinforce_interval = getattr(model_config, 'eet_reinforce_interval', 0)
                if (_reinforce_interval > 0 and eet_phase == 3 and
                    step % _reinforce_interval == 0 and eet_do_route):
                    dense_ce = orig_model._compute_dense_per_token_ce(x, y)
                    orig_model._reinforce_dense_ce = dense_ce

                _distill_interval = getattr(model_config, 'eet_dense_distill_interval', 0)
                eet_dense_x = None
                if (_distill_interval > 0 and eet_phase in {2, 3} and
                    step % _distill_interval == 0 and eet_do_route):
                    with torch.no_grad():
                        eet_dense_x = orig_model._compute_dense_logits(x)

                loss = model(x, y,
                             eet_do_route=eet_do_route,
                             eet_phase=eet_phase,
                             eet_lambda_r=torch.tensor(_eet_phase_info['lambda_r'], device=x.device, dtype=torch.float32),
                             eet_lambda_e=torch.tensor(_eet_phase_info['lambda_e'], device=x.device, dtype=torch.float32),
                             eet_gumbel_temp=eet_gumbel_temp_tensor,
                             eet_step=eet_step_tensor,
                             eet_total_steps=eet_total_steps_tensor,
                             eet_dense_x=eet_dense_x)
        else:
            loss = model(x, y)
            
        if model_config.use_eet and (last_step or step == num_iterations - 1) and micro_step == grad_accum_steps - 1:
            if hasattr(orig_model, '_last_exit_probs') and orig_model._last_exit_probs is not None:
                orig_model._final_train_exit_probs = orig_model._last_exit_probs.detach().cpu().clone()
                orig_model._final_train_tokens = x.detach().cpu().clone()
            if hasattr(orig_model, '_last_enforced_capacities'):
                orig_model._final_enforced_capacities = list(orig_model._last_enforced_capacities)
                orig_model._final_active_counts = list(orig_model._last_active_counts)
                orig_model._final_T = orig_model._last_T
                
        if is_dp:
            loss = loss.mean()
        train_loss = loss.detach()

        # EET: ce_guided depth classification re-run global router
        if (model_config.use_eet and model_config.eet_compute_skip and
            getattr(model_config, 'eet_loss_variant', '') == 'ce_guided' and
            hasattr(orig_model, '_last_x0_for_ce') and
            hasattr(orig_model, '_last_per_token_ce')):
            _x0 = orig_model._last_x0_for_ce
            _ptce = orig_model._last_per_token_ce
            _tgt = y

            _global_router = orig_model.eet_routers[0]
            _freq_bias = getattr(orig_model, '_freq_bias', None)
            _pos_bias = getattr(orig_model, '_pos_bias', None)
            _rl = _global_router(
                _x0,
                freq_bias=_freq_bias,
                pos_bias=_pos_bias,
                freq_alpha=model_config.eet_freq_prior_alpha,
                pos_beta=model_config.eet_pos_prior_beta
            )

            _ce_loss = orig_model._compute_ce_guided_loss(_rl, _ptce)
            loss = loss + model_config.eet_ce_guided_lambda * _ce_loss

        if scaler is not None:
            scaler.scale(loss / grad_accum_steps).backward()
        else:
            (loss / grad_accum_steps).backward()

        x, y, dataloader_state_dict = next(train_loader)

    # clip gradients
    if scaler is not None:
        scaler.unscale_(optimizer)
    
    _eet_grad_pending = model_config.use_eet and hasattr(orig_model, 'eet_routers') and eet_phase >= 2
    if _eet_grad_pending:
        # custom gradient scaling / tracking for EET parameters if desired
        _gi = {}
        # Calculate router/translator grad norms
        _r_gn = 0.0
        for p in orig_model.eet_routers.parameters():
            if p.grad is not None:
                _r_gn += p.grad.float().norm().item() ** 2
        _gi['router_total_grad_norm'] = _r_gn ** 0.5

        _t_gn = 0.0
        for p in orig_model.eet_translators.parameters():
            if p.grad is not None:
                _t_gn += p.grad.float().norm().item() ** 2
        _gi['translator_total_grad_norm'] = _t_gn ** 0.5

        for i, r in enumerate(orig_model.eet_routers):
            _l_gn = 0.0
            for p in r.parameters():
                if p.grad is not None:
                    _l_gn += p.grad.float().norm().item() ** 2
            _gi[f'router_{i}_total_grad_norm'] = _l_gn ** 0.5
        
        _n_with_grad = sum(1 for p in orig_model.eet_routers.parameters() if p.grad is not None)
        _gi['n_router_params_with_grad'] = _n_with_grad
        orig_model._eet_grad_info = _gi

    clip_val = 10.0 if (model_config.use_eet and args.max_grad_norm == 1.0) else args.max_grad_norm
    if clip_val > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    else:
        grad_norm = 0.0

    # step the optimizer
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    synchronize()
    
    dt = time.time() - t0
    total_training_time += dt

    # decay learning rate
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # update Muon momentum and weight decay schedulers
    for group in optimizer.param_groups:
        if group.get("is_muon", False):
            group["momentum"] = get_muon_momentum(step)
            group["weight_decay"] = get_weight_decay(step)

    # debiased smooth train loss
    smooth_train_loss = EMA_BETA * smooth_train_loss + (1 - EMA_BETA) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - EMA_BETA**(step + 1))
    
    # print step log
    if args.log_every > 0 and step % args.log_every == 0:
        tok_per_sec = total_batch_size / dt
        epoch = step * total_batch_size / total_tokens
        if gpu_peak_flops != float('inf'):
            mfu = (num_flops_per_token * total_batch_size) / (gpu_peak_flops * dt * ddp_world_size) * 100
        else:
            mfu = 0.0
        print0(f"step {step:05d}/{num_iterations:05d} | loss: {debiased_smooth_loss:.6f} | lr: {optimizer.param_groups[0]['lr']:.3e} | dt: {dt*1000:.1f}ms | tok/sec: {tok_per_sec:,.0f} | mfu: {mfu:.2f}% | epoch: {epoch:.3f}")

        # EET: Early Exit diagnostics
        if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics'):
            _eet_diag = orig_model._eet_diagnostics
            _eet_phase = _eet_diag.get('phase', 0)
            _eet_active = _eet_diag['active_frac'].item() if hasattr(_eet_diag.get('active_frac', 0), 'item') else _eet_diag.get('active_frac', 1.0)
            _eet_exit = _eet_diag['total_exit_frac'].item() if hasattr(_eet_diag.get('total_exit_frac', 0), 'item') else _eet_diag.get('total_exit_frac', 0.0)
            _active_counts = getattr(orig_model, '_last_active_counts', None)
            _counts_str = f" | tokens={_active_counts}" if _active_counts else ""
            _a3d_str = " [A³D]" if _eet_diag.get('a3d', False) else ""
            print0(f"  eet{_a3d_str} | phase={_eet_phase} | active={_eet_active:.3f} | exit_frac={_eet_exit:.3f}{_counts_str}")
        
        # EET: Router gradient diagnostics
        if model_config.use_eet and hasattr(orig_model, '_eet_grad_info'):
            _gi = orig_model._eet_grad_info
            _r_gnorm = _gi.get('router_total_grad_norm', 0.0)
            _t_gnorm = _gi.get('translator_total_grad_norm', 0.0)
            _n_with_grad = _gi.get('n_router_params_with_grad', 0)
            per_layer_parts = []
            for layer_idx in range(len(orig_model.eet_routers)):
                ln = _gi.get(f'router_{layer_idx}_total_grad_norm', 0.0)
                per_layer_parts.append(f'L{layer_idx}={ln:.3e}')
            per_layer_str = ' '.join(per_layer_parts) if per_layer_parts else 'none'
            print0(f"  eet_grad | ∇router={_r_gnorm:.3e} ∇trans={_t_gnorm:.3e} params_with_grad={_n_with_grad} | {per_layer_str}")
            if master_process and _gi.get('n_router_params_with_grad', 0) > 0:
                _gi['eet_phase'] = _eet_diag.get('phase', 0) if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics') else 0
                _gi['active_frac'] = _eet_active if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics') else 1.0
                eet_grad_log_path = os.path.join(checkpoint_dir, "eet_grad_log.jsonl")
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(eet_grad_log_path, 'a') as _ef:
                    _ef.write(json.dumps(_gi) + '\n')

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
    print0(f"Minimum validation loss (nats/byte): {min_val_loss:.6f}")

# EET final diagnostics
if model_config.use_eet:
    print0("\n================================================================================")
    print0("[EET FINAL ROUTER DIAGNOSTICS]")
    print0("================================================================================")

    if hasattr(orig_model, '_final_enforced_capacities') and hasattr(orig_model, '_final_active_counts'):
        capacities = orig_model._final_enforced_capacities
        active_counts = orig_model._final_active_counts
        T = orig_model._final_T
        n_blocks = len(active_counts)
        n_rl = len(capacities)

        B = 1
        if hasattr(orig_model, '_final_train_exit_probs') and orig_model._final_train_exit_probs is not None:
            B = orig_model._final_train_exit_probs.size(0)

        total_tokens = B * T
        print0(f"Enforced physical exit distribution ({total_tokens:,} tokens, B={B}, T={T}, {n_blocks} blocks, {n_rl} routing slots):")
        for slot in range(n_rl):
            block_idx = slot + 1
            k_before = active_counts[block_idx]
            if block_idx + 1 < n_blocks:
                k_after = active_counts[block_idx + 1]
            else:
                k_after = active_counts[-1]
            exited = k_before - k_after
            pct = (exited / T) * 100
            label = f'exit_{slot}'
            print0(f"  {label:12s}: {pct:6.2f}% ({exited * B:,} tokens) [capacity: {capacities[slot] * B:,}]")
        final_active = active_counts[-1]
        final_pct = (final_active / T) * 100
        print0(f"  {'final_layer':12s}: {final_pct:6.2f}% ({final_active * B:,} tokens active)")

        enforced_active = sum(active_counts) / (n_blocks * T)
        enforced_exit = 1.0 - (active_counts[-1] / T)
        print0(f"\nEnforced active fraction: {enforced_active:.3f} (exit_frac={enforced_exit:.3f})")
    else:
        print0("(No enforced capacity data available — compute_skip may not have been enabled)")

    if hasattr(orig_model, '_final_train_exit_probs') and orig_model._final_train_exit_probs is not None:
        exit_probs = orig_model._final_train_exit_probs.detach().cpu().float()
        n_exits = exit_probs.size(-1)

        argmax_exits = exit_probs.argmax(dim=-1).numpy().ravel()
        total_tokens_evaluated = len(argmax_exits)

        print0(f"\nRouter soft probability diagnostics ({total_tokens_evaluated:,} tokens):")
        print0("Router preferred exit (argmax of soft probs):")
        for slot in range(n_exits):
            count = (argmax_exits == slot).sum()
            pct = (count / max(total_tokens_evaluated, 1)) * 100
            label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
            print0(f"  {label:12s}: {pct:6.2f}% ({count:,} tokens)")

        print0("\nSoft exit probability stats per slot:")
        for slot in range(n_exits):
            slot_probs = exit_probs[:, :, slot].numpy().ravel()
            label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
            print0(f"  {label:12s}: mean={slot_probs.mean():.6f} std={slot_probs.std():.6f} min={slot_probs.min():.6f} max={slot_probs.max():.6f}")

        layer_indices = torch.arange(n_exits).float()
        expected_exit = (exit_probs * layer_indices).sum(dim=-1).numpy().ravel()
        print0(f"\nMean expected exit layer (soft): {expected_exit.mean():.4f}")
        print0(f"Std expected exit layer (soft):  {expected_exit.std():.6f}")

        if expected_exit.std() < 0.01:
            print0(f"\n[EET WARNING] ⚠ ROUTER COLLAPSE: std={expected_exit.std():.6f} — router is near-constant across all tokens.")
        elif expected_exit.std() < 0.1:
            print0(f"\n[EET WARNING] ⚠ LOW DIFFERENTIATION: std={expected_exit.std():.6f} — router is barely differentiating tokens.")
        else:
            print0("\n[EET INFO] ✓ Router has learned token-dependent exit behavior!")
    else:
        print0("\n[EET FINAL DIAGNOSTIC] Warning: No router exit probabilities captured during training steps.")
    print0("================================================================================\n")

# Log to report
from nanochat.report import get_report
section_name = "Base model training"
if args.model_tag:
    section_name += f" ({args.model_tag})"
get_report().log(section=section_name, data=[
    user_config,
    {
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    {
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

compute_cleanup()
