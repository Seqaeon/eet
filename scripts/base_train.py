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

# import wandb
import torch
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
# Research branches
parser.add_argument("--use-moe", action="store_true", help="enable research MoE embedding branch")
parser.add_argument("--use-perm", action="store_true", help="use permutation MoE branch (only with --use-moe)")
parser.add_argument("--moe-num-experts", type=int, default=8, help="number of experts for research MoE branch")
parser.add_argument("--moe-router-dim", type=int, default=64, help="router dim for MoE embedding branch")
parser.add_argument("--moe-embed-dim", type=int, default=64, help="embedding dim for MoE branch (must match n_embd)")
parser.add_argument("--use-remix-linear", action="store_true", help="enable remixed linear blocks")
parser.add_argument("--remix-context-dim", type=int, default=64, help="context dim for remixed linear control")
parser.add_argument("--remix-basis-size", type=int, default=64, help="basis size for remixed linear")
parser.add_argument("--use-pos-embed", action="store_true", help="add learned absolute positional embeddings on top of token/research embeddings")
parser.add_argument("--moe-use-abs-pos-embed", type=int, default=0, choices=[0, 1], help="use learned absolute positional embeddings inside permutation MoE embeddings (1/0)")
parser.add_argument("--remix-use-basis-gate", type=int, default=1, choices=[0, 1], help="enable basis gating in remixed linear (1/0)")
parser.add_argument("--remix-use-output-gate", type=int, default=1, choices=[0, 1], help="enable output gating in remixed linear (1/0)")
parser.add_argument("--remix-use-context", type=int, default=1, choices=[0, 1], help="enable context modulation in remixed linear (1/0)")
parser.add_argument("--remix-basis-gate-mode", type=str, default="mlp",
                    choices=["mlp", "linear", "centered", "attn", "random", "none", "lowrank"],
                    help="Gate architecture: mlp=2-layer, linear=single layer (zero-init), centered=1+tanh passthrough init, attn=bilinear, random=linear with random init (no zero-init), none=no basis gate, lowrank=low rank centered output gate symmetry")

parser.add_argument("--p22-n-templates", type=int, default=1, help="22: number of template_mixing matrices (1=standard, K>1=MoE routing)")
parser.add_argument("--p22-template-routing-learned", type=int, default=0, choices=[0, 1], help="22: learned template routing (0=frozen, 1=learned)")
parser.add_argument("--p22-template-topk", type=int, default=0, help="22: hard top-k for legacy template bank (0=soft over all K, N=top-N sparse routing)")
parser.add_argument("--p22-attn-moe-route", type=str, default="none", choices=["none", "sequence", "token"], help="22: MoE routing for attention Q/K/V/Proj ('none'=off, 'sequence'=per-seq, 'token'=per-tok)")
# Phase 23: Tiny-Experts RemixedLinear + Standard MoE baseline
parser.add_argument("--p23-tiny-expert", type=int, default=0, choices=[0, 1], help="23: enable Tiny Experts mode in RemixedLinear (each expert_dim=basis_size//topk for compute parity)")
parser.add_argument("--p23-n-experts", type=int, default=64, help="23: total experts in the Tiny Expert bank (K_total)")
parser.add_argument("--p23-topk", type=int, default=16, help="23: active experts per forward pass (expert_dim=basis_size//topk); 0=soft all-expert routing")
parser.add_argument("--p23-learned-route", type=int, default=0, choices=[0, 1], help="23: learned routing projection in Tiny Expert (0=frozen, 1=learned)")
parser.add_argument("--p23-std-moe-experts", type=int, default=0, help="23: enable StandardMoE_MLP with K full-size experts (0=off)")
parser.add_argument("--p23-std-moe-topk", type=int, default=-1, help="23: top-k active experts for StandardMoE_MLP (-1=optimal sparsity via E^(1-c), 0=all/soft, N=fixed)")
parser.add_argument("--p23-std-moe-aux-weight", type=float, default=0.01, help="23: load-balance auxiliary loss weight for StandardMoE_MLP")
parser.add_argument("--p23-lokr", type=int, default=0, choices=[0, 1], help="23: enable LoKR mode in RemixedLinear")
# Phase 26: Streamlined Full-Rank — OutputGatedLinear (single W + low-rank output gate)
parser.add_argument("--p26-output-gated-linear", type=int, default=0, choices=[0, 1], help="26: replace RemixedLinear with single-W + low-rank output gate (no W_b/W_m factorization)")
# Phase 28: FLOPs-efficient template routing
parser.add_argument("--p28-shared-basis", type=int, default=0, choices=[0, 1], help="28C: share a single W_b projection across all attn Q/K/V/O RL layers per block (saves 3/4 of attn basis FLOPs)")
parser.add_argument("--p28-chunk-routing-size", type=int, default=0, help="28D: amortize template routing over N-token chunks (0=per-token, 64/256=chunk-level)")
parser.add_argument("--p28-global-template-bank", type=str, default="none", choices=["none", "ffn", "all"], help="28E/F: cross-layer global template bank mode ('none'=off, 'ffn'=FFN only, 'all'=FFN+attn)")
parser.add_argument("--p28-attn-proj-templates", type=int, default=0, help="28C2: override n_templates for attn c_proj per-layer (0=use default from remixed_linear_kwargs)")
parser.add_argument("--p28-attn-qk-templates",   type=int, default=0, help="28C3: override n_templates for attn c_q and c_k per-layer (0=use default)")
parser.add_argument("--target-active-params",     type=int, default=0, choices=[0, 1], help="use active (topk/chunk-adjusted) params instead of total params when computing target_tokens")
parser.add_argument("--remix-basis-gate-rank", type=int, default=8, help="rank for lowrank basis gate mode (basis_gate_mode=lowrank)")
parser.add_argument("--p23-lokr-rank", type=int, default=4, help="23: low-rank bottleneck for each LoKR expert")
parser.add_argument("--p23-use-shared-block-router", type=int, default=0, choices=[0, 1], help="23: block-level single pass router for all RemixedLinear inner experts")
parser.add_argument("--p23-linear-moe-experts", type=int, default=0, help="23: enable weight-space LinearMoE with K experts (0=off)")
parser.add_argument("--p23-linear-moe-topk", type=int, default=0, help="23: top-k selected experts in LinearMoE (0=soft all-expert blend)")
parser.add_argument("--p23-quantile-route", type=int, default=0, choices=[0, 1, 2], help="23: 1=EMA quantile routing, 2=Causal Expert Cross-Attention")
parser.add_argument("--remix-shared-context-gates", type=int, default=0, choices=[0, 1], help="23: batch all 6 per-RL context gate computations into 3 block-level matmuls (~6x fewer gate kernel launches)")
parser.add_argument("--remix-use-dual-gate", type=int, default=0, choices=[0, 1], help="25: use DualGateLinear instead of RemixedLinear (single dense W + dual D-dim gate, no basis compression)")
parser.add_argument("--gate-stats-every", type=int, default=50, help="log gate activation/gradient stats every N steps to gate_stats.log (0=off); only active when --use-remix-linear")
parser.add_argument("--remix-basis-scale-factor", type=int, default=4, help="basis compression ratio: factor=4 → C//4 (default), factor=1 → full rank C. Depth-adaptive via min(in,out)//factor")
parser.add_argument("--remix-output-gate-rank", type=int, default=16, help="rank of the low-rank output gate in RemixedLinear (default 16)")
parser.add_argument("--remix-gate-lr-scale", type=float, default=0.3, help="LR multiplier for gate params relative to structural matrix LR (default 0.3; lower = slower gate learning)")
# Phase 30: LayerNorm ablation flags
parser.add_argument("--remix-disable-ln-basis", type=int, default=0, choices=[0, 1], help="30B: disable intermediate LayerNorm in RemixedLinear basis projection (0=keep LN, 1=remove LN)")
parser.add_argument("--dense-intermediate-ln", type=int, default=0, choices=[0, 1], help="30A: add intermediate LayerNorm to dense linear projections for controlled ablation (0=off, 1=on)")
# ── MST: Modular Sub-Transformer ──
parser.add_argument("--use-mst", type=int, default=0, choices=[0, 1], help="MST: enable Modular Sub-Transformer mode")
parser.add_argument("--mst-n-subs", type=int, default=8, help="MST: number of sub-transformers N per layer")
parser.add_argument("--mst-sub-dim", type=int, default=64, help="MST: dimension d per sub-transformer")
parser.add_argument("--mst-head-dim", type=int, default=0,
                    help="MST: attention head_dim (0=auto d//n_head; e.g. 64 expands QKV to n_head*64 then projects back to d)")
parser.add_argument("--mst-input-mode", type=str, default="fixed_slice",
                    choices=["fixed_slice", "learned_proj", "rotated_slice", "per_sub_embed", "stem"],
                    help="MST Axis 1: input distribution mode")
parser.add_argument("--mst-rotated-slice-learned", type=int, default=0, choices=[0, 1],
                    help="MST I-C: learn rotation matrix (0=frozen orthogonal, 1=learned)")
parser.add_argument("--mst-routing-mode", type=str, default="soft_weighted",
                    choices=["soft_weighted", "topk_hard", "sequence_path"],
                    help="MST Axis 2: routing/combination mode")
parser.add_argument("--mst-routing-topk", type=int, default=4, help="MST R-B: k for top-k hard routing")
parser.add_argument("--mst-routing-aux-weight", type=float, default=0.01, help="MST: load balance aux loss weight")
parser.add_argument("--mst-diversity-weight", type=float, default=0.0, help="MST: cosine diversity penalty weight (0=off, e.g. 0.01)")
parser.add_argument("--mst-ffn-mode", type=str, default="standard", choices=["standard", "no_downproj", "linear"],
                    help="MST Axis 3: FFN mode (standard=d->4d->d, no_downproj=d->4d)")
parser.add_argument("--mst-transition-mode", type=str, default="parallel",
                    choices=["parallel", "aggregate_distribute", "cross_attend", "concat_proj", "free_for_all", "micro_attention", "micro_attention_shared_kv"],
                    help="MST Axis 4: layer-to-layer transition mode")
parser.add_argument("--mst-final-mode", type=str, default="aggregate_proj",
                    choices=["aggregate_proj", "weighted_logits", "concat_proj"],
                    help="MST Axis 5: final layer output mode")
parser.add_argument("--mst-final-topk", type=int, default=-1,
                    help="MST Axis 5: topk for final head (-1=same as routing_topk, 0=all subs)")
parser.add_argument("--mst-ffn-shared-up", type=int, default=0, choices=[0, 1],
                    help="MST: share FFN up-projection across all subs")
parser.add_argument("--mst-ffn-inner-dim", type=int, default=0,
                    help="MST: inner dim for shared FFN (0=4*d)")
parser.add_argument("--mst-sub-dropout", type=float, default=0.0,
                    help="MST: dropout entire sub outputs during training")
parser.add_argument("--mst-transition-every", type=int, default=1,
                    help="MST: apply transition every N layers (1=every, 2=alternate)")
parser.add_argument("--mst-ffa-temperature", type=float, default=1.0,
                    help="MST: temperature for FFA softmax routing")
parser.add_argument("--mst-global-residual", type=int, default=0, choices=[0, 1],
                    help="MST: D-dim global residual stream across layers")
parser.add_argument("--mst-hybrid-dense", type=int, default=0, choices=[0, 1],
                    help="MST: alternate dense (D) and MST (N×d) layers")
parser.add_argument("--mst-cross-sub-kv", type=int, default=0, choices=[0, 1],
                    help="MST: share K,V projections across all subs")
parser.add_argument("--mst-sub-aux-weight", type=float, default=0.0,
                    help="MST: per-sub auxiliary prediction loss weight (H3)")
parser.add_argument("--mst-progressive-merge", type=int, default=0, choices=[0, 1],
                    help="MST: progressive sub-merging pyramid (N1)")
parser.add_argument("--mst-multi-scale-windows", type=int, default=0, choices=[0, 1],
                    help="MST: per-sub window sizes for multi-scale attention (W1)")
parser.add_argument("--mst-delta-residual", type=int, default=0, choices=[0, 1],
                    help="MST: delta residual mode — subs produce corrections to full-D stream (DR1)")
parser.add_argument("--mst-sub-layers", type=int, default=1,
                    help="MST: layers per sub-transformer (SL1, default 1)")
# MST Stage 7: Scaling improvements
parser.add_argument("--mst-grad-equalize", type=int, default=0, choices=[0, 1],
                    help="MST: enable per-sub gradient equalization (1A)")
parser.add_argument("--mst-block-diagonal-muon", type=int, default=0, choices=[0, 1],
                    help="MST: enable block-diagonal Muon Newton-Schulz (1B)")
parser.add_argument("--mst-transition-width-mult", type=float, default=1.0,
                    help="MST: width multiplier for transition bottleneck (1C)")
parser.add_argument("--mst-sub-lr-scale", type=float, default=1.0,
                    help="MST: learning rate multiplier for sub parameters")
parser.add_argument("--mst-shared-expert", type=int, default=0, choices=[0, 1],
                    help="MST: enable DeepSeek-style shared expert (2A)")
parser.add_argument("--mst-router-entropy-weight", type=float, default=0.0,
                    help="MST: weight for router entropy regularization (2C)")
parser.add_argument("--mst-shared-kv-attn", type=int, default=0, choices=[0, 1],
                    help="MST: share K,V projections across sub-transformers (3A)")
parser.add_argument("--mst-contrastive-diversity-weight", type=float, default=0.0,
                    help="MST: weight for contrastive representation diversity loss (3B)")
# MST Stage 8: Transition expressivity
parser.add_argument("--mst-transition-nonlinear", type=int, default=0, choices=[0, 1],
                    help="MST: add SiLU activation at AggDist bottleneck")
parser.add_argument("--mst-transition-gated", type=int, default=0, choices=[0, 1],
                    help="MST: input-dependent routing via concat→gate (replaces mean-based)")
parser.add_argument("--mst-transition-mlp", type=int, default=0, choices=[0, 1],
                    help="MST: replace AggDist with concat→MLP→split (nonlinear cross-sub mixing)")
# MST Stage 9: Cross-sub expressivity
parser.add_argument("--mst-cross-sub-gate", type=int, default=0,
                    help="MST: cross-sub FFN gating rank (0=off, e.g. 32)")
parser.add_argument("--mst-hyper-connect", type=int, default=0, choices=[0, 1],
                    help="MST: hyper-connected sub residuals via EMA lookback (0=off, 1=on)")
parser.add_argument("--mst-cross-kv-inject", type=int, default=0, choices=[0, 1],
                    help="MST: cross-sub KV injection attention (0=off, 1=on)")
# MST Stage 10: Structural transition improvements
parser.add_argument("--mst-slice-transition", type=int, default=0,
                    help="MST: SliceMoE per-slice routing (0=off, S=num slices e.g. 4)")
parser.add_argument("--mst-lookback-layers", type=int, default=0,
                    help="MST: DenseFormer lookback depth (0=off, K=num past layers e.g. 2)")
parser.add_argument("--mst-bilinear-transition", type=int, default=0, choices=[0, 1],
                    help="MST: bilinear aggregate for 2nd-order cross-sub interaction (0=off, 1=on)")
# MST Stage 11: Attention bottleneck + structural improvements
parser.add_argument("--mst-cross-sub-qmod", type=int, default=0,
                    help="MST: low-rank cross-sub query modulation rank (0=off, e.g. 16)")
parser.add_argument("--mst-feature-cycle", type=int, default=0, choices=[0, 1],
                    help="MST: sub-feature cycling at transition (0=off, 1=on)")
parser.add_argument("--mst-mean-transition", type=int, default=0, choices=[0, 1],
                    help="MST: parameter-free mean-add transition replacing AggDist (0=off, 1=on)")
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
parser.add_argument("--p24-use-sliced-weight", type=int, default=0, choices=[0, 1], help="24: enable SlicedWeightLinear (LinearMoE2-style)")
parser.add_argument("--p24-sliced-weight-reduction-scale", type=int, default=8, help="24: big_dim = in_features * reduction_scale")
parser.add_argument("--p24-sliced-weight-min-select", type=int, default=128, help="24: minimum selected columns from weight bank")
parser.add_argument("--p24-sliced-weight-scope", type=str, default="per_token", choices=["per_token", "per_block", "global"], help="24: routing scope for sliced weight")
parser.add_argument("--p24-sliced-weight-balance-coeff", type=float, default=0.01, help="24: aux loss coeff for sliced router usage balance")
parser.add_argument("--p24-quantile-route", type=int, default=0, choices=[0, 1, 2], help="24: reserved routing mode selector")
parser.add_argument("--p24-use-folded-mod", type=int, default=0, choices=[0, 1], help="24: enable FoldedModulationLinear (LinearMoE3-style)")
parser.add_argument("--p24-folded-mod-reduction-scale", type=int, default=8, help="24: fold R consecutive dims by summation")
parser.add_argument("--p24-folded-mod-min-dim", type=int, default=128, help="24: floor on folded_dim")
parser.add_argument("--p24-folded-mod-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"], help="24: gate sharing scope for folded modulation")
parser.add_argument("--p24-folded-mod-gate-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"], help="24: gate activation for folded modulation")
parser.add_argument("--p24-use-sequence-gated-linear", type=int, default=0, choices=[0, 1], help="24: enable SequenceGatedLinear (dense + sequence gate)")
parser.add_argument("--p24-sequence-gated-scope", type=str, default="per_layer", choices=["per_layer", "per_block", "global"], help="24: gate sharing scope for dense sequence gating")
parser.add_argument("--p24-sequence-gated-act", type=str, default="tanh_centered", choices=["sigmoid", "tanh_centered"], help="24: gate activation for dense sequence gating")
# CCL block modulation (only active when --use-remix-linear is set)

parser.add_argument("--cclblock-modulation", type=str, default="weight",
                    choices=["weight", "normalization", "householder", "spectral", "ocd", "lie", "polynomial", "grassmann", "decoupled", "tucker", "svs", "vq", "dcu", "fsi", "aesp", "ckr", "ckr_ffn", "com", "giad", "psg", "splitstream", "lokr", "pgr", "cil", "prb", "arg", "kfl"],
                    help="CCL block strategy")
parser.add_argument("--cclblock-orth-lambda", type=float, default=0.0,
                    help="OCD overlap penalty weight (0 disables)")
parser.add_argument("--cclblock-context-stream", type=str, default="local", 
                    choices=["local", "shifted", "ema", "selective", "multiscale", "ssm", "boundary", "chunk", "predictive_chunk", "evidence_ssm", "dacs", "prefix", "warmup_ema", "dacs_ema", "decay_prefix"],
                    help="Context stream: 'local', 'shifted', 'ema', 'selective', 'multiscale', 'ssm', 'boundary', 'chunk', 'predictive_chunk', 'evidence_ssm', 'dacs', 'prefix', 'warmup_ema', 'dacs_ema', 'decay_prefix'")
parser.add_argument("--cclblock-ema-factor", type=float, default=0.99,
                    help="EMA factor for the legacy EMAContextStream")
parser.add_argument("--cclblock-stale-ctx-lag", type=int, default=0,
                    help="Design C stale context lag (0=disabled)")
# Novel ablation designs
parser.add_argument("--cclblock-sparse-gate-k", type=int, default=0,
                    help="Design 3: sparse top-k basis gate (0=soft sigmoid, N=activate top-N basis functions)")
parser.add_argument("--cclblock-gate-temperature", type=float, default=1.0,
                    help="Design 6: basis gate temperature (<1=sharper, >1=softer, 1.0=standard sigmoid)")
parser.add_argument("--cclblock-context-bank-size", type=int, default=0,
                    help="Design 4: context prototype bank size (0=disabled, e.g. 16=16 learned prototypes)")
parser.add_argument("--cclblock-per-head-ctx", type=int, default=0, choices=[0, 1],
                    help="Design 7: separate ctx projections for attn vs ffn (0=off, 1=on)")
parser.add_argument("--cclblock-context-source", type=str, default="norm_x",
                    choices=["norm_x", "attn_heads", "attn_geometry"],
                    help="Context source for FFN gate/router ('norm_x', 'attn_heads', 'attn_geometry')")
# Phase 8: Boundary-Gated / Chunk Context / Auxiliary Objective
parser.add_argument("--cclblock-chunk-size", type=int, default=0,
                    help="Design 9: hard chunk pooling stride in tokens (0=off, e.g. 64)")
parser.add_argument("--cclblock-aux-objective", type=str, default="none",
                    choices=["none", "boundary", "entropy"],
                    help="Design 10: aux context objective ('none'=off, 'boundary'=boundary BCE, 'entropy'=difficulty MSE)")
parser.add_argument("--cclblock-aux-lambda", type=float, default=0.1,
                    help="Design 10: weight of auxiliary context loss (default 0.1)")
parser.add_argument("--cclblock-boundary-token-id", type=int, default=198,
                    help="Design 10: token ID for boundary detection (default 198=newline in many tokenizers)")
# Phase 9: RAL & FiLM
parser.add_argument("--use-ral", type=int, default=0, choices=[0, 1], help="Proposal A: Use ResidualAdaptiveLinear instead of RemixedLinear")
parser.add_argument("--ral-rank", type=int, default=32, help="Proposal A: Rank for the RAL context delta")
parser.add_argument("--cclblock-film-gate", type=int, default=0, choices=[0, 1], help="Proposal C: Use FiLM affine basis gate in RemixedLinear")
parser.add_argument("--cclblock-attn-shadow-dim", type=int, default=0, help="Dual-V shadow routing width (0=off)")
parser.add_argument("--cclblock-dynamic-ratio", type=float, default=0.25, help="Paradigm 1 (decoupled): fraction of channels routed to dynamic path")
parser.add_argument("--cclblock-gate-rank", type=int, default=8, help="Paradigm 1 (decoupled): low-rank context gate rank")
parser.add_argument("--cclblock-num-regimes", type=int, default=8, help="Paradigm 2 (evidence_ssm): number of latent regimes K")
parser.add_argument("--cclblock-regime-temperature", type=float, default=1.0, help="Paradigm 2 (evidence_ssm): softmax temperature over regimes")
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
# Phase 12: FSI / AESP / CKR
parser.add_argument("--cclblock-fsi-rotations", type=int, default=8, help="FSI: number of frozen orthogonal rotations")
parser.add_argument("--cclblock-fsi-selector-dim", type=int, default=64, help="FSI: frozen routing projection dim")
parser.add_argument("--cclblock-aesp-strata", type=int, default=4, help="AESP: number of entropy strata")
parser.add_argument("--cclblock-aesp-delta-rank", type=int, default=4, help="AESP: rank of per-stratum low-rank deltas")
parser.add_argument("--cclblock-ckr-branches", type=int, default=4, help="CKR: number of parallel dense branches")
parser.add_argument("--cclblock-ckr-kernel-size", type=int, default=64, help="CKR: causal conv1d kernel size")
# Phase 13: CKR enhancements
parser.add_argument("--cclblock-ckr-pos-channels", type=int, default=1, help="CKR: multi-channel position signal (1=original, 3=multi-scale)")
parser.add_argument("--cclblock-ckr-dual-optim", type=int, default=0, choices=[0, 1], help="CKR: route gate params to dedicated conservative AdamW")
parser.add_argument("--cclblock-ckr-content-bias", type=float, default=0.0, help="CKR: frozen content hash bias scale (0=pure position)")
# Phase 14: Gradient-isolated content conditioning
parser.add_argument("--cclblock-giad-rank", type=int, default=32, help="GIAD: low-rank bottleneck dimension")
parser.add_argument("--cclblock-psg-kernel-size", type=int, default=64, help="PSG: causal conv kernel size")
parser.add_argument("--cclblock-ss-dynamic-ratio", type=float, default=0.25, help="SplitStream: dynamic channel fraction")
parser.add_argument("--cclblock-ss-branches", type=int, default=2, help="SplitStream: CKR branches on dynamic path")
parser.add_argument("--cclblock-ss-kernel-size", type=int, default=64, help="SplitStream: causal conv kernel size")
# Phase 15: LoKR + diagnostics
parser.add_argument("--cclblock-lokr-branches", type=int, default=8, help="LoKR: number of low-rank perturbation branches")
parser.add_argument("--cclblock-lokr-rank", type=int, default=16, help="LoKR: rank of each perturbation")
# Phase 16: CKR-Anneal / COM
parser.add_argument("--cclblock-ckr-temp-start", type=float, default=2.0, help="CKR-Anneal: initial softmax temperature")
parser.add_argument("--cclblock-ckr-temp-end", type=float, default=0.3, help="CKR-Anneal: final softmax temperature")
parser.add_argument("--cclblock-com-kernel-size", type=int, default=32, help="COM: causal output mixer kernel size")
# Phase 17: CKR enhancements + new architectures
parser.add_argument("--cclblock-ckr-ortho-init", type=int, default=0, choices=[0, 1], help="17D: orthogonal branch init")
parser.add_argument("--cclblock-ckr-branch-dropout", type=float, default=0.0, help="17E: branch dropout probability")
parser.add_argument("--cclblock-ckr-diversity-lambda", type=float, default=0.0, help="17H: branch diversity loss weight")
parser.add_argument("--cclblock-pgr-kernel-size", type=int, default=64, help="17C: PGR causal conv kernel size")
parser.add_argument("--cclblock-cil-kernel-size", type=int, default=64, help="17I: CIL causal conv kernel size")
parser.add_argument("--cclblock-prb-kernel-size", type=int, default=64, help="17J: PRB causal conv kernel size")
parser.add_argument("--modulation-diagnostics", type=int, default=0, choices=[0, 1], help="enable modulation layer diagnostics logging")
# Phase 18: Beyond CKR — orthogonal improvements
parser.add_argument("--p18-layer-drop", type=float, default=0.0, help="18E: stochastic depth drop probability (0=off)")
parser.add_argument("--p18-dynamic-activation", type=int, default=0, choices=[0, 1], help="18I: learned activation mix (ReLU²+GELU+SiLU)")
parser.add_argument("--p18-mixture-norm", type=int, default=0, choices=[0, 1], help="18H: learned RMSNorm+LayerNorm mixture")
parser.add_argument("--p18-aux-sim-lambda", type=float, default=0.0, help="18G: layer similarity penalty weight (0=off)")
parser.add_argument("--p18-gradient-penalty", type=float, default=0.0, help="18B: gradient penalty weight for Lipschitz regularization (0=off)")
parser.add_argument("--p18-per-channel-scale", type=int, default=0, choices=[0, 1], help="18F: learnable per-channel output scale")
# Phase 19: Zero-overhead indirect modulation
parser.add_argument("--p19-residual-gate", type=int, default=0, choices=[0, 1], help="19A: per-layer learned scalar on block output (0/1)")
parser.add_argument("--p19-head-importance", type=int, default=0, choices=[0, 1], help="19B: per-head learned scalar on attn output (0/1)")
parser.add_argument("--p19-residual-mix-groups", type=int, default=0, help="19C: grouped 1x1 conv between blocks (0=off, N=group_size)")
parser.add_argument("--p19-attn-logit-bias", type=int, default=0, choices=[0, 1], help="19D: per-head learned QK temperature (0/1)")
parser.add_argument("--p19-residual-decay", type=int, default=0, choices=[0, 1], help="19E: learned depth-dependent x0 decay (0/1)")
parser.add_argument("--p19-grad-equilibrium", type=float, default=0.0, help="19F: gradient equilibrium regularization lambda (0=off)")
parser.add_argument("--p19-spectral-reparam", type=int, default=0, choices=[0, 1, 2], help="19G: spectral reparameterization (0=off, 1=c_proj, 2=c_fc+c_proj)")
parser.add_argument("--p19-weight-anticollapse", type=float, default=0.0, help="19H: weight anti-collapse penalty lambda (0=off)")
parser.add_argument("--p19-ve-bias", type=int, default=0, choices=[0, 1], help="19I: add learnable bias to VE gate (0/1)")
parser.add_argument("--p19-weight-noise", type=float, default=0.0, help="19J: training-time weight perturbation epsilon (0=off)")
# Phase 20: Context-conditioned dynamic weight computation
parser.add_argument("--p20-hrcs-scale", type=int, default=0, help="20A: Hash-routed column selection (0=off, scale=D_stored/D_active)")
parser.add_argument("--p20-lswr-scale", type=int, default=0, help="20B: LSH weight routing (0=off, scale factor)")
parser.add_argument("--p20-lswr-planes", type=int, default=8, help="20B: number of LSH hash planes")
parser.add_argument("--p20-lrcfb-branches", type=int, default=0, help="20C: Content-routed branches (0=off, K=branches)")
parser.add_argument("--p20-lrcfb-narrow", type=int, default=0, choices=[0, 1], help="20C: narrow branches (0=full-size, 1=H//K param parity)")
parser.add_argument("--p20-lrcfb-learned", type=int, default=0, choices=[0, 1], help="20C: learned routing (0=frozen, 1=learnable)")
parser.add_argument("--p20-lrcfb-topk", type=int, default=0, help="20C: top-k sparse routing (0=soft/all)")
parser.add_argument("--p20-dgcr-branches", type=int, default=0, help="20D: Detached-gradient content-routed branches (0=off, K=branches)")
parser.add_argument("--p20-dgcr-aux-weight", type=float, default=0.01, help="20D: auxiliary routing loss weight")
parser.add_argument("--p20-mone-experts", type=int, default=0, help="20F: Mixture of Narrow Experts (0=off, K=num experts)")
parser.add_argument("--p20-mone-topk", type=int, default=0, help="20F: top-k expert routing (0=compute all, K=top-k sparse)")
parser.add_argument("--p20-mone-narrow", type=int, default=1, choices=[0, 1], help="20F: narrow experts (1=4D/K, 0=full 4D each)")
parser.add_argument("--p20-mone-frozen", type=int, default=0, choices=[0, 1], help="20F: frozen routing (0=learned, 1=frozen random proj)")
parser.add_argument("--p20-ncea-branches", type=int, default=0, help="20H: Noise-contrastive expert assignment (0=off, K=branches)")
parser.add_argument("--p20-ncea-eps", type=float, default=0.1, help="20H: perturbation magnitude")
parser.add_argument("--p20-adwi", type=int, default=0, choices=[0, 1], help="20I: Attention-derived weight interpolation (0=off, 1=on)")
# Phase 20 — Phase 2 proposals (require pre-trained checkpoint)
parser.add_argument("--p20-pwu-branches", type=int, default=0, help="20E: Progressive weight unfreezing (0=off, K=branches)")
parser.add_argument("--p20-pwu-phase", type=int, default=1, choices=[1, 2, 3], help="20E: training phase (1=pretrain, 2=router only, 3=joint)")
parser.add_argument("--p20-fsvd-gate", type=int, default=0, choices=[0, 1], help="20G: Frozen-SVD σ gating (0=off, 1=on)")
parser.add_argument("--p20-wbfc-clusters", type=int, default=0, help="20J: Weight bank frozen clustering (0=off, K=clusters)")
parser.add_argument("--p20-wbfc-active", type=int, default=0, help="20J: active clusters per token (0=auto K//4)")
# Phase 21: Pervasive Expert Routing
parser.add_argument("--p21-per-experts", type=int, default=0, help="21: MoELinear experts per layer (0=off, K=experts)")
parser.add_argument("--p21-per-topk", type=int, default=0, help="21: top-k routing (0=soft/all)")
parser.add_argument("--p21-per-learned", type=int, default=0, choices=[0, 1], help="21: learned routing (0=frozen, 1=learnable)")
parser.add_argument("--p21-per-attn", type=int, default=0, choices=[0, 1], help="21: also replace attention Q/K/V/O (0=MLP only, 1=all)")
# Fix 1A: per-layer context updaters
parser.add_argument("--use-layer-context", type=int, default=1, choices=[0, 1], help="per-layer context deltas for remix_linear: 1=enable (Fix 1A), 0=static base context")
parser.add_argument("--router-context-window", type=int, default=-1, help="sliding window size for GlobalContextManager (-1 for full)")
# Fix 1B: basis scaling
parser.add_argument("--scale-basis-size", type=int, default=1, choices=[0, 1], help="auto-scale RemixedLinear basis_size to max(basis_size, in_features//4) (Fix 1B)")
# Fix 1D: PermutationMoE expert mode
parser.add_argument("--perm-expert-mode", type=str, default="low_rank", choices=["full", "low_rank", "factored"], help="PermutationMoE expert mode: 'full' (original D×D), 'low_rank', or 'factored' (Fix 1D)")
parser.add_argument("--perm-rank", type=int, default=16, help="rank divisor for 'low_rank' mode or block size for 'factored' mode (Fix 1D)")
# Fix 4C: gradient clipping
parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient norm clip threshold (-1 to disable, Fix 4C)")
# Fix 1H: PermutationMoE temperature scheduling
parser.add_argument("--perm-temp-start", type=float, default=5.0, help="initial PermutationMoE temperature (decays to 1.0 over first 50%% of training, Fix 1H)")
parser.add_argument("--research-onecycle", type=int, default=1, choices=[0, 1], help="for research runs: 1=use OneCycle LR schedule, 0=fallback to base warmup/flat/warmdown")
parser.add_argument("--use-onecycle", type=int, default=None, choices=[0, 1], help="alias for --research-onecycle")
parser.add_argument("--research-warmup-ratio", type=float, default=0.05, help="research-only warmup ratio/pct_start for OneCycle")
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
parser.add_argument("--disable-mu-p", action="store_true",
                    help="disable μP-style LR scaling (model_dim/768)^-0.5 for AdamW params. "
                         "Use when sweeping absolute LRs directly for research models.")
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
if args.use_onecycle is not None:
    args.research_onecycle = args.use_onecycle
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# nn.DataParallel optimization (multi-GPU without torchrun)
is_dp = args.parallel == "dp" and device_type == "cuda" and torch.cuda.device_count() > 1
if is_dp:
    # When using DP, we act like a single world but with larger device_batch_size
    ddp_world_size = torch.cuda.device_count()
    ddp_rank = 0
    ddp_local_rank = 0
    print0(f"✓ Using nn.DataParallel (detected {ddp_world_size} GPUs)")
else:
    if args.parallel == "dp":
        print0(f"i DataParallel requested but suppressed: device_type={device_type}, gpu_count={torch.cuda.device_count()}")

master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
if is_dp:
    print0(f"DataParallel enabled: world_size={ddp_world_size}")

# wandb logging init
use_dummy_wandb = True # args.run == "dummy" or not master_process
wandb_run = DummyWandb() # if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

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
    if (HAS_FA4 or HAS_FA3) and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3/4 only support bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        major, _ = torch.cuda.get_device_capability() if device_type == 'cuda' else (0, 0)
        if major >= 10:
            print0("WARNING: Blackwell GPU detected but FA4 (flash-attn-4) and FA3 (kernels) not found.")
            print0("WARNING: Install FA3:  pip install kernels  (or pip install flash-attn-4 for FA4)")
        elif major >= 9:
            print0("WARNING: Hopper GPU detected but FA3 (kernels package) not found.")
            print0("WARNING: Install FA3:  pip install kernels")
        else:
            print0("WARNING: Flash Attention 3/4 not available (requires Hopper sm90+ or Blackwell sm100+)")
        print0("WARNING: Falling back to PyTorch SDPA — training will be less efficient.")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer(tokenizer_dir=args.tokenizer_dir)
token_bytes = get_token_bytes(device=device, tokenizer_dir=args.tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    if getattr(args, 'model_dim', 0) > 0:
        base_model_dim = args.model_dim
    else:
        base_dim = depth * args.aspect_ratio
        base_model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    base_num_heads = base_model_dim // args.head_dim
    if args.use_moe or args.use_remix_linear:
        model_dim = args.moe_embed_dim
        # Keep research branches compatible with repo attention constraints while
        # preferring a head count close to the base model's count.
        def _choose_research_heads(embed_dim: int, preferred_heads: int) -> int:
            pow2 = [1 << i for i in range(0, 12)]  # up to 2048 heads, way above practical usage
            valid_pow2 = [h for h in pow2 if h <= embed_dim and embed_dim % h == 0 and (embed_dim // h) % 8 == 0]
            if valid_pow2:
                return min(valid_pow2, key=lambda h: (abs(h - preferred_heads), -h))
            # Fallback: any divisor that keeps integer head_dim.
            valid_any = [h for h in range(1, embed_dim + 1) if embed_dim % h == 0]
            return min(valid_any, key=lambda h: abs(h - preferred_heads)) if valid_any else 1

        num_heads = _choose_research_heads(model_dim, base_num_heads)
        assert model_dim % num_heads == 0, f"moe_embed_dim must be divisible by n_head ({num_heads}), got {model_dim}"
    else:
        model_dim = base_model_dim
        num_heads = base_num_heads
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        use_moe=args.use_moe,
        use_perm=args.use_perm,
        moe_num_experts=args.moe_num_experts,
        moe_router_dim=args.moe_router_dim,
        moe_embed_dim=args.moe_embed_dim,
        use_remix_linear=args.use_remix_linear,
        remix_context_dim=args.remix_context_dim,
        remix_basis_size=args.remix_basis_size,
        use_pos_embed=args.use_pos_embed,
        moe_use_abs_pos_embed=bool(args.moe_use_abs_pos_embed),
        remixed_linear_kwargs=dict(
            use_basis_gate=bool(args.remix_use_basis_gate),
            use_output_gate=bool(args.remix_use_output_gate),
            use_context=bool(args.remix_use_context),
            basis_gate_mode=getattr(args, 'remix_basis_gate_mode', 'mlp'),
            sparse_gate_k=getattr(args, 'cclblock_sparse_gate_k', 0),
            gate_temperature=getattr(args, 'cclblock_gate_temperature', 1.0),
            basis_scale_factor=getattr(args, 'remix_basis_scale_factor', 4),
            # n_templates = K_total (total experts in the bank):
            # When p23_tiny_expert=1, p23_n_experts sets K_total.
            # Otherwise falls back to p22_n_templates (legacy template bank).
            n_templates=(
                getattr(args, 'p23_n_experts', 64)
                if getattr(args, 'p23_tiny_expert', 0)
                else getattr(args, 'p22_n_templates', 1)
            ),
            template_routing_learned=bool(
                getattr(args, 'p23_learned_route', 0)
                if getattr(args, 'p23_tiny_expert', 0) or getattr(args, 'p23_lokr', 0)
                else getattr(args, 'p22_template_routing_learned', 0)
            ),
            template_topk=getattr(args, 'p22_template_topk', 0),
            tiny_expert=bool(getattr(args, 'p23_tiny_expert', 0)),
            tiny_expert_topk=getattr(args, 'p23_topk', 16),
            lokr_expert=bool(getattr(args, 'p23_lokr', 0)),
            lokr_n_experts=getattr(args, 'p23_n_experts', 64),
            lokr_topk=getattr(args, 'p23_topk', 16),
            lokr_rank=getattr(args, 'p23_lokr_rank', 4),
            lokr_learned=bool(getattr(args, 'p23_learned_route', 0)),
            basis_gate_rank=getattr(args, 'remix_basis_gate_rank', 8),
            # Phase 30: LayerNorm ablation
            disable_ln_basis=bool(getattr(args, 'remix_disable_ln_basis', 0)),
        ),

        # Fix 1A
        use_layer_context=bool(getattr(args, 'use_layer_context', 1)),
        # Fix 1B
        scale_basis_size=bool(getattr(args, 'scale_basis_size', 1)),
        # Fix 1D
        perm_expert_mode=getattr(args, 'perm_expert_mode', 'low_rank'),
        perm_rank=getattr(args, 'perm_rank', 16),
        router_context_window=getattr(args, 'router_context_window', -1),
        # CCL block redesign
        cclblock_modulation=getattr(args, 'cclblock_modulation', 'weight'),
        cclblock_orth_lambda=getattr(args, 'cclblock_orth_lambda', 0.0),
        cclblock_context_stream=getattr(args, 'cclblock_context_stream', 'local'),
        cclblock_ema_factor=getattr(args, 'cclblock_ema_factor', 0.99),
        cclblock_stale_ctx_lag=getattr(args, 'cclblock_stale_ctx_lag', 0),
        # Novel ablation designs
        cclblock_sparse_gate_k=getattr(args, 'cclblock_sparse_gate_k', 0),
        cclblock_gate_temperature=getattr(args, 'cclblock_gate_temperature', 1.0),
        cclblock_context_bank_size=getattr(args, 'cclblock_context_bank_size', 0),
        cclblock_per_head_ctx=bool(getattr(args, 'cclblock_per_head_ctx', 0)),
        cclblock_context_source=getattr(args, 'cclblock_context_source', 'norm_x'),
        # Phase 8
        cclblock_chunk_size=getattr(args, 'cclblock_chunk_size', 0),
        cclblock_aux_objective=getattr(args, 'cclblock_aux_objective', 'none'),
        cclblock_aux_lambda=getattr(args, 'cclblock_aux_lambda', 0.1),
        cclblock_boundary_token_id=getattr(args, 'cclblock_boundary_token_id', 198),
        use_ral=bool(getattr(args, 'use_ral', 0)),
        ral_rank=getattr(args, 'ral_rank', 32),
        cclblock_film_gate=bool(getattr(args, 'cclblock_film_gate', 0)),
        cclblock_attn_shadow_dim=getattr(args, 'cclblock_attn_shadow_dim', 0),
        cclblock_dynamic_ratio=getattr(args, 'cclblock_dynamic_ratio', 0.25),
        cclblock_gate_rank=getattr(args, 'cclblock_gate_rank', 8),
        cclblock_num_regimes=getattr(args, 'cclblock_num_regimes', 8),
        cclblock_regime_temperature=getattr(args, 'cclblock_regime_temperature', 1.0),
        cclblock_poly_order=getattr(args, 'cclblock_poly_order', 2),
        cclblock_lie_generators=getattr(args, 'cclblock_lie_generators', 4),
        cclblock_grassmann_bank_size=getattr(args, 'cclblock_grassmann_bank_size', 4),
        cclblock_tucker_rank=getattr(args, 'cclblock_tucker_rank', 32),
        cclblock_tucker_modes=getattr(args, 'cclblock_tucker_modes', 8),
        cclblock_svs_rank=getattr(args, 'cclblock_svs_rank', 64),
        cclblock_svs_eps=getattr(args, 'cclblock_svs_eps', 0.1),
        cclblock_vq_codes=getattr(args, 'cclblock_vq_codes', 8),
        cclblock_vq_temperature=getattr(args, 'cclblock_vq_temperature', 1.0),
        cclblock_dcu_warmup_steps=getattr(args, 'cclblock_dcu_warmup_steps', 0),
        # Phase 12: FSI/AESP/CKR
        cclblock_fsi_rotations=getattr(args, 'cclblock_fsi_rotations', 8),
        cclblock_fsi_selector_dim=getattr(args, 'cclblock_fsi_selector_dim', 64),
        cclblock_aesp_strata=getattr(args, 'cclblock_aesp_strata', 4),
        cclblock_aesp_delta_rank=getattr(args, 'cclblock_aesp_delta_rank', 4),
        cclblock_ckr_branches=getattr(args, 'cclblock_ckr_branches', 4),
        cclblock_ckr_kernel_size=getattr(args, 'cclblock_ckr_kernel_size', 64),
        # Phase 13: CKR enhancements
        cclblock_ckr_pos_channels=getattr(args, 'cclblock_ckr_pos_channels', 1),
        cclblock_ckr_dual_optim=getattr(args, 'cclblock_ckr_dual_optim', 0),
        cclblock_ckr_content_bias=getattr(args, 'cclblock_ckr_content_bias', 0.0),
        # Phase 14: Gradient-isolated content conditioning
        cclblock_giad_rank=getattr(args, 'cclblock_giad_rank', 32),
        cclblock_psg_kernel_size=getattr(args, 'cclblock_psg_kernel_size', 64),
        cclblock_ss_dynamic_ratio=getattr(args, 'cclblock_ss_dynamic_ratio', 0.25),
        cclblock_ss_branches=getattr(args, 'cclblock_ss_branches', 2),
        cclblock_ss_kernel_size=getattr(args, 'cclblock_ss_kernel_size', 64),
        # Phase 15: LoKR
        cclblock_lokr_branches=getattr(args, 'cclblock_lokr_branches', 8),
        cclblock_lokr_rank=getattr(args, 'cclblock_lokr_rank', 16),
        # Phase 18: Beyond CKR
        p18_layer_drop=getattr(args, 'p18_layer_drop', 0.0),
        p18_dynamic_activation=getattr(args, 'p18_dynamic_activation', 0),
        p18_mixture_norm=getattr(args, 'p18_mixture_norm', 0),
        p18_aux_sim_lambda=getattr(args, 'p18_aux_sim_lambda', 0.0),
        p18_gradient_penalty=getattr(args, 'p18_gradient_penalty', 0.0),
        p18_per_channel_scale=getattr(args, 'p18_per_channel_scale', 0),
        # Phase 19: Zero-overhead indirect modulation
        p19_residual_gate=getattr(args, 'p19_residual_gate', 0),
        p19_head_importance=getattr(args, 'p19_head_importance', 0),
        p19_residual_mix_groups=getattr(args, 'p19_residual_mix_groups', 0),
        p19_attn_logit_bias=getattr(args, 'p19_attn_logit_bias', 0),
        p19_residual_decay=getattr(args, 'p19_residual_decay', 0),
        p19_grad_equilibrium=getattr(args, 'p19_grad_equilibrium', 0.0),
        p19_spectral_reparam=getattr(args, 'p19_spectral_reparam', 0),
        p19_weight_anticollapse=getattr(args, 'p19_weight_anticollapse', 0.0),
        p19_ve_bias=getattr(args, 'p19_ve_bias', 0),
        p19_weight_noise=getattr(args, 'p19_weight_noise', 0.0),
        # Phase 20: Context-conditioned dynamic weight computation
        p20_hrcs_scale=getattr(args, 'p20_hrcs_scale', 0),
        p20_lswr_scale=getattr(args, 'p20_lswr_scale', 0),
        p20_lswr_planes=getattr(args, 'p20_lswr_planes', 8),
        p20_lrcfb_branches=getattr(args, 'p20_lrcfb_branches', 0),
        p20_lrcfb_narrow=getattr(args, 'p20_lrcfb_narrow', 0),
        p20_lrcfb_learned=getattr(args, 'p20_lrcfb_learned', 0),
        p20_lrcfb_topk=getattr(args, 'p20_lrcfb_topk', 0),
        p20_dgcr_branches=getattr(args, 'p20_dgcr_branches', 0),
        p20_dgcr_aux_weight=getattr(args, 'p20_dgcr_aux_weight', 0.01),
        p20_mone_experts=getattr(args, 'p20_mone_experts', 0),
        p20_mone_topk=getattr(args, 'p20_mone_topk', 0),
        p20_mone_narrow=getattr(args, 'p20_mone_narrow', 1),
        p20_mone_frozen=getattr(args, 'p20_mone_frozen', 0),
        p20_ncea_branches=getattr(args, 'p20_ncea_branches', 0),
        p20_ncea_eps=getattr(args, 'p20_ncea_eps', 0.1),
        p20_adwi=getattr(args, 'p20_adwi', 0),
        # Phase 2 proposals
        p20_pwu_branches=getattr(args, 'p20_pwu_branches', 0),
        p20_pwu_phase=getattr(args, 'p20_pwu_phase', 1),
        p20_fsvd_gate=getattr(args, 'p20_fsvd_gate', 0),
        p20_wbfc_clusters=getattr(args, 'p20_wbfc_clusters', 0),
        p20_wbfc_active=getattr(args, 'p20_wbfc_active', 0),
        # Phase 21
        p21_per_experts=getattr(args, 'p21_per_experts', 0),
        p21_per_topk=getattr(args, 'p21_per_topk', 0),
        p21_per_learned=getattr(args, 'p21_per_learned', 0),
        p21_per_attn=getattr(args, 'p21_per_attn', 0),
        # Phase 22
        p22_attn_moe_route=getattr(args, 'p22_attn_moe_route', 'none'),
        # Phase 23: Tiny Expert RemixedLinear + Standard MoE baseline
        p23_tiny_expert=getattr(args, 'p23_tiny_expert', 0),
        p23_n_experts=getattr(args, 'p23_n_experts', 64),
        p23_topk=getattr(args, 'p23_topk', 16),
        p23_learned_route=getattr(args, 'p23_learned_route', 0),
        p23_std_moe_experts=getattr(args, 'p23_std_moe_experts', 0),
        p23_std_moe_topk=getattr(args, 'p23_std_moe_topk', 1),
        p23_std_moe_aux_weight=getattr(args, 'p23_std_moe_aux_weight', 0.01),
        p23_lokr=getattr(args, 'p23_lokr', 0),
        p23_lokr_rank=getattr(args, 'p23_lokr_rank', 4),
        p23_use_shared_block_router=getattr(args, 'p23_use_shared_block_router', 0),
        p23_linear_moe_experts=getattr(args, 'p23_linear_moe_experts', 0),
        p23_linear_moe_topk=getattr(args, 'p23_linear_moe_topk', 0),
        p23_quantile_route=getattr(args, 'p23_quantile_route', 0),
        remix_shared_context_gates=getattr(args, 'remix_shared_context_gates', 0),
        remix_use_dual_gate=bool(getattr(args, 'remix_use_dual_gate', 0)),
        p26_output_gated_linear=getattr(args, 'p26_output_gated_linear', 0),
        # Phase 28: FLOPs-efficient template routing
        p28_shared_basis=getattr(args, 'p28_shared_basis', 0),
        p28_chunk_routing_size=getattr(args, 'p28_chunk_routing_size', 0),
        p28_global_template_bank=getattr(args, 'p28_global_template_bank', 'none'),
        p28_attn_proj_templates=getattr(args, 'p28_attn_proj_templates', 0),
        p28_attn_qk_templates=getattr(args, 'p28_attn_qk_templates', 0),
        # Phase 24
        p24_use_sliced_weight=getattr(args, 'p24_use_sliced_weight', 0),
        p24_sliced_weight_reduction_scale=getattr(args, 'p24_sliced_weight_reduction_scale', 8),
        p24_sliced_weight_min_select=getattr(args, 'p24_sliced_weight_min_select', 128),
        p24_sliced_weight_scope=getattr(args, 'p24_sliced_weight_scope', 'per_token'),
        p24_sliced_weight_balance_coeff=getattr(args, 'p24_sliced_weight_balance_coeff', 0.01),
        p24_quantile_route=getattr(args, 'p24_quantile_route', 0),
        p24_use_folded_mod=getattr(args, 'p24_use_folded_mod', 0),
        p24_folded_mod_reduction_scale=getattr(args, 'p24_folded_mod_reduction_scale', 8),
        p24_folded_mod_min_dim=getattr(args, 'p24_folded_mod_min_dim', 128),
        p24_folded_mod_scope=getattr(args, 'p24_folded_mod_scope', 'per_layer'),
        p24_folded_mod_gate_act=getattr(args, 'p24_folded_mod_gate_act', 'tanh_centered'),
        p24_use_sequence_gated_linear=getattr(args, 'p24_use_sequence_gated_linear', 0),
        p24_sequence_gated_scope=getattr(args, 'p24_sequence_gated_scope', 'per_layer'),
        p24_sequence_gated_act=getattr(args, 'p24_sequence_gated_act', 'tanh_centered'),
        remix_output_gate_rank=getattr(args, 'remix_output_gate_rank', 16),
        # Phase 30: LayerNorm ablation
        remix_disable_ln_basis=getattr(args, 'remix_disable_ln_basis', 0),
        dense_intermediate_ln=getattr(args, 'dense_intermediate_ln', 0),
        # MST: Modular Sub-Transformer
        use_mst=bool(getattr(args, 'use_mst', 0)),
        mst_n_subs=getattr(args, 'mst_n_subs', 8),
        mst_sub_dim=getattr(args, 'mst_sub_dim', 64),
        mst_head_dim=getattr(args, 'mst_head_dim', 0),
        mst_input_mode=getattr(args, 'mst_input_mode', 'fixed_slice'),
        mst_rotated_slice_learned=bool(getattr(args, 'mst_rotated_slice_learned', 0)),
        mst_routing_mode=getattr(args, 'mst_routing_mode', 'soft_weighted'),
        mst_routing_topk=getattr(args, 'mst_routing_topk', 4),
        mst_routing_aux_weight=getattr(args, 'mst_routing_aux_weight', 0.01),
        mst_diversity_weight=getattr(args, 'mst_diversity_weight', 0.0),
        mst_ffn_mode=getattr(args, 'mst_ffn_mode', 'standard'),
        mst_transition_mode=getattr(args, 'mst_transition_mode', 'parallel'),
        mst_final_mode=getattr(args, 'mst_final_mode', 'aggregate_proj'),
        mst_final_topk=getattr(args, 'mst_final_topk', -1),
        mst_ffn_shared_up=getattr(args, 'mst_ffn_shared_up', 0),
        mst_ffn_inner_dim=getattr(args, 'mst_ffn_inner_dim', 0),
        mst_sub_dropout=getattr(args, 'mst_sub_dropout', 0.0),
        mst_transition_every=getattr(args, 'mst_transition_every', 1),
        mst_ffa_temperature=getattr(args, 'mst_ffa_temperature', 1.0),
        mst_global_residual=getattr(args, 'mst_global_residual', 0),
        mst_hybrid_dense=getattr(args, 'mst_hybrid_dense', 0),
        mst_cross_sub_kv=getattr(args, 'mst_cross_sub_kv', 0),
        mst_sub_aux_weight=getattr(args, 'mst_sub_aux_weight', 0.0),
        mst_progressive_merge=getattr(args, 'mst_progressive_merge', 0),
        mst_multi_scale_windows=getattr(args, 'mst_multi_scale_windows', 0),
        mst_delta_residual=getattr(args, 'mst_delta_residual', 0),
        mst_sub_layers=getattr(args, 'mst_sub_layers', 1),
        # Stage 7: Scaling improvements (P07)
        mst_grad_equalize=getattr(args, 'mst_grad_equalize', 0),
        mst_block_diagonal_muon=getattr(args, 'mst_block_diagonal_muon', 0),
        mst_transition_width_mult=getattr(args, 'mst_transition_width_mult', 1.0),
        mst_sub_lr_scale=getattr(args, 'mst_sub_lr_scale', 1.0),
        mst_shared_expert=getattr(args, 'mst_shared_expert', 0),
        mst_router_entropy_weight=getattr(args, 'mst_router_entropy_weight', 0.0),
        mst_shared_kv_attn=getattr(args, 'mst_shared_kv_attn', 0),
        mst_contrastive_diversity_weight=getattr(args, 'mst_contrastive_diversity_weight', 0.0),
        # Stage 8: Transition expressivity
        mst_transition_nonlinear=getattr(args, 'mst_transition_nonlinear', 0),
        mst_transition_gated=getattr(args, 'mst_transition_gated', 0),
        mst_transition_mlp=getattr(args, 'mst_transition_mlp', 0),
        # Stage 9: Cross-sub expressivity
        mst_cross_sub_gate=getattr(args, 'mst_cross_sub_gate', 0),
        mst_hyper_connect=getattr(args, 'mst_hyper_connect', 0),
        mst_cross_kv_inject=getattr(args, 'mst_cross_kv_inject', 0),
        # Stage 10: Structural transition improvements
        mst_slice_transition=getattr(args, 'mst_slice_transition', 0),
        mst_lookback_layers=getattr(args, 'mst_lookback_layers', 0),
        mst_bilinear_transition=getattr(args, 'mst_bilinear_transition', 0),
        # Stage 11: Attention bottleneck + structural improvements
        mst_cross_sub_qmod=getattr(args, 'mst_cross_sub_qmod', 0),
        mst_feature_cycle=getattr(args, 'mst_feature_cycle', 0),
        mst_mean_transition=getattr(args, 'mst_mean_transition', 0),
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
    # Stash tokenizer_dir on config for lazy prior loading in EET
    config._tokenizer_dir = getattr(args, 'tokenizer_dir', None)

    with torch.device("meta"):
        if config.use_eet:
            from nanochat.eet import EarlyExitGPT
            model_meta = EarlyExitGPT(config)
        elif config.use_mst:
            from nanochat.mst import MST
            model_meta = MST(config)
        else:
            model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# Phase 17: Auto-enable modulation diagnostics for any research model
# (always on for research, no need for --modulation-diagnostics flag)
mod_diag = None
diag_metrics = None  # Track last-collected metrics for wandb logging
if args.use_remix_linear:
    from nanochat.gpt import ModulationDiagnostics
    mod_diag = ModulationDiagnostics(model)
    if mod_diag._layers:
        print0(f"Modulation diagnostics enabled: tracking {len(mod_diag._layers)} conditioning layers")
    else:
        print0(f"Modulation diagnostics: no position-conditioned layers found (mode={args.cclblock_modulation})")
        mod_diag = None

# ── Gate stats collector ──────────────────────────────────────────────────────
def collect_gate_stats(model, step):
    """Walk all RemixedLinear / DualGateLinear layers and aggregate gate stats.

    Returns a dict with:
      basis_mean / basis_std / basis_dead / basis_sat   — sigmoid(gate_logits) stats
      out_mean   / out_std                              — 1+tanh(output_gate) stats
      gate_grad_norm   — L2 norm of all gate param gradients (0 if not yet computed)
      struct_grad_norm — L2 norm of all structural param gradients
    Each value is the mean across all tracked layers that have that stat.
    """
    from nanochat.gpt import RemixedLinear, DualGateLinear
    accum = {}
    counts = {}
    layers_seen = 0

    raw = model.module if hasattr(model, 'module') else model
    for mod in raw.modules():
        if not isinstance(mod, (RemixedLinear, DualGateLinear)):
            continue
        gs = getattr(mod, '_gate_stats', {})
        if not gs:
            continue
        layers_seen += 1
        for k, v in gs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            accum[k]  = accum.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1

    result = {'step': step, 'layers': layers_seen}
    for k in accum:
        result[k] = accum[k] / counts[k]

    # Gradient norms (only meaningful after loss.backward())
    gate_sq, gate_n, struct_sq, struct_n = 0.0, 0, 0.0, 0
    for mod in raw.modules():
        if not isinstance(mod, (RemixedLinear, DualGateLinear)):
            continue
        for p in mod.gate_parameters():
            if p.grad is not None:
                gate_sq += p.grad.float().norm().item() ** 2
                gate_n  += 1
        for p in mod.non_gate_parameters():
            if p.grad is not None:
                struct_sq += p.grad.float().norm().item() ** 2
                struct_n  += 1
    result['gate_grad_norm']   = (gate_sq   ** 0.5) if gate_n   > 0 else 0.0
    result['struct_grad_norm'] = (struct_sq ** 0.5) if struct_n > 0 else 0.0
    return result

# If we are resuming, overwrite the model parameters with those of the checkpoint
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
if args.checkpoints_dir:
    checkpoints_root = os.path.abspath(args.checkpoints_dir)
else:
    # default to a "base_checkpoints" folder inside the nanochat base directory
    checkpoints_root = os.path.join(get_base_dir(), "base_checkpoints")

checkpoint_dir = os.path.abspath(os.path.join(checkpoints_root, output_dirname))
print0(f"Checkpoints directory: {checkpoint_dir}")
# Gate stats log (only written by master process when --use-remix-linear and gate-stats-every > 0)
gate_stats_log = os.path.join(checkpoint_dir, "gate_stats.log") if (args.use_remix_linear and getattr(args, 'gate_stats_every', 0) > 0) else None
if gate_stats_log and master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(gate_stats_log, 'w') as _f:
        _f.write('# gate_stats.log — one JSON line per logged step\n')
        _f.write('# Keys: step, layers, basis_mean, basis_std, basis_dead, basis_sat, out_mean, out_std, gate_grad_norm, struct_grad_norm\n')
if args.step_loss_file and master_process:
    step_loss_dir = os.path.dirname(os.path.abspath(args.step_loss_file))
    if step_loss_dir:
        os.makedirs(step_loss_dir, exist_ok=True)
    # Fresh file per run.
    with open(args.step_loss_file, "w", encoding="utf-8"):
        pass
eet_ever_routed = False
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy
    eet_ever_routed = meta_data.get("eet_ever_routed", False)

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        args.fp8 = False
    else:
        # Check compute capability (requires 8.9+ for L4/4090 or 9.0+ for H100)
        major, minor = torch.cuda.get_device_capability()
        if major < 8 or (major == 8 and minor < 9):
            print0(f"Warning: FP8 training requires compute capability >= 8.9 (e.g. H100, L4, 4090), but detected {major}.{minor}. Disabling FP8.")
            args.fp8 = False

if args.fp8:
    # our custom fp8 is simpler than torchao, written for exact API compatibility
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn

    # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
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

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
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
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# Disable requires_grad for EET parameters during Phase 1 warmup
# to avoid DDP find_unused_parameters overhead and match dense speed perfectly.
if model_config.use_eet:
    for param in model.eet_routers.parameters():
        param.requires_grad = False
    for param in model.eet_translators.parameters():
        param.requires_grad = False

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = wrap_model(model, parallel_type=args.parallel, compile=args.compile, device=device)

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = orig_model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token, num_active_flops_per_token, num_active_params = orig_model.estimate_flops()
print0(f"Estimated FLOPs per token (total):  {num_flops_per_token:e}")
print0(f"Estimated FLOPs per token (active): {num_active_flops_per_token:e}")
print0(f"Estimated active params:            {num_active_params:,}")


# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(orig_model)
if args.target_tokens > 0:
    target_tokens = args.target_tokens
else:
    # When --target-active-params=1, adjust for inactive template/expert params so sparse models
    # receive a Chinchilla-optimal token budget proportional to their *active* parameter count.
    if getattr(args, 'target_active_params', 0) and num_active_params < num_params:
        inactive_params = num_params - num_active_params
        active_scaling_params = max(1, num_scaling_params - inactive_params)
        print0(f"Active scaling params: {active_scaling_params:,}  (total: {num_scaling_params:,}, inactive: {inactive_params:,})")
    else:
        active_scaling_params = num_scaling_params
    target_tokens = int(args.target_param_data_ratio * active_scaling_params)

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# Phase 20 (E/G/J): Convert standard MLP to Phase 2 variant if flags are set
# This must happen AFTER weights are initialized/loaded but BEFORE optimizer setup
_p20_pwu = getattr(args, 'p20_pwu_branches', 0)
_p20_fsvd = getattr(args, 'p20_fsvd_gate', 0)
_p20_wbfc = getattr(args, 'p20_wbfc_clusters', 0)
if _p20_pwu > 0 or _p20_fsvd > 0 or _p20_wbfc > 0:
    n_converted = orig_model.convert_to_phase2()
    print0(f"Phase 2 conversion complete: {n_converted} MLP modules converted")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = orig_model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    # μP
    disable_mu_p=args.disable_mu_p,
    mu_p_scale_override=args.mu_p_scale_override,
    # Gate LR scale (default 0.3×; lower values slow gate learning relative to structural weights)
    gate_lr_scale=args.remix_gate_lr_scale,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
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
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Research branches use a OneCycle-style schedule; base keeps the original warmup/flat/warmdown schedule
use_research_mode = args.use_moe or args.use_perm or args.use_remix_linear
use_research_scheduler = use_research_mode and bool(args.research_onecycle)
if use_research_scheduler:
    print0("Using research scheduler: OneCycle-style LR multiplier")
elif use_research_mode:
    print0("Research mode with OneCycle disabled: using base warmup/flat/warmdown schedule")
else:
    print0("Using base scheduler: linear warmup/flat/linear warmdown")

# Learning rate schedule (linear warmup, constant, linear warmdown)
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


def get_lr_multiplier_onecycle(it):
    """
    OneCycle-style multiplier in [final_lr_frac, 1.0].
    - rise phase: cosine from final_lr_frac -> 1.0
    - decay phase: cosine from 1.0 -> final_lr_frac
    Uses warmup_ratio as pct_start for the peak.
    """
    if num_iterations <= 1:
        return 1.0
    t = min(max(it, 0), num_iterations - 1)
    pct = t / (num_iterations - 1)
    warmup_src = args.warmup_ratio if args.research_warmup_ratio < 0 else args.research_warmup_ratio
    pct_start = min(max(warmup_src, 0.01), 0.99)
    low = args.final_lr_frac
    if pct <= pct_start:
        phase = pct / pct_start
        return low + (1.0 - low) * (1 - math.cos(math.pi * phase)) * 0.5
    phase = (pct - pct_start) / (1 - pct_start)
    return low + (1.0 - low) * (1 + math.cos(math.pi * phase)) * 0.5

# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# Fix 1H: PermutationMoE temperature scheduler
# Exponentially decays from perm_temp_start -> 1.0 over the first 50% of training
# Prevents early routing collapse from logit saturation and hard-argmax-like softmax behavior
def get_perm_temperature(it):
    t_start = args.perm_temp_start
    t_end = 1.0
    frac = min(it / max(num_iterations * 0.5, 1), 1.0)
    return t_start * (t_end / t_start) ** frac

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
mfu = 0.0  # updated each training step; initialized here so the post-loop report always has a value
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    min_val_loss = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    last_periodic_ckpt_step = -1 # step of the last periodic (non-final) checkpoint saved
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    min_val_loss = loop_state.get("min_val_loss", float("inf"))
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    # On resume, the checkpoint we resumed from is the last known periodic save
    last_periodic_ckpt_step = step

# ── MST Experiment Tracker ────────────────────────────────────────────────────
if model_config.use_mst and master_process:
    class _MSTTracker:
        """Accumulates per-step MST router diagnostics and val_loss milestones,
        then writes a single summary CSV row at end of training."""
        def __init__(self, config, run_dir):
            self.config = config
            self.run_dir = run_dir
            self.entropy_vals, self.balance_vals, self.grad_norm_vals = [], [], []
            self.val_at = {}  # step -> bpb

        def collect(self, model, grad_norm=None):
            """Call after each optimizer step to sample router diagnostics."""
            from nanochat.mst import MSTRouter
            for m in model.modules():
                if isinstance(m, MSTRouter):
                    if m._last_entropy is not None:
                        self.entropy_vals.append(float(m._last_entropy))
                    if m._last_balance is not None:
                        self.balance_vals.append(float(m._last_balance))
            if grad_norm is not None:
                self.grad_norm_vals.append(float(grad_norm))

        def record_val(self, step, bpb):
            self.val_at[step] = bpb

        def write_csv(self, step, val_bpb, train_loss_final, total_training_time,
                      num_flops_per_token, num_active_flops_per_token, num_active_params,
                      total_batch_size, num_params, sp):
            import csv, os, statistics
            c = self.config
            row = {
                'config_id':           getattr(args, 'model_tag', '') or f'd{args.depth}',
                'input_mode':          c.mst_input_mode,
                'routing_mode':        c.mst_routing_mode,
                'ffn_mode':            c.mst_ffn_mode,
                'transition_mode':     c.mst_transition_mode,
                'final_mode':          c.mst_final_mode,
                'n_subs':              c.mst_n_subs,
                'sub_dim':             c.mst_sub_dim,
                'head_dim':            c.mst_head_dim if c.mst_head_dim > 0 else c.mst_sub_dim // c.n_head,
                'val_loss_1k':         self.val_at.get(1000, ''),
                'val_loss_5k':         self.val_at.get(5000, ''),
                'val_loss_final':      val_bpb if val_bpb is not None else '',
                'train_loss_final':    f'{train_loss_final:.6f}' if train_loss_final is not None else '',
                'router_entropy_mean': f'{statistics.mean(self.entropy_vals):.4f}' if self.entropy_vals else '',
                'router_entropy_min':  f'{min(self.entropy_vals):.4f}'              if self.entropy_vals else '',
                'load_balance_score':  f'{statistics.mean(self.balance_vals):.4f}'  if self.balance_vals else '',
                'grad_norm_mean':      f'{statistics.mean(self.grad_norm_vals):.4f}' if self.grad_norm_vals else '',
                'wall_time_min':       f'{total_training_time / 60:.2f}',
                'flops_per_token':     f'{num_flops_per_token:.4e}',
                'active_flops_per_token': f'{num_active_flops_per_token:.4e}',
                'total_flops':         f'{num_flops_per_token * total_batch_size * step:.4e}',
                'active_total_flops':  f'{num_active_flops_per_token * total_batch_size * step:.4e}',
                'params_full':         num_params,
                'active_params':       num_active_params,
                'params_transformer':  sp.get('transformer_matrices', '') if isinstance(sp, dict) else '',
                'params_value_embeds': sp.get('value_embeds', '')         if isinstance(sp, dict) else '',
                'params_total_sp':     sp.get('total', '')                if isinstance(sp, dict) else '',
                # P07 scaling improvement flags
                'grad_equalize':       c.mst_grad_equalize,
                'block_diagonal_muon': c.mst_block_diagonal_muon,
                'transition_width_mult': c.mst_transition_width_mult,
                'sub_lr_scale':        c.mst_sub_lr_scale,
                'shared_expert':       c.mst_shared_expert,
                'router_entropy_weight': c.mst_router_entropy_weight,
                'shared_kv_attn':      c.mst_shared_kv_attn,
                'contrastive_div_weight': c.mst_contrastive_diversity_weight,
                # Stage 8: Transition expressivity
                'transition_nonlinear': c.mst_transition_nonlinear,
                'transition_gated':     c.mst_transition_gated,
                'transition_mlp':       c.mst_transition_mlp,
                # Stage 9: Cross-sub expressivity
                'cross_sub_gate':       c.mst_cross_sub_gate,
                'hyper_connect':        c.mst_hyper_connect,
                'cross_kv_inject':      c.mst_cross_kv_inject,
                # Stage 10: Structural transition improvements
                'slice_transition':     c.mst_slice_transition,
                'lookback_layers':      c.mst_lookback_layers,
                'bilinear_transition':  c.mst_bilinear_transition,
                # Stage 11: Attention bottleneck + structural improvements
                'cross_sub_qmod':       c.mst_cross_sub_qmod,
                'feature_cycle':        c.mst_feature_cycle,
                'mean_transition':      c.mst_mean_transition,
                'global_residual':      c.mst_global_residual,
            }
            # Write to checkpoint parent dir (original location)
            csv_path = os.path.normpath(os.path.join(self.run_dir, '..', 'mst_results.csv'))
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print0(f"[MST] Results appended → {csv_path}")
            # Also write to cwd so results survive regardless of checkpoint persistence
            cwd_csv = os.path.join(os.getcwd(), 'mst_results.csv')
            if os.path.abspath(cwd_csv) != os.path.abspath(csv_path):
                cwd_header = not os.path.exists(cwd_csv)
                with open(cwd_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if cwd_header:
                        writer.writeheader()
                    writer.writerow(row)
                print0(f"[MST] Results also appended → {cwd_csv}")

    _mst_tracker = _MSTTracker(model_config, checkpoint_dir)
    _mst_diag_log = os.path.join(checkpoint_dir, 'mst_diagnostics.jsonl')
    _mst_diag_every = args.log_every  # log diagnostics at same frequency as training logs
    print0(f"[MST] Diagnostics will be logged to: {_mst_diag_log}")
else:
    _mst_tracker = None
    _mst_diag_log = None
    _mst_diag_every = 0


# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
effective_device_batch_size = args.device_batch_size * (ddp_world_size if is_dp else 1)
tokens_per_fwdbwd = effective_device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * (1 if is_dp else ddp_world_size) # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
EMA_BETA = 0.9

# -----------------------------------------------------------------------------
# Pre-compilation warmup
# On the first real training step, three things overlap to create a memory spike
# that doesn't exist on subsequent steps:
#   1. torch.compile graph capture allocates temporary scratch buffers
#   2. Muon + AdamW m/v state tensors are lazily allocated on the first .step()
#   3. The CUDA caching allocator reserves large contiguous blocks on first use
# Together these can push peak VRAM 10–20 GB above the true steady-state cost,
# preventing you from sizing the batch to use the available headroom.
#
# Fix: run one throwaway forward + backward + step before real training.
# After this, all lazy allocations are done, compiler scratch has been released,
# and the allocator has defragmented. Step 0 of real training then looks like step 2.
#
# We use zero-filled tensors with the real batch shapes so the compiler traces
# with correct sizes but the data ordering is completely unaffected.
if not resuming and device_type == "cuda":
    print0("Running pre-compilation warmup (1 dummy forward+backward to init lazy allocations)...")
    torch.cuda.reset_peak_memory_stats()
    _wx = torch.zeros_like(x)
    _wy = torch.zeros_like(y)
    if model_config.use_eet:
        # Phase 1 delegates to super().forward() which ignores EET kwargs,
        # but we still need to pass eet_do_route=False so the guard matches.
        _wloss = model(_wx, _wy, eet_do_route=False)
    else:
        _wloss = model(_wx, _wy)
    if is_dp:
        _wloss = _wloss.mean()
    (_wloss / grad_accum_steps).backward()
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]  # sensible default for the dummy step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    del _wx, _wy, _wloss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # reset so step-0 peak reflects real training only
    print0(f"Warmup complete. Steady-state VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")

# Go!
while True:
    last_step = step == num_iterations # normal end

    # early stop: force this to be the last step if we hit the token limit
    if args.early_stop_tokens > 0 and step * total_batch_size >= args.early_stop_tokens:
        print0(f"[early stop] Reached {step * total_batch_size:,} tokens (limit: {args.early_stop_tokens:,}). Initiating final eval/save.")
        last_step = True

    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
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
        if _mst_tracker is not None:
            _mst_tracker.record_val(step, val_bpb)
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if (args.core_metric_every != 0) and (last_step or (args.core_metric_every > 0 and step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    is_periodic_save = (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0)
    if last_step or is_periodic_save:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "eet_ever_routed": eet_ever_routed,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        # Rolling checkpoints: after a periodic save, delete the previous periodic checkpoint
        # to avoid disk bloat. The final checkpoint (last_step) is always kept.
        if is_periodic_save and master_process and last_periodic_ckpt_step >= 0:
            prev = last_periodic_ckpt_step
            for pattern in [
                os.path.join(checkpoint_dir, f"model_{prev:06d}.pt"),
                os.path.join(checkpoint_dir, f"meta_{prev:06d}.json"),
            ]:
                if os.path.exists(pattern):
                    os.remove(pattern)
            # Remove optimizer shards from all ranks (rank files we don't own are best-effort)
            for r in range(ddp_world_size):
                optim_path = os.path.join(checkpoint_dir, f"optim_{prev:06d}_rank{r:d}.pt")
                if os.path.exists(optim_path):
                    os.remove(optim_path)
            print0(f"Removed previous periodic checkpoint at step {prev:06d}")
        if is_periodic_save:
            last_periodic_ckpt_step = step

    # termination conditions
    if last_step:
        # Ensure sweep parsers always see an end-of-run loss line, even when
        # early-stop triggers before hitting a log_every boundary.
        debiased_at_step = smooth_train_loss / (1 - EMA_BETA**max(step, 1))
        print0(f"step {step:05d}/{num_iterations:05d} (final) | loss: {debiased_at_step:.6f} | early_stop: {int(args.early_stop_tokens > 0)}")
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    # Enable EET Phase 2 to Phase 3 transition diagnostic check
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
        # Only run the Phase 3 transition diagnostic when there actually is a Phase 3 and we had an exploration phase
        if _eet_sched.explore_end < num_iterations and _eet_sched.explore_end > _eet_sched.warmup_end and step == _eet_sched.explore_end:
            print0(f"\n[EET DIAGNOSTIC] Step {step:05d}: Running router structure check before entering Phase 3...")
            orig_model.train()  # must be training=True so do_route = eet_do_route and self.training → True
            with torch.no_grad():
                # Forward pass in soft-routing mode to populate _last_exit_probs
                _ = orig_model(x, eet_do_route=True, eet_phase=2, eet_lambda_r=0.0, eet_lambda_e=0.0)
            
            if hasattr(orig_model, '_last_exit_probs'):
                try:
                    check_router_learned(orig_model._last_exit_probs, x, orig_model)
                except Exception as e:
                    print0(f"[EET DIAGNOSTIC] Warning: correlation check failed with error: {e}")
            else:
                print0("[EET DIAGNOSTIC] Warning: exit probabilities not captured during forward pass.")
            
            orig_model.train()

    # Enable MST diagnostic capture on log steps (last micro-step only)
    _mst_diag_this_step = (_mst_tracker is not None and _mst_diag_every > 0 and
                           (step % _mst_diag_every == 0 or step == num_iterations - 1))
    for micro_step in range(grad_accum_steps):
        # Only capture diagnostics on the last micro-step to avoid overhead
        if _mst_diag_this_step and micro_step == grad_accum_steps - 1:
            orig_model._diag_enabled = True
        if model_config.use_eet:
            # Phase scheduling computed OUTSIDE compiled forward to prevent
            # torch.compile from generating a single graph covering all phases
            # (which pre-allocates buffers for reconstruction loss etc.)
            from nanochat.eet import EETPhaseScheduler
            _eet_phase_info = EETPhaseScheduler(
                num_iterations,
                warmup_frac=model_config.eet_warmup_frac,
                explore_frac=model_config.eet_explore_frac,
                reconstruct_lambda=model_config.eet_reconstruct_lambda,
                efficiency_lambda_start=model_config.eet_efficiency_lambda_start,
                efficiency_lambda_end=model_config.eet_efficiency_lambda_end,
            ).get_phase(step)

            use_gumbel = getattr(model_config, 'eet_gumbel_temp_start', 0.0) > 0.0
            is_layer_weighted = (model_config.eet_loss_variant == 'layer_weighted')
            bypass_phases = use_gumbel or is_layer_weighted

            eet_do_route = _eet_phase_info['do_route']
            eet_phase = _eet_phase_info['phase']
            if eet_do_route:
                eet_ever_routed = True

            if eet_phase == 1:
                # Phase 1: take the EXACT same code path as non-EET dense training.
                # No extra kwargs, no tensor allocations, no requires_grad loops.
                # This ensures torch.compile generates the identical graph as dense.
                loss = model(x, y)
            else:
                # Phase 2/3: full EET forward with routing

                # Check for transition from Phase 1 to Phase 2/3
                if (hasattr(orig_model, 'eet_current_phase') and 
                    orig_model.eet_current_phase == 1):
                    orig_model.eet_current_phase = eet_phase
                    orig_model.eet_phase_tracker[0] = eet_phase
                    print0(f"[EET] Transitioning from Phase 1 (Dense Warmup) to Phase {eet_phase} (Routing active).")

                # Ensure routers and translators remain trainable (avoid writing if already True)
                for param in orig_model.eet_routers.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                for param in orig_model.eet_translators.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True

                # Capacity annealing: update target_active_frac at discrete intervals to avoid recompilation breaks
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

                # Two-pass ce_guided: run dense forward first to get per-token CE baseline
                _reinforce_interval = getattr(model_config, 'eet_reinforce_interval', 0)
                if (_reinforce_interval > 0 and eet_phase == 3 and
                    step % _reinforce_interval == 0 and eet_do_route):
                    dense_ce = orig_model._compute_dense_per_token_ce(x, y)
                    orig_model._reinforce_dense_ce = dense_ce

                # Concurrent dense distillation: run dense forward pass to get teacher targets
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
            
        # Capture the final training step probabilities on the last micro-step of the final training step
        if model_config.use_eet and (last_step or step == num_iterations - 1) and micro_step == grad_accum_steps - 1:
            if hasattr(orig_model, '_last_exit_probs') and orig_model._last_exit_probs is not None:
                orig_model._final_train_exit_probs = orig_model._last_exit_probs.detach().cpu().clone()
                orig_model._final_train_tokens = x.detach().cpu().clone()
            if hasattr(orig_model, '_last_enforced_capacities'):
                orig_model._final_enforced_capacities = list(orig_model._last_enforced_capacities)
                orig_model._final_active_counts = list(orig_model._last_active_counts)
                orig_model._final_T = orig_model._last_T
                
        if _mst_diag_this_step and micro_step == grad_accum_steps - 1:
            orig_model._diag_enabled = False
        if is_dp:
            loss = loss.mean()
        train_loss = loss.detach() # for logging
        # 18B: Gradient penalty — weight Frobenius norm regularization
        # Penalizes large weight norms to constrain Lipschitz constant
        gp_lambda = float(getattr(args, 'p18_gradient_penalty', 0.0))
        if gp_lambda > 0:
            gp_loss = sum(
                p.float().square().mean() for p in model.parameters()
                if p.ndim >= 2 and p.requires_grad
            )
            loss = loss + gp_lambda * gp_loss
        # Phase 20: Auxiliary routing loss for DGCR (20D) and NCEA (20H)
        # These train the router to predict which branch is best, separate from main loss
        _p20_dgcr = getattr(args, 'p20_dgcr_branches', 0)
        _p20_ncea = getattr(args, 'p20_ncea_branches', 0)
        _p23_std_moe = getattr(args, 'p23_std_moe_experts', 0)
        if _p20_dgcr > 0 or _p20_ncea > 0 or _p23_std_moe > 0:
            aux_loss_total = torch.tensor(0.0, device=loss.device)
            for block in orig_model.transformer.h:
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'compute_aux_loss'):
                    aux = block.mlp.compute_aux_loss()
                    if aux is not None:
                        aux_loss_total = aux_loss_total + aux
            if aux_loss_total.item() > 0:
                loss = loss + aux_loss_total
        # EET: ce_guided depth classification — computed OUTSIDE the compiled forward
        # to avoid changing the compiled graph. Re-runs the tiny global router on the
        # stored embedding to get fresh router_logits WITH gradients.
        if (model_config.use_eet and model_config.eet_compute_skip and
            getattr(model_config, 'eet_loss_variant', '') == 'ce_guided' and
            hasattr(orig_model, '_last_x0_for_ce') and
            hasattr(orig_model, '_last_per_token_ce')):
            _x0 = orig_model._last_x0_for_ce         # (B, T, C), detached
            _ptce = orig_model._last_per_token_ce     # (B, T), detached
            _tgt = y

            # Re-run the global router (tiny: 2 linear layers) to get differentiable logits
            _global_router = orig_model.eet_routers[0]
            _freq_bias = getattr(orig_model, '_freq_bias', None)
            _pos_bias = getattr(orig_model, '_pos_bias', None)
            _rl = _global_router(
                _x0,
                freq_bias=_freq_bias,
                pos_bias=_pos_bias,
                freq_alpha=model_config.eet_freq_prior_alpha,
                pos_beta=model_config.eet_pos_prior_beta
            )  # (B, T, n_exits) — WITH gradients to router params

            _n_exits = _rl.shape[-1]
            _valid_mask = (_tgt != -1).float()
            _n_valid = _valid_mask.sum().clamp(min=1.0)

            # Two-pass: use differential CE if dense baseline is available
            _dense_ce = getattr(orig_model, '_reinforce_dense_ce', None)
            _ce_signal = (_ptce - _dense_ce) if _dense_ce is not None else _ptce

            with torch.no_grad():
                _valid_ce = _ce_signal[_tgt != -1]
                if _valid_ce.numel() > 1:
                    _ce_mean = _valid_ce.mean()
                    _ce_std = _valid_ce.std().clamp(min=0.1)
                    _normalized = torch.sigmoid((_ce_signal - _ce_mean) / _ce_std)
                    _target_exit = (_normalized * _n_exits).long().clamp(0, _n_exits - 1)
                else:
                    _target_exit = torch.zeros_like(_tgt)
                _target_exit = _target_exit * (_tgt != -1).long()

            _depth_loss = F.cross_entropy(
                _rl.float().view(-1, _n_exits),
                _target_exit.view(-1),
                ignore_index=-100,
                reduction='none',
            )
            _depth_loss = (_depth_loss.view_as(_tgt) * _valid_mask).sum() / _n_valid
            _ce_lambda = getattr(model_config, 'eet_ce_guided_lambda', 1.0)
            loss = loss + _ce_lambda * _depth_loss

        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # Capture MST per-sub grad norms BEFORE optimizer step / zero_grad clears them
    if _mst_diag_this_step:
        _cached_sub_grad_norms = {}
        N = model_config.get('mst_n_subs', 0) if isinstance(model_config, dict) else getattr(model_config, 'mst_n_subs', 0)
        # Accumulate grad norms per sub index across layers.
        # With progressive merge, later layers have fewer subs — only iterate valid indices.
        sub_grad_sq = [0.0] * N
        for layer in orig_model.layers:
            # BatchedMSTLayer stores weights as fused 2D tensors — no sub_blocks
            if not hasattr(layer, 'sub_blocks'):
                # For batched layers, attribute per-sub grad norms from the fused
                # attention/FFN weights. Each weight is (N*out, in), so we split by N.
                for attr in ('c_q_w', 'c_k_w', 'c_v_w', 'c_proj_w', 'fc_w', 'fc_proj_w'):
                    p = getattr(layer, attr, None)
                    if p is not None and p.grad is not None:
                        n_subs = getattr(layer, 'N', N)
                        chunks = p.grad.view(n_subs, -1, p.grad.shape[-1])
                        for j in range(min(n_subs, N)):
                            sub_grad_sq[j] += float(chunks[j].float().norm() ** 2)
            else:
                n_subs_this_layer = len(layer.sub_blocks)
                for j in range(n_subs_this_layer):
                    block = layer.sub_blocks[j]
                    for p in block.parameters():
                        if p.grad is not None:
                            # Map merged sub indices back: after merge, sub j in this layer
                            # encompasses original subs [j*stride, (j+1)*stride).
                            # For simplicity, accumulate to sub index j (valid for uniform N).
                            if j < N:
                                sub_grad_sq[j] += float(p.grad.float().norm() ** 2)
        for j in range(N):
            _cached_sub_grad_norms[f'grad_norm_S{j}'] = float(sub_grad_sq[j] ** 0.5)
        orig_model._cached_grad_norms = _cached_sub_grad_norms
    # Placeholder: EET gradient diagnostics are captured AFTER clipping (see below)
    _eet_grad_pending = model_config.use_eet and hasattr(orig_model, 'eet_routers') and eet_phase >= 2
    # step the optimizer
    lrm = get_lr_multiplier_onecycle(step) if use_research_scheduler else get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # Clip early-exit router gradients specifically to keep routing updates slow and stable
        if hasattr(orig_model, 'eet_routers') and orig_model.eet_routers is not None and eet_phase >= 2:
            torch.nn.utils.clip_grad_norm_(orig_model.eet_routers.parameters(), max_norm=10.0)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Clip early-exit router gradients specifically to keep routing updates slow and stable
        if hasattr(orig_model, 'eet_routers') and orig_model.eet_routers is not None and eet_phase >= 2:
            torch.nn.utils.clip_grad_norm_(orig_model.eet_routers.parameters(), max_norm=10.0)
    # Capture EET router/translator gradient norms AFTER clipping so logged values
    # reflect what the optimizer actually sees (not the raw pre-clip norms)
    if _eet_grad_pending:
        _eet_grad_info = {'step': step}
        total_router_grad_sq = 0.0
        n_router_params_with_grad = 0
        for layer_idx, router in enumerate(orig_model.eet_routers):
            layer_grad_sq = 0.0
            layer_n = 0
            for pname, p in router.named_parameters():
                if p.grad is not None:
                    gnorm = float(p.grad.float().norm())
                    layer_grad_sq += gnorm ** 2
                    layer_n += 1
                    _eet_grad_info[f'router_{layer_idx}_{pname}_grad_norm'] = gnorm
            layer_gnorm = layer_grad_sq ** 0.5
            _eet_grad_info[f'router_{layer_idx}_total_grad_norm'] = layer_gnorm
            total_router_grad_sq += layer_grad_sq
            n_router_params_with_grad += layer_n
        _eet_grad_info['router_total_grad_norm'] = total_router_grad_sq ** 0.5
        _eet_grad_info['n_router_params_with_grad'] = n_router_params_with_grad
        total_trans_grad_sq = 0.0
        for layer_idx, translator in enumerate(orig_model.eet_translators):
            layer_grad_sq = 0.0
            for pname, p in translator.named_parameters():
                if p.grad is not None:
                    gnorm = float(p.grad.float().norm())
                    layer_grad_sq += gnorm ** 2
                    _eet_grad_info[f'translator_{layer_idx}_{pname}_grad_norm'] = gnorm
            _eet_grad_info[f'translator_{layer_idx}_total_grad_norm'] = layer_grad_sq ** 0.5
            total_trans_grad_sq += layer_grad_sq
        _eet_grad_info['translator_total_grad_norm'] = total_trans_grad_sq ** 0.5
        orig_model._eet_grad_info = _eet_grad_info
    # Resume non-scaler optimizer step logic (Fix 4C, GER, optimizer.step)
    if scaler is None:
        # Fix 4C: gradient clipping before optimizer step (protects against large gradients
        # in adaptive gate pathways early in training, e.g. PermutationMoE, RemixedLinear)
        clip_val = 10.0 if (model_config.use_eet and args.max_grad_norm == 1.0) else args.max_grad_norm
        if clip_val > 0:
            torch.nn.utils.clip_grad_norm_(orig_model.parameters(), clip_val)
        # 19F: Gradient Equilibrium Regularization (gradient modifier)
        # Equalizes per-block gradient norms to prevent gradient starvation/domination
        ger_lambda = float(getattr(args, 'p19_grad_equilibrium', 0.0))
        if ger_lambda > 0:
            with torch.no_grad():
                block_grad_norms = []
                block_key_params = []
                for block in orig_model.transformer.h:
                    mlp = block.mlp if hasattr(block, 'mlp') else (block.ffwd if hasattr(block, 'ffwd') else None)
                    if mlp is not None and hasattr(mlp, 'c_fc') and mlp.c_fc.weight.grad is not None:
                        gn = mlp.c_fc.weight.grad.float().norm()
                        block_grad_norms.append(gn)
                        block_key_params.append(mlp.c_fc.weight)
                if len(block_grad_norms) >= 2:
                    gn_stack = torch.stack(block_grad_norms)
                    gn_mean = gn_stack.mean()
                    # Correction factor: scale each layer's gradients toward the mean
                    for idx, (gn, param) in enumerate(zip(block_grad_norms, block_key_params)):
                        if gn > 1e-12:
                            correction = (gn_mean / gn).clamp(1.0 - ger_lambda, 1.0 + ger_lambda)
                            param.grad.mul_(correction.to(param.grad.dtype))
        optimizer.step()
    # Fix 1H: update PermutationMoE temperature each step
    if use_research_mode:
        perm_temp = get_perm_temperature(step)
        for module in orig_model.modules():
            if hasattr(module, 'temperature') and isinstance(getattr(module, 'temperature'), torch.Tensor):
                module.temperature.fill_(perm_temp)
    # Phase 16A: CKR temperature annealing (exponential: temp_start → temp_end)
    if getattr(args, 'cclblock_modulation', '') in ('ckr', 'ckr_ffn'):
        temp_start = getattr(args, 'cclblock_ckr_temp_start', 1.0)
        temp_end = getattr(args, 'cclblock_ckr_temp_end', 1.0)
        if temp_start != temp_end and num_iterations > 1:
            progress = min(step / (num_iterations - 1), 1.0)
            current_temp = temp_start * (temp_end / temp_start) ** progress
            from nanochat.gpt import CausalKernelLinear
            for module in orig_model.modules():
                if isinstance(module, CausalKernelLinear):
                    module._temperature.fill_(current_temp)
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    # Collect MST router diagnostics (reads tensors already computed in forward)
    if _mst_tracker is not None:
        _mst_tracker.collect(orig_model)
        # Compute and log full diagnostics if this was a diagnostic step
        if _mst_diag_this_step and master_process:
            try:
                mst_diag = orig_model.compute_diagnostics()
                mst_diag['step'] = step
                mst_diag['train_loss'] = train_loss_f
                with open(_mst_diag_log, 'a') as f:
                    # Convert any non-serializable values
                    f.write(json.dumps(mst_diag, default=str) + '\n')
                # Print summary to stdout
                sim_keys = [k for k in mst_diag if 'sub_sim' in k and '_mean' in k]
                ent_keys = [k for k in mst_diag if 'route_entropy' in k]
                grad_keys = [k for k in mst_diag if 'grad_norm' in k]
                sim_vals = [mst_diag[k] for k in sorted(sim_keys)]
                ent_vals = [mst_diag[k] for k in sorted(ent_keys)]
                grad_vals = [mst_diag[k] for k in sorted(grad_keys)]
                sim_str = ', '.join(f'{v:.3f}' for v in sim_vals) if sim_vals else 'n/a'
                ent_str = ', '.join(f'{v:.3f}' for v in ent_vals) if ent_vals else 'n/a'
                grad_str = ', '.join(f'{v:.3f}' for v in grad_vals) if grad_vals else 'n/a'
                print0(f"  [MST diag] sub_sim=[{sim_str}] | route_ent=[{ent_str}] | grad_norm=[{grad_str}]")
            except Exception as e:
                print0(f"  [MST diag] ERROR: {e}")
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    smooth_train_loss = EMA_BETA * smooth_train_loss + (1 - EMA_BETA) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - EMA_BETA**(step + 1)) # debias the EMA
    if args.step_loss_file and master_process:
        with open(args.step_loss_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": int(step),
                "tokens": int(step * total_batch_size),
                "loss": float(debiased_smooth_loss),
            }) + "\n")
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    adamw_lrs = [g["lr"] for g in optimizer.param_groups if g.get("kind") == "adamw"]
    muon_lrs = [g["lr"] for g in optimizer.param_groups if g.get("kind") == "muon"]
    lr_msg = f"lr(adamw:{(sum(adamw_lrs)/len(adamw_lrs)) if adamw_lrs else 0:.3e}, muon:{(sum(muon_lrs)/len(muon_lrs)) if muon_lrs else 0:.3e})"
    if step % args.log_every == 0 or step == num_iterations - 1 or last_step:
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | {lr_msg} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
        # Phase 17: Modulation diagnostics at log intervals
        if mod_diag is not None:
            diag_metrics = mod_diag.collect()
            if diag_metrics:
                print0(mod_diag.format(diag_metrics))
                if master_process:
                    diag_file = os.path.join(checkpoint_dir, "modulation_diagnostics.jsonl")
                    mod_diag.save_to_file(diag_metrics, step, diag_file)
            else:
                diag_metrics = None
            # Phase 19: Expanded diagnostics (always collected, even without CKR layers)
            p19_metrics = mod_diag.collect_p19(orig_model)
            if p19_metrics:
                p19_log = mod_diag.format_p19(p19_metrics)
                if p19_log:
                    print0(p19_log)
            # Phase 20: Dynamic weight computation diagnostics
            p20_metrics = mod_diag.collect_p20(orig_model)
            if p20_metrics:
                p20_log = mod_diag.format_p20(p20_metrics)
                if p20_log:
                    print0(p20_log)
        else:
            p19_metrics = None
            p20_metrics = None
        # ── Gate stats logging ────────────────────────────────────────────────
        if gate_stats_log and args.gate_stats_every > 0 and (step % args.gate_stats_every == 0 or last_step):
            gs = collect_gate_stats(model, step)
            print0(
                f"  gate_stats | layers={gs['layers']} "
                f"| basis µ={gs.get('basis_mean',float('nan')):.3f} "
                f"σ={gs.get('basis_std',float('nan')):.3f} "
                f"dead={gs.get('basis_dead',float('nan')):.1%} "
                f"sat={gs.get('basis_sat',float('nan')):.1%} "
                f"| out µ={gs.get('out_mean',float('nan')):.3f} "
                f"σ={gs.get('out_std',float('nan')):.3f} "
                f"| ∇gate={gs['gate_grad_norm']:.3e} "
                f"∇struct={gs['struct_grad_norm']:.3e}"
            )
            if master_process:
                with open(gate_stats_log, 'a') as _gf:
                    _gf.write(json.dumps(gs) + '\n')
        # EET: Early Exit diagnostics (safe to call .item() here — outside compiled forward)
        if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics'):
            _eet_diag = orig_model._eet_diagnostics
            _eet_phase = _eet_diag.get('phase', 0)
            _eet_active = _eet_diag['active_frac'].item() if hasattr(_eet_diag.get('active_frac', 0), 'item') else _eet_diag.get('active_frac', 1.0)
            _eet_exit = _eet_diag['total_exit_frac'].item() if hasattr(_eet_diag.get('total_exit_frac', 0), 'item') else _eet_diag.get('total_exit_frac', 0.0)
            _active_counts = getattr(orig_model, '_last_active_counts', None)
            _counts_str = f" | tokens={_active_counts}" if _active_counts else ""
            _a3d_str = " [A³D]" if _eet_diag.get('a3d', False) else ""
            print0(f"  eet{_a3d_str} | phase={_eet_phase} | active={_eet_active:.3f} | exit_frac={_eet_exit:.3f}{_counts_str}")
        # EET: Router gradient diagnostics — console print + JSONL log file
        if model_config.use_eet and hasattr(orig_model, '_eet_grad_info'):
            _gi = orig_model._eet_grad_info
            _r_gnorm = _gi.get('router_total_grad_norm', 0.0)
            _t_gnorm = _gi.get('translator_total_grad_norm', 0.0)
            _n_with_grad = _gi.get('n_router_params_with_grad', 0)
            # Per-layer summary
            per_layer_parts = []
            for layer_idx in range(len(orig_model.eet_routers)):
                ln = _gi.get(f'router_{layer_idx}_total_grad_norm', 0.0)
                per_layer_parts.append(f'L{layer_idx}={ln:.3e}')
            per_layer_str = ' '.join(per_layer_parts) if per_layer_parts else 'none'
            print0(f"  eet_grad | ∇router={_r_gnorm:.3e} ∇trans={_t_gnorm:.3e} params_with_grad={_n_with_grad} | {per_layer_str}")
            # Write full grad info to JSONL log file (only when routing is active)
            if master_process and _gi.get('n_router_params_with_grad', 0) > 0:
                import json
                _gi['eet_phase'] = _eet_diag.get('phase', 0) if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics') else 0
                _gi['active_frac'] = _eet_active if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics') else 1.0
                eet_grad_log_path = os.path.join(checkpoint_dir, "eet_grad_log.jsonl")
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(eet_grad_log_path, 'a') as _ef:
                    _ef.write(json.dumps(_gi) + '\n')
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        # Add modulation diagnostics to wandb if available
        if mod_diag is not None and diag_metrics is not None:
            log_data.update(mod_diag.to_dict(diag_metrics))
        # Phase 19: expanded diagnostics
        if mod_diag is not None and p19_metrics:
            log_data.update(mod_diag.to_dict_p19(p19_metrics))
        # Phase 20: dynamic weight diagnostics
        if mod_diag is not None and p20_metrics:
            log_data.update(mod_diag.to_dict_p20(p20_metrics))
        # EET: add exit diagnostics to step log for post-hoc effective FLOPs analysis
        if model_config.use_eet and hasattr(orig_model, '_eet_diagnostics'):
            _eet_diag = orig_model._eet_diagnostics
            _af = _eet_diag.get('active_frac', None)
            _ef = _eet_diag.get('total_exit_frac', None)
            log_data['eet/phase'] = _eet_diag.get('phase', 0)
            log_data['eet/active_frac'] = _af.item() if hasattr(_af, 'item') else float(_af or 1.0)
            log_data['eet/exit_frac'] = _ef.item() if hasattr(_ef, 'item') else float(_ef or 0.0)
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
    print0(f"Minimum validation loss (nats/byte): {min_val_loss:.6f}")

# EET: Print comprehensive final router diagnostics at the end of the run
if model_config.use_eet:
    print0("\n================================================================================")
    print0("[EET FINAL ROUTER DIAGNOSTICS]")
    print0("================================================================================")

    # 1. Enforced physical capacity distribution (what actually happened with tokens)
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
        # Compute per-slot exit counts from active_counts
        # active_counts[i] = K_cur at block i (before block runs)
        # Exit at slot k = active_counts[routing_block_k] - active_counts[routing_block_k+1]
        for slot in range(n_rl):
            # Block index for this routing slot: warmup block is 0, routing slots are 1..n_rl
            block_idx = slot + 1  # after warmup
            k_before = active_counts[block_idx]
            if block_idx + 1 < n_blocks:
                k_after = active_counts[block_idx + 1]
            else:
                k_after = active_counts[-1]  # last block's active count
            exited = k_before - k_after
            pct = (exited / T) * 100
            label = f'exit_{slot}'
            print0(f"  {label:12s}: {pct:6.2f}% ({exited * B:,} tokens) [capacity: {capacities[slot] * B:,}]")
        # Final layer
        final_active = active_counts[-1]
        final_pct = (final_active / T) * 100
        print0(f"  {'final_layer':12s}: {final_pct:6.2f}% ({final_active * B:,} tokens active)")

        # Enforced active fraction
        enforced_active = sum(active_counts) / (n_blocks * T)
        enforced_exit = 1.0 - (active_counts[-1] / T)
        print0(f"\nEnforced active fraction: {enforced_active:.3f} (exit_frac={enforced_exit:.3f})")
    else:
        print0("(No enforced capacity data available — compute_skip may not have been enabled)")

    # 2. Soft probability diagnostics (router's learned preferences)
    if hasattr(orig_model, '_final_train_exit_probs') and orig_model._final_train_exit_probs is not None:
        exit_probs = orig_model._final_train_exit_probs.detach().cpu().float() # (B, T, n_exits)
        n_exits = exit_probs.size(-1)

        # Argmax exit layer for every token
        argmax_exits = exit_probs.argmax(dim=-1).numpy().ravel()
        total_tokens_evaluated = len(argmax_exits)

        print0(f"\nRouter soft probability diagnostics ({total_tokens_evaluated:,} tokens):")
        print0("Router preferred exit (argmax of soft probs):")
        for slot in range(n_exits):
            count = (argmax_exits == slot).sum()
            pct = (count / max(total_tokens_evaluated, 1)) * 100
            label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
            print0(f"  {label:12s}: {pct:6.2f}% ({count:,} tokens)")

        # Soft probability stats per slot
        print0("\nSoft exit probability stats per slot:")
        for slot in range(n_exits):
            slot_probs = exit_probs[:, :, slot].numpy().ravel()
            label = 'final_layer' if slot == n_exits - 1 else f'exit_{slot}'
            print0(f"  {label:12s}: mean={slot_probs.mean():.6f} std={slot_probs.std():.6f} min={slot_probs.min():.6f} max={slot_probs.max():.6f}")

        # Mean and std of expected exit layer
        layer_indices = torch.arange(n_exits).float()
        expected_exit = (exit_probs * layer_indices).sum(dim=-1).numpy().ravel()
        print0(f"\nMean expected exit layer (soft): {expected_exit.mean():.4f}")
        print0(f"Std expected exit layer (soft):  {expected_exit.std():.6f}")

        # Collapse warnings
        if expected_exit.std() < 0.01:
            print0(f"\n[EET WARNING] ⚠ ROUTER COLLAPSE: std={expected_exit.std():.6f} — router is near-constant across all tokens.")
        elif expected_exit.std() < 0.1:
            print0(f"\n[EET WARNING] ⚠ LOW DIFFERENTIATION: std={expected_exit.std():.6f} — router is barely differentiating tokens.")
        else:
            print0("\n[EET INFO] ✓ Router has learned token-dependent exit behavior!")
    else:
        print0("\n[EET FINAL DIAGNOSTIC] Warning: No router exit probabilities captured during training steps.")
    print0("================================================================================\n")

# MST: write per-run summary row to mst_results.csv
if _mst_tracker is not None:
    sp = orig_model.num_scaling_params() if hasattr(orig_model, 'num_scaling_params') else {}
    final_train_loss = smooth_train_loss / (1 - EMA_BETA**max(step, 1)) if step > 0 else None
    _mst_tracker.write_csv(
        step=step,
        val_bpb=val_bpb,
        train_loss_final=final_train_loss,
        total_training_time=total_training_time,
        num_flops_per_token=num_flops_per_token,
        num_active_flops_per_token=num_active_flops_per_token,
        num_active_params=num_active_params,
        total_batch_size=total_batch_size,
        num_params=num_params,
        sp=sp,
    )

# Log to report
from nanochat.report import get_report
section_name = "Base model training"
if args.model_tag:
    section_name += f" ({args.model_tag})"
get_report().log(section=section_name, data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
        "research_warmup_ratio": args.research_warmup_ratio,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
