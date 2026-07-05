"""
Unified Flash Attention interface with automatic FA4/FA3/SDPA switching.

Detection priority (highest to lowest):
  1. Flash Attention 4 (Blackwell, sm100+)  — flash-attn-4 package, flash_attn.cute module
  2. Flash Attention 3 (Hopper, sm90)        — kernels package, varunneal/flash-attention-3
  3. PyTorch SDPA                            — everything else (Ampere, MPS, CPU, …)

Exports `flash_attn` module that matches the FA3/FA4 API exactly.

Usage (drop-in replacement for FA3 / FA4):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try FA4 first (Blackwell, sm100+), then FA3 (Hopper, sm90)
# =============================================================================

def _load_flash_attention_4():
    """Try to load Flash Attention 4 (Blackwell sm100+).

    FA4 ships as the `flash-attn-4` pip package with its API under `flash_attn.cute`.
    FA4 supports Blackwell, so we try it first on modern GPUs.
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    # FA4 targets Blackwell (sm100+)
    if major < 10:
        return None
    try:
        # FA4 exports: flash_attn_func, flash_attn_varlen_func
        # (flash_attn_with_kvcache not yet implemented in FA4)
        from flash_attn.cute import flash_attn_func  # noqa: F401
        import flash_attn.cute as fa4_module
        return fa4_module
    except Exception as e:
        import sys, traceback
        print(f"[flash_attention] FA4 import failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90).

    FA3 is loaded via the `kernels` hub package (varunneal/flash-attention-3).
    Only tried when FA4 is not available.
    Note: varunneal FA3 kernels target sm90 (Hopper) only — not Blackwell.
    """
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 Hopper kernels target sm90 exactly; Blackwell uses FA4 instead.
        if major != 9:
            return None
        import os, sys
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        result = get_kernel('varunneal/flash-attention-3').flash_attn_interface
        return result
    except Exception as e:
        import sys, traceback
        print(f"[flash_attention] FA3 load failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


_fa4 = _load_flash_attention_4()
_fa3 = _load_flash_attention_3() if _fa4 is None else None

HAS_FA4 = _fa4 is not None
HAS_FA3 = _fa3 is not None

# Backward-compat alias: anything that imported HAS_FA3 as a proxy for
# "do we have a fast kernel?" will still get True on Hopper.
# On Blackwell it will be False (HAS_FA3 is False), but HAS_FA4 is True.

# Override for testing: set to 'fa4', 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_backend():
    """Return one of 'fa4', 'fa3', 'sdpa' based on availability, override, and dtype."""
    from nanochat.common import COMPUTE_DTYPE

    if _override_impl == 'fa4':
        assert HAS_FA4, "Cannot override to FA4: not available on this hardware"
        return 'fa4'
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return 'fa3'
    if _override_impl == 'sdpa':
        return 'sdpa'

    # FA4 and FA3 only support bf16 (and fp8 internally); fp16/fp32 → SDPA
    if HAS_FA4:
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa4'
        return 'sdpa'
    if HAS_FA3:
        if COMPUTE_DTYPE == torch.bfloat16:
            return 'fa3'
        return 'sdpa'
    return 'sdpa'


_BACKEND = _resolve_backend()

# Convenience booleans used by the rest of the codebase
USE_FA4 = _BACKEND == 'fa4'
USE_FA3 = _BACKEND == 'fa3'     # kept for backward-compat with existing imports


def _resolve_use_fa3():
    global _BACKEND, USE_FA4, USE_FA3
    _BACKEND = _resolve_backend()
    USE_FA4 = _BACKEND == 'fa4'
    USE_FA3 = _BACKEND == 'fa3'
    return USE_FA3


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        # Ensure dtypes match to prevent "Expected query, key, and value to have the same dtype" RuntimeError
        if q.dtype != k.dtype or q.dtype != v.dtype:
            q = q.to(k.dtype)
            v = v.to(k.dtype)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        if q.dtype != k.dtype or q.dtype != v.dtype:
            q = q.to(k.dtype)
            v = v.to(k.dtype)
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    if q.dtype != k.dtype or q.dtype != v.dtype:
        q = q.to(k.dtype)
        v = v.to(k.dtype)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3 / FA4
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if _BACKEND == 'fa4':
        # FA4 requires head_dim >= 32 (smaller dims overflow shared memory on sm_100a).
        # Fall through to SDPA for tiny head dims (e.g. MST sub-transformers with d=64, n_head=4).
        head_dim = q.shape[-1]
        if head_dim >= 32:
            try:
                result = _fa4.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
                # FA4 returns (out, lse) tuple; FA3/SDPA return just out — normalise here.
                return result[0] if isinstance(result, tuple) else result
            except RuntimeError as e:
                if 'shared memory' in str(e).lower() or 'cudaErrorInvalidValue' in str(e):
                    import sys
                    print(f"[flash_attention] FA4 CUDA error (falling back to SDPA): {e}", file=sys.stderr)
                else:
                    raise
    if _BACKEND == 'fa3':
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA4/FA3 update k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    # FA4 does not yet implement flash_attn_with_kvcache — handled below by SDPA.
    if _BACKEND == 'fa3':
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
    # FA4 does not yet implement flash_attn_with_kvcache — fall through to SDPA.
    # Training still uses FA4's flash_attn_func (the hot path), so this is fine.

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA4/FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA4 / FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
