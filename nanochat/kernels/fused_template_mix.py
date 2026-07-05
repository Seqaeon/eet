"""
Fused RemixedLinear template-mixed GEMM kernel in Triton (v2).

The key operation we're fusing:
    W_eff = Σ_k α_k T_k        # (num_chunks, d_out, B)  ← eliminated
    y     = h @ W_eff.T         # (num_chunks, chunk_size, d_out)

Into a single kernel:
    y[c, t, n] = Σ_b h[c,t,b] * Σ_k α[c,k] * T[k,n,b]

The fusion avoids materializing W_eff in HBM (the primary bottleneck).

Strategy:
    - Grid over (num_chunks, token_tiles, output_tiles)
    - For each output tile, accumulate over B in chunks (BLOCK_K)
    - For each B-tile: load h once, then for each template k:
        load T[k, rn, rb], scale by α[c,k], accumulate dot product
    - All accumulation in float32 to avoid bf16 precision loss
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_tmix_fwd(
        H_ptr,       # (num_chunks, chunk_size, B)
        T_ptr,       # (K, d_out, B) contiguous
        A_ptr,       # (num_chunks, K)
        Y_ptr,       # (num_chunks, chunk_size, d_out)
        num_chunks: tl.constexpr,
        chunk_size: tl.constexpr,
        d_out: tl.constexpr,
        B: tl.constexpr,
        K: tl.constexpr,
        stride_hc, stride_ht, stride_hb,
        stride_tk, stride_tn, stride_tb,
        stride_ac, stride_ak,
        stride_yc, stride_yt, stride_yn,
        BLOCK_M: tl.constexpr,   # tokens
        BLOCK_N: tl.constexpr,   # d_out
        BLOCK_B: tl.constexpr,   # B (reduction)
    ):
        """y[c,t,n] = Σ_b h[c,t,b] * Σ_k α[c,k] * T[k,n,b]"""
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # token indices
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # output indices
        mask_m = rm < chunk_size
        mask_n = rn < d_out

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Pre-load all K alpha values for this chunk (K is small, ~8)
        alpha_vals = tl.zeros((K,), dtype=tl.float32)
        for k in tl.static_range(K):
            alpha_vals = tl.where(
                tl.arange(0, K) == k,
                tl.load(A_ptr + pid_c * stride_ac + k * stride_ak).to(tl.float32),
                alpha_vals,
            )

        # Tile over B (reduction dimension)
        for b_start in range(0, B, BLOCK_B):
            rb = b_start + tl.arange(0, BLOCK_B)
            mask_b = rb < B

            # Load h[c, rm, rb] -> (BLOCK_M, BLOCK_B), cast to f32
            h_tile = tl.load(
                H_ptr + pid_c * stride_hc + rm[:, None] * stride_ht + rb[None, :] * stride_hb,
                mask=mask_m[:, None] & mask_b[None, :],
                other=0.0,
            ).to(tl.float32)

            # For each template, load T[k, rn, rb] and accumulate
            for k in tl.static_range(K):
                # Load alpha for this template
                a_k = tl.load(A_ptr + pid_c * stride_ac + k * stride_ak).to(tl.float32)

                # Load T[k, rn, rb] -> (BLOCK_N, BLOCK_B), cast to f32
                t_tile = tl.load(
                    T_ptr + k * stride_tk + rn[:, None] * stride_tn + rb[None, :] * stride_tb,
                    mask=mask_n[:, None] & mask_b[None, :],
                    other=0.0,
                ).to(tl.float32)

                # acc += a_k * (h_tile @ t_tile.T)
                # h_tile: (BLOCK_M, BLOCK_B)  t_tile: (BLOCK_N, BLOCK_B)
                # h_tile @ t_tile.T -> (BLOCK_M, BLOCK_N)
                acc += a_k * tl.dot(h_tile, tl.trans(t_tile))

        # Store result
        tl.store(
            Y_ptr + pid_c * stride_yc + rm[:, None] * stride_yt + rn[None, :] * stride_yn,
            acc.to(Y_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _fused_tmix_bwd_dh(
        GY_ptr,      # (num_chunks, chunk_size, d_out)
        T_ptr,       # (K, d_out, B)
        A_ptr,       # (num_chunks, K)
        GH_ptr,      # (num_chunks, chunk_size, B) — output
        num_chunks: tl.constexpr,
        chunk_size: tl.constexpr,
        d_out: tl.constexpr,
        B: tl.constexpr,
        K: tl.constexpr,
        stride_gyc, stride_gyt, stride_gyn,
        stride_tk, stride_tn, stride_tb,
        stride_ac, stride_ak,
        stride_ghc, stride_ght, stride_ghb,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,   # B (output of this kernel)
        BLOCK_K: tl.constexpr,   # d_out (reduction)
    ):
        """grad_h[c,t,b] = Σ_n (Σ_k α[c,k] * T[k,n,b]) * grad_y[c,t,n]"""
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # B indices
        mask_m = rm < chunk_size
        mask_n = rn < B

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for n_start in range(0, d_out, BLOCK_K):
            rk = n_start + tl.arange(0, BLOCK_K)
            mask_k = rk < d_out

            # Load grad_y[c, rm, rk] -> (BLOCK_M, BLOCK_K)
            gy_tile = tl.load(
                GY_ptr + pid_c * stride_gyc + rm[:, None] * stride_gyt + rk[None, :] * stride_gyn,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)

            for k in tl.static_range(K):
                a_k = tl.load(A_ptr + pid_c * stride_ac + k * stride_ak).to(tl.float32)

                # Load T[k, rk, rn] -> (BLOCK_K, BLOCK_N)
                t_tile = tl.load(
                    T_ptr + k * stride_tk + rk[:, None] * stride_tn + rn[None, :] * stride_tb,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                ).to(tl.float32)

                # acc += a_k * (gy_tile @ t_tile)
                acc += a_k * tl.dot(gy_tile, t_tile)

        tl.store(
            GH_ptr + pid_c * stride_ghc + rm[:, None] * stride_ght + rn[None, :] * stride_ghb,
            acc.to(GH_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


# ─── PyTorch wrappers ────────────────────────────────────────────────────────


def _naive_template_mix(h, T_bank, alpha):
    """Reference: y = (Σ_k α_k T_k) @ h^T per chunk, via explicit W_eff."""
    W_eff = torch.einsum('ck, knb -> cnb', alpha.float(), T_bank.float()).to(h.dtype)
    return torch.bmm(h, W_eff.transpose(1, 2))


class FusedTemplateMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, T_bank, alpha):
        assert HAS_TRITON, "Triton not available"
        num_chunks, chunk_size, B_dim = h.shape
        K, d_out, B2 = T_bank.shape
        assert B_dim == B2

        h = h.contiguous()
        T_bank = T_bank.contiguous()
        alpha = alpha.contiguous()

        y = torch.empty(num_chunks, chunk_size, d_out, device=h.device, dtype=h.dtype)

        # Block sizes tuned for H200 shared memory limit (232 KB).
        # Shared mem per stage ≈ (BLOCK_M*BLOCK_B + BLOCK_N*BLOCK_B)*2 + BLOCK_M*BLOCK_N*4
        BLOCK_M = min(32, chunk_size)
        BLOCK_N = min(64, d_out)
        BLOCK_B = min(64, B_dim)

        grid = (
            num_chunks,
            triton.cdiv(chunk_size, BLOCK_M),
            triton.cdiv(d_out, BLOCK_N),
        )

        _fused_tmix_fwd[grid](
            h, T_bank, alpha, y,
            num_chunks, chunk_size, d_out, B_dim, K,
            h.stride(0), h.stride(1), h.stride(2),
            T_bank.stride(0), T_bank.stride(1), T_bank.stride(2),
            alpha.stride(0), alpha.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_B=BLOCK_B,
            num_stages=1, num_warps=4,
        )

        ctx.save_for_backward(h, T_bank, alpha)
        ctx.block_sizes = (BLOCK_M, BLOCK_N, BLOCK_B)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        h, T_bank, alpha = ctx.saved_tensors
        num_chunks, chunk_size, B_dim = h.shape
        K, d_out, _ = T_bank.shape

        grad_y = grad_y.contiguous()

        # grad_h via fused kernel
        grad_h = torch.empty_like(h)
        BLOCK_M = min(32, chunk_size)
        BLOCK_N = min(64, B_dim)
        BLOCK_K = min(64, d_out)

        grid = (
            num_chunks,
            triton.cdiv(chunk_size, BLOCK_M),
            triton.cdiv(B_dim, BLOCK_N),
        )
        _fused_tmix_bwd_dh[grid](
            grad_y, T_bank, alpha, grad_h,
            num_chunks, chunk_size, d_out, B_dim, K,
            grad_y.stride(0), grad_y.stride(1), grad_y.stride(2),
            T_bank.stride(0), T_bank.stride(1), T_bank.stride(2),
            alpha.stride(0), alpha.stride(1),
            grad_h.stride(0), grad_h.stride(1), grad_h.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_stages=1, num_warps=4,
        )

        # grad_T and grad_alpha via PyTorch (not the bottleneck)
        grad_T = torch.einsum('ck, ctn, ctb -> knb', alpha.float(), grad_y.float(), h.float()).to(T_bank.dtype)
        grad_alpha = torch.einsum('ctn, knb, ctb -> ck', grad_y.float(), T_bank.float(), h.float()).to(alpha.dtype)

        return grad_h, grad_T, grad_alpha


def fused_template_mix(h, T_bank, alpha):
    """Drop-in replacement for naive template mixing."""
    if HAS_TRITON and h.is_cuda:
        return FusedTemplateMix.apply(h, T_bank, alpha)
    return _naive_template_mix(h, T_bank, alpha)


# ─── Benchmark ────────────────────────────────────────────────────────────────

def benchmark():
    import time
    device = 'cuda'
    dtype = torch.bfloat16

    configs = [
        # (name, num_chunks, chunk_size, B, d_out, K)
        ("d4",  512, 64, 256, 256, 8),
        ("d8",  256, 64, 512, 512, 8),
        ("d12", 128, 64, 768, 768, 8),
        # Real batch-size shapes (B=64, seq=2048 -> 32 chunks of 64)
        ("d12_real", 2048, 64, 768, 768, 8),   # batch=64 → 64*32=2048 chunks
    ]

    for name, nc, cs, B, d_out, K in configs:
        print(f"\n{'='*70}")
        print(f"Config: {name} — chunks={nc}, chunk_size={cs}, B={B}, d_out={d_out}, K={K}")
        print(f"  W_eff size if materialized: {nc * d_out * B * 2 / 1e6:.1f} MB")
        print(f"{'='*70}")

        h = torch.randn(nc, cs, B, device=device, dtype=dtype)
        T_bank = torch.randn(K, d_out, B, device=device, dtype=dtype)
        alpha = torch.softmax(torch.randn(nc, K, device=device), dim=-1).to(dtype)

        # Correctness
        y_naive = _naive_template_mix(h, T_bank, alpha)
        if HAS_TRITON:
            y_fused = FusedTemplateMix.apply(h, T_bank, alpha)
            max_diff = (y_naive - y_fused).abs().max().item()
            mean_mag = y_naive.abs().mean().item()
            rel_err = max_diff / (mean_mag + 1e-8)
            print(f"  Max abs diff: {max_diff:.6f}")
            print(f"  Mean |y|:     {mean_mag:.4f}")
            print(f"  Rel error:    {rel_err:.6f}")

        # Timing
        warmup, measure = 20, 100

        for _ in range(warmup):
            _ = _naive_template_mix(h, T_bank, alpha)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(measure):
            _ = _naive_template_mix(h, T_bank, alpha)
        torch.cuda.synchronize()
        naive_ms = (time.perf_counter() - t0) / measure * 1000
        print(f"  Naive:  {naive_ms:.3f} ms")

        if HAS_TRITON:
            for _ in range(warmup):
                _ = FusedTemplateMix.apply(h, T_bank, alpha)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(measure):
                _ = FusedTemplateMix.apply(h, T_bank, alpha)
            torch.cuda.synchronize()
            fused_ms = (time.perf_counter() - t0) / measure * 1000
            print(f"  Fused:  {fused_ms:.3f} ms")
            print(f"  Speedup: {naive_ms / fused_ms:.2f}x")

            # Backward correctness
            h_n = h.clone().requires_grad_(True)
            T_n = T_bank.clone().requires_grad_(True)
            a_n = alpha.clone().requires_grad_(True)
            y_n = _naive_template_mix(h_n, T_n, a_n)
            y_n.sum().backward()

            h_f = h.clone().requires_grad_(True)
            T_f = T_bank.clone().requires_grad_(True)
            a_f = alpha.clone().requires_grad_(True)
            y_f = FusedTemplateMix.apply(h_f, T_f, a_f)
            y_f.sum().backward()

            print(f"  Bwd grad_h diff:     {(h_n.grad - h_f.grad).abs().max().item():.6f}")
            print(f"  Bwd grad_T diff:     {(T_n.grad - T_f.grad).abs().max().item():.6f}")
            print(f"  Bwd grad_alpha diff: {(a_n.grad - a_f.grad).abs().max().item():.6f}")


if __name__ == "__main__":
    benchmark()
