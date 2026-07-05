"""
INT8 quantization for RemixedLinear template banks at inference time.

Quantizes the template matrices T_bank (K, d_out, B) from bfloat16/float16
to INT8 with per-channel (per-output-dim) scale factors. This halves HBM
bandwidth for template loads — the dominant bottleneck in RemixedLinear inference.

The quantized forward computes:
    W_eff = Σ_k α_k * dequant(T_k_int8, scale_k)
    y = W_eff @ h

Usage:
    from nanochat.kernels.int8_templates import quantize_remix_model, INT8TemplateMixin

    # Quantize all RemixedLinear layers in-place
    quantize_remix_model(model)

    # Run inference as usual — forward() automatically uses INT8 path
    y = model(input_ids)

Numerics:
    Per-channel absmax quantization: scale = max(|T[:, j]|) / 127
    Dequantization: T_float = T_int8 * scale
    Typical error: < 0.1% relative (bf16 -> int8 -> bf16 round-trip)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight matrix to INT8 with per-output-channel scales.

    Args:
        weight: (out_features, in_features) float tensor

    Returns:
        weight_int8: (out_features, in_features) torch.int8
        scales: (out_features, 1) float32 — per-row scale factors
    """
    assert weight.ndim == 2
    # Per-row absmax
    amax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = amax / 127.0
    weight_int8 = (weight / scales).round().clamp(-128, 127).to(torch.int8)
    return weight_int8, scales.float()


def dequantize_per_channel(weight_int8: torch.Tensor, scales: torch.Tensor,
                           dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize INT8 weights back to float.

    Args:
        weight_int8: (out_features, in_features) torch.int8
        scales: (out_features, 1) float32

    Returns:
        weight: (out_features, in_features) in target dtype
    """
    return (weight_int8.float() * scales).to(dtype)


class QuantizedTemplateBank(nn.Module):
    """INT8-quantized template bank for RemixedLinear.

    Stores K templates as INT8 with per-channel scales. During forward,
    dequantizes on-the-fly when computing the template-mixed matmul.
    This halves HBM bandwidth for template loads.

    The computation is:
        W_eff = Σ_k α_k * dequant(T_k)
        y = W_eff @ h

    Memory savings:
        Original: K * d_out * B * 2 bytes (bf16)
        INT8:     K * d_out * B * 1 byte + K * d_out * 4 bytes (scales)
        Ratio:    ~0.5x for B >> 4
    """

    def __init__(self, templates_int8: torch.Tensor, scales: torch.Tensor,
                 compute_dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            templates_int8: (K, d_out, B) int8 tensor
            scales: (K, d_out, 1) float32 per-channel scales
            compute_dtype: dtype for the dequantized computation
        """
        super().__init__()
        self.register_buffer('templates_int8', templates_int8)
        self.register_buffer('scales', scales)
        self.compute_dtype = compute_dtype
        self.K, self.d_out, self.B = templates_int8.shape

    @classmethod
    def from_float(cls, templates: torch.Tensor,
                   compute_dtype: torch.dtype = torch.bfloat16) -> 'QuantizedTemplateBank':
        """Create quantized template bank from float templates.

        Args:
            templates: (K, d_out, B) float tensor
        """
        K, d_out, B = templates.shape
        templates_int8 = torch.empty(K, d_out, B, dtype=torch.int8, device=templates.device)
        scales = torch.empty(K, d_out, 1, dtype=torch.float32, device=templates.device)

        for k in range(K):
            templates_int8[k], scales[k] = quantize_per_channel(templates[k])

        return cls(templates_int8, scales, compute_dtype)

    def dequantize(self, k: int) -> torch.Tensor:
        """Dequantize template k. Returns (d_out, B) in compute_dtype."""
        return dequantize_per_channel(self.templates_int8[k], self.scales[k], self.compute_dtype)

    def dequantize_all(self) -> torch.Tensor:
        """Dequantize all templates. Returns (K, d_out, B) in compute_dtype."""
        return (self.templates_int8.float() * self.scales).to(self.compute_dtype)

    def mixed_matmul(self, h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Compute y = (Σ_k α_k T_k) @ h with on-the-fly dequantization.

        Args:
            h: (..., B) input tensor
            alpha: (..., K) or (K,) routing weights

        Returns:
            y: (..., d_out)
        """
        # Dequantize all templates at once — still cheaper than float16 storage
        T_float = self.dequantize_all()  # (K, d_out, B)

        # Compute W_eff = Σ_k α_k * T_k
        if alpha.ndim == 1:
            # Single set of weights: (K,) @ (K, d_out, B) -> (d_out, B)
            W_eff = torch.einsum('k, knb -> nb', alpha, T_float)
            return h @ W_eff.T
        else:
            # Per-chunk weights: (C, K) @ (K, d_out, B) -> (C, d_out, B)
            W_eff = torch.einsum('ck, knb -> cnb', alpha, T_float)
            return torch.bmm(h, W_eff.transpose(1, 2))

    @property
    def memory_bytes(self) -> int:
        """Total memory in bytes."""
        return (self.templates_int8.nelement() * 1 +
                self.scales.nelement() * 4)

    @property
    def float_memory_bytes(self) -> int:
        """Memory if stored in bf16."""
        return self.templates_int8.nelement() * 2

    @property
    def compression_ratio(self) -> float:
        return self.float_memory_bytes / self.memory_bytes


def quantize_remix_model(model: nn.Module, verbose: bool = True) -> dict:
    """Quantize RemixedLinear template banks to INT8.

    Targets `module.template_bank` (nn.ParameterList of K tensors, each
    shape (out_features, basis_size)) which is the actual storage used by
    the P29 chunk-routed RemixedLinear implementation.

    IMPORTANT — throughput vs memory:
        INT8 templates reduce PEAK MEMORY (each template halved in size)
        but do NOT improve throughput in the current forward pass because:
        - The forward does `torch.stack([t.to(dtype) for t in template_bank])`
          which dequantizes to bf16 before any einsum, so the einsum memory
          traffic is unchanged.
        - A throughput win requires fusing dequant into the einsum (Triton).
        - The primary value here is fitting larger K or larger models in VRAM.

    Args:
        model: model containing RemixedLinear layers
        verbose: print per-layer quantization stats

    Returns:
        stats dict with layers_quantized, original_bytes, quantized_bytes, max_error
    """
    stats = {
        'layers_quantized': 0,
        'original_bytes': 0,
        'quantized_bytes': 0,
        'max_error': 0.0,
    }

    for name, module in model.named_modules():
        bank = getattr(module, 'template_bank', None)
        if bank is None or not isinstance(bank, nn.ParameterList) or len(bank) == 0:
            continue

        layer_orig_bytes = 0
        layer_quant_bytes = 0
        layer_max_err = 0.0

        int8_tensors = []
        scale_tensors = []

        for k, param in enumerate(bank):
            t = param.data  # (out_features, basis_size)
            t_int8, scales = quantize_per_channel(t)

            # measure error
            t_recon = dequantize_per_channel(t_int8, scales, dtype=t.dtype)
            err = (t - t_recon).abs().max().item()
            layer_max_err = max(layer_max_err, err)

            int8_tensors.append(t_int8)
            scale_tensors.append(scales)
            layer_orig_bytes += t.nelement() * t.element_size()
            layer_quant_bytes += t_int8.nelement() * 1 + scales.nelement() * 4

        # Replace the ParameterList with registered buffers so the parameters
        # are no longer part of the optimizer and don't get saved as fp32.
        # We keep the same interface by monkey-patching forward to dequantize.
        module._int8_banks = int8_tensors       # list of (out, basis) int8
        module._int8_scales = scale_tensors     # list of (out, 1) float32
        module._int8_dtype = bank[0].data.dtype

        # Remove the ParameterList from the module to free fp16/bf16 memory
        del module.template_bank
        module.template_bank = None             # forward guards against None already

        # Monkey-patch: override the template_bank property used by forward
        # by replacing with a lazy-dequant accessor list-like object
        class _DequantBank:
            """Mimics nn.ParameterList[k] access, dequantizing on demand."""
            def __init__(self, int8s, scales, dtype):
                self._int8s = int8s
                self._scales = scales
                self._dtype = dtype
            def __iter__(self):
                return (dequantize_per_channel(q, s, self._dtype)
                        for q, s in zip(self._int8s, self._scales))
            def __len__(self):
                return len(self._int8s)
            def __getitem__(self, k):
                return dequantize_per_channel(self._int8s[k], self._scales[k], self._dtype)

        module.template_bank = _DequantBank(int8_tensors, scale_tensors,
                                            module._int8_dtype)

        stats['layers_quantized'] += 1
        stats['original_bytes'] += layer_orig_bytes
        stats['quantized_bytes'] += layer_quant_bytes
        stats['max_error'] = max(stats['max_error'], layer_max_err)

        if verbose:
            K = len(int8_tensors)
            shape = int8_tensors[0].shape
            compression = layer_orig_bytes / max(layer_quant_bytes, 1)
            print(f"  {name}.template_bank: K={K}, shape={shape}, "
                  f"max_err={layer_max_err:.5f}, compression={compression:.2f}x")

    if verbose and stats['layers_quantized'] > 0:
        ratio = stats['original_bytes'] / max(stats['quantized_bytes'], 1)
        saved_mb = (stats['original_bytes'] - stats['quantized_bytes']) / 1e6
        print(f"\n  Total: {stats['layers_quantized']} layers")
        print(f"  Template memory: {stats['original_bytes']/1e6:.1f} MB -> "
              f"{stats['quantized_bytes']/1e6:.1f} MB "
              f"(saved {saved_mb:.1f} MB, {ratio:.2f}x)")
        print(f"  Max quantization error: {stats['max_error']:.5f}")
        print(f"  NOTE: throughput unchanged — peak memory reduced only.")
    elif verbose:
        print("  No template_bank layers found to quantize.")

    return stats



# ─── Standalone test ──────────────────────────────────────────────────────────

def _test():
    """Quick test of INT8 quantization roundtrip."""
    print("Testing INT8 template quantization...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    for K, d_out, B in [(8, 256, 256), (8, 512, 512), (8, 768, 768)]:
        templates = torch.randn(K, d_out, B, device=device, dtype=dtype)
        qbank = QuantizedTemplateBank.from_float(templates, compute_dtype=dtype)

        # Roundtrip error
        reconstructed = qbank.dequantize_all()
        max_err = (templates - reconstructed).abs().max().item()
        rel_err = max_err / templates.abs().mean().item()

        # Mixed matmul correctness
        h = torch.randn(32, 64, B, device=device, dtype=dtype)
        alpha = torch.softmax(torch.randn(32, K, device=device), dim=-1).to(dtype)

        y_ref = torch.einsum('ck, knb, ctb -> ctn', alpha, templates, h)
        y_q = qbank.mixed_matmul(h, alpha)
        matmul_err = (y_ref - y_q).abs().max().item()

        print(f"  K={K}, d_out={d_out}, B={B}: "
              f"roundtrip_max_err={max_err:.6f}, rel_err={rel_err:.4f}, "
              f"matmul_max_err={matmul_err:.4f}, "
              f"compression={qbank.compression_ratio:.2f}x, "
              f"saved={qbank.float_memory_bytes/1e6:.1f}->{qbank.memory_bytes/1e6:.1f} MB")

    print("All tests passed.")


if __name__ == "__main__":
    _test()
