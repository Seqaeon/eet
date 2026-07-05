import re

file_path = "/home/seqaeon/.gemini/antigravity/brain/f4802947-cfa1-487e-813c-206f14ae6c9c/remixed_linear_formula.md"
with open(file_path, "r") as f:
    content = f.read()

clean_code = """```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class RemixedLinear(nn.Module):
    \"\"\"
    RemixedLinear: A factorized linear layer with basis, operator modulation, and output gating.
    \"\"\"
    def __init__(self, in_features, out_features, context_dim, basis_size=64, remixed_linear_kwargs=None, scale_basis=True, film_gate=False, routing_scope='per_sequence'):
        super().__init__()
        if remixed_linear_kwargs is None:
            remixed_linear_kwargs = {}
        
        if scale_basis:
            basis_size = max(basis_size, min(in_features, out_features) // 4)
        
        self.in_features = in_features
        self.out_features = out_features
        self.basis_size = basis_size
        self.use_context = remixed_linear_kwargs.get('use_context', True)
        self.use_basis_gate = remixed_linear_kwargs.get('use_basis_gate', True)
        self.use_output_gate = remixed_linear_kwargs.get('use_output_gate', True)
        self.gate_temperature = remixed_linear_kwargs.get('gate_temperature', 1.0)
        self.sparse_gate_k = remixed_linear_kwargs.get('sparse_gate_k', 0)
        self.basis_gate_mode = remixed_linear_kwargs.get('basis_gate_mode', 'mlp')
        self.operator_modulation = remixed_linear_kwargs.get('operator_modulation', 'none')
        self._film_gate_flag = film_gate

        # ── Structural Projections ───────────────────────────────────────────
        self.basis = Linear(in_features, basis_size, bias=False)
        self.ln_basis = nn.LayerNorm(basis_size)
        self.template_mixing = nn.Parameter(torch.empty(out_features, basis_size))
        nn.init.normal_(self.template_mixing, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ── Operator Modulation Params ───────────────────────────────────────
        if self.use_context:
            if self.operator_modulation == 'householder':
                self.householder_v = Linear(context_dim, basis_size, bias=False)
            elif self.operator_modulation == 'ckr':
                # Simplified representation of CKR inside the block
                self.ckr_gate = Linear(context_dim, basis_size, bias=True)
            elif self.operator_modulation == 'spectral':
                self.spectral_scale = Linear(context_dim, basis_size, bias=True)

        # ── Basis Gate ───────────────────────────────────────────────────────
        if self.use_context and self.use_basis_gate:
            _gate_out_size = basis_size * 2 if self._film_gate_flag else basis_size
            if self.basis_gate_mode == 'mlp':
                self.basis_modulator = nn.Sequential(
                    Linear(context_dim, context_dim // 2, bias=False),
                    nn.SiLU(),
                    Linear(context_dim // 2, _gate_out_size, bias=True)
                )
                nn.init.zeros_(self.basis_modulator[-1].weight)
                nn.init.zeros_(self.basis_modulator[-1].bias)
            elif self.basis_gate_mode == 'linear':
                self.basis_modulator = Linear(context_dim, _gate_out_size, bias=True)
                nn.init.zeros_(self.basis_modulator.weight)
                nn.init.zeros_(self.basis_modulator.bias)
            self.basis_gate_content = None
            self.basis_gate_context = None

        # ── Output Gate (Low-Rank) ───────────────────────────────────────────
        if self.use_context:
            r = remixed_linear_kwargs.get('output_gate_rank', 8)
            self.output_gate_coeffs = Linear(context_dim, r, bias=True)
            self.output_gate_basis  = nn.Parameter(torch.zeros(r, out_features))
            self.output_gate_scale  = nn.Parameter(torch.ones(1) * 0.1)

    def gate_parameters(self):
        \"\"\"Yield gate-specific parameters for lower-LR optimizer group.\"\"\"
        if self.use_context:
            if self.use_basis_gate and hasattr(self, 'basis_modulator') and self.basis_modulator is not None:
                yield from self.basis_modulator.parameters()
            yield self.output_gate_coeffs.weight
            if self.output_gate_coeffs.bias is not None:
                yield self.output_gate_coeffs.bias
            yield self.output_gate_basis
            yield self.output_gate_scale
            if self.operator_modulation == 'householder':
                yield from self.householder_v.parameters()
            elif self.operator_modulation == 'ckr':
                yield from self.ckr_gate.parameters()
            elif self.operator_modulation == 'spectral':
                yield from self.spectral_scale.parameters()

    def non_gate_parameters(self):
        \"\"\"Yield structural parameters for Muon/normal-LR group.\"\"\"
        yield self.basis.weight
        yield self.template_mixing
        yield self.bias

    def forward(self, x, context_state, route_weights=None, context_gates=None, **kwargs):
        dtype = x.dtype
        # ── Step 1: Project to basis + normalise ──────────────────────────────
        h_basis = self.ln_basis(self.basis(x).to(dtype=self.ln_basis.weight.dtype)).to(dtype=dtype)

        if self.use_context and context_state is not None:
            ctx = context_state.to(dtype=dtype)
            
            # ── Step 1B: Operator Modulation ──────────────────────────────────
            if self.operator_modulation == 'householder':
                v = self.householder_v(ctx)
                v_norm_sq = torch.sum(v ** 2, dim=-1, keepdim=True) + 1e-6
                h_dot_v = torch.sum(h_basis * v, dim=-1, keepdim=True)
                h_basis = h_basis - 2 * (h_dot_v / v_norm_sq) * v
            elif self.operator_modulation == 'ckr':
                # Simplified CKR modulation path
                ckr_g = torch.sigmoid(self.ckr_gate(ctx))
                h_basis = h_basis * ckr_g
            elif self.operator_modulation == 'spectral':
                scale = 1.0 + torch.tanh(self.spectral_scale(ctx))
                h_basis = h_basis * scale

            # ── Step 2: Basis Gate ────────────────────────────────────────────
            if self.use_basis_gate:
                if context_gates is not None and 'basis_gate' in context_gates:
                    gate_logits = context_gates['basis_gate'].to(dtype=dtype)
                else:
                    gate_logits = self.basis_modulator(ctx)

                if self._film_gate_flag:
                    scale_logits, shift = gate_logits.chunk(2, dim=-1)
                    gate_basis = (1.0 + torch.tanh(scale_logits * 0.1)).to(dtype=dtype)
                    h_basis = h_basis * gate_basis + shift.to(dtype=dtype)
                    gate_basis = None
                else:
                    gate_basis = torch.sigmoid(gate_logits / self.gate_temperature).to(dtype=dtype)
            else:
                gate_basis = torch.ones_like(h_basis)

            # ── Step 3: Output Gate (Low-Rank) ────────────────────────────────
            if self.use_output_gate:
                if context_gates is not None and 'output_coeffs' in context_gates:
                    coeffs = context_gates['output_coeffs'].to(dtype=dtype)
                else:
                    coeffs = self.output_gate_coeffs(ctx)
                gate_logits = torch.matmul(coeffs, self.output_gate_basis.to(dtype=dtype))
                gate_out = 1.0 + torch.tanh(self.output_gate_scale.to(dtype=dtype) * gate_logits)
            else:
                gate_out = None
        else:
            gate_basis = torch.ones_like(h_basis)
            gate_out = None

        # ── Step 4: Apply Basis Gate ──────────────────────────────────────────
        if gate_basis is not None:
            h_gated = (h_basis * gate_basis).to(dtype=dtype)
        else:
            h_gated = h_basis

        # ── Step 5: Mix to Output Space ───────────────────────────────────────
        pre_output = F.linear(h_gated, self.template_mixing.to(dtype=dtype))

        # ── Step 6: Apply Output Gate + Bias ──────────────────────────────────
        if gate_out is not None:
            pre_output = pre_output * gate_out
        
        return pre_output + self.bias.to(dtype=dtype)
```"""

# Find the code block and replace it
new_content = re.sub(r'```python\n.*?(?=## Design Notes)', clean_code + '\n', content, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(new_content)
