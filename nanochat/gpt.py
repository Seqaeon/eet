"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (optionally combined with learned absolute positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    dropout: float = 0.0
    use_pos_embed: bool = False
    window_pattern: str = "SSSSL"

    # EET: Early Exit Transformer
    use_eet: bool = False
    eet_frozen_kv: bool = True
    eet_reenter_final: bool = False
    eet_router_type: str = 'mlp2'
    eet_router_hidden: int = 0
    eet_freq_prior_alpha: float = 0.0
    eet_pos_prior_beta: float = 0.0
    eet_domain_prior: bool = False
    eet_warmup_frac: float = 0.02
    eet_explore_frac: float = 0.15
    eet_reconstruct_lambda: float = 1.0
    eet_efficiency_lambda_start: float = 0.01
    eet_efficiency_lambda_end: float = 0.1
    eet_translator_rank: int = 0
    eet_max_frozen_kv_frac: float = 0.75
    eet_exit_threshold: float = 0.5
    eet_min_exit_layer: int = 1
    eet_loss_variant: str = 'reconstruct'
    eet_topk_vocab: int = 512
    eet_entropy_lambda: float = 0.3
    eet_surprise_lambda: float = 0.1
    eet_adv_lambda: float = 1.0
    eet_adv_entropy_lambda: float = 0.2
    eet_quality_lambda: float = 1.0
    eet_quality_entropy_bonus: float = 0.1
    eet_gumbel_temp_start: float = 0.0
    eet_gumbel_temp_end: float = 0.0
    eet_gumbel_hard: int = 1
    eet_commitment_beta: float = 0.1
    eet_global_router: bool = False
    eet_freq_efficiency_alpha: float = 0.0
    eet_diversity_lambda: float = 0.0
    eet_ce_guided_lambda: float = 1.0
    eet_router_lr_mult: float = 5.0
    eet_model_lr_mult: float = 1.0
    eet_depth_weight_type: str = 'none'
    eet_depth_weight_max: float = 2.5
    eet_use_override: int = 0
    eet_override_prob_start: float = 0.5
    eet_override_prob_end: float = 0.1
    eet_compute_skip: bool = False
    eet_target_active_frac: float = 0.125
    eet_capacity_schedule: str = 'bell'
    eet_exit_fracs: list[float] | None = None
    eet_capacity_alignment_lambda: float = 0.0
    eet_router_task_grad: bool = True
    eet_reinforce_interval: int = 0
    eet_reinforce_lambda: float = 0.1
    eet_exit_adapter_rank: int = 0
    eet_router_after_block: int = 0
    eet_ffn_skip: bool = False
    eet_ffn_target_frac: float = 0.50
    eet_ffn_full_attn: bool = True
    eet_depth_affine: bool = False
    eet_capacity_anneal_frac: float = 0.0
    eet_learned_schedule: bool = False
    eet_departure_summary: bool = False
    eet_route_consistency_lambda: float = 0.0
    eet_dense_distill_interval: int = 0
    eet_dense_distill_lambda: float = 0.5
    eet_depth_lr_scale: bool = False
    eet_depth_grad_scale: bool = False
    eet_detach_aux_from_backbone: bool = False
    eet_detach_exit_from_backbone: bool = False

    # Ignored legacy / research fields (defined to prevent argument errors)
    use_moe: bool = False
    use_perm: bool = False
    moe_num_experts: int = 8
    moe_router_dim: int = 64
    moe_embed_dim: int = 64
    use_remix_linear: bool = False
    remix_context_dim: int = 64
    remix_context_dim_ratio: int = 6
    remix_basis_size: int = 0
    remix_output_gate_rank: int = 16
    remixed_linear_kwargs: dict | None = None
    moe_use_abs_pos_embed: bool = False
    use_layer_context: bool = True
    scale_basis_size: bool = True
    remix_use_dual_gate: bool = False
    p26_output_gated_linear: int = 0
    p28_shared_basis: int = 0
    p28_chunk_routing_size: int = 0
    p28_global_template_bank: str = 'none'
    p28_attn_proj_templates: int = 0
    p28_attn_qk_templates: int = 0
    perm_expert_mode: str = 'low_rank'
    perm_rank: int = 16
    router_context_window: int = -1
    router_causal: bool = True
    router_num_heads: int = 4
    router_num_queries: int = 16
    router_n_layers: int = 2
    router_use_vocab_prior: bool = False
    cclblock_modulation: str = 'weight'
    cclblock_orth_lambda: float = 0.0
    cclblock_context_stream: str = 'local'
    cclblock_ema_factor: float = 0.99
    cclblock_stale_ctx_lag: int = 0
    cclblock_sparse_gate_k: int = 0
    cclblock_gate_temperature: float = 1.0
    cclblock_context_bank_size: int = 0
    cclblock_per_head_ctx: bool = False
    cclblock_context_source: str = 'norm_x'
    cclblock_chunk_size: int = 0
    cclblock_aux_objective: str = 'none'
    cclblock_aux_lambda: float = 0.1
    cclblock_boundary_token_id: int = 198
    use_ral: bool = False
    ral_rank: int = 32
    cclblock_film_gate: bool = False
    cclblock_attn_shadow_dim: int = 0
    cclblock_dynamic_ratio: float = 0.25
    cclblock_gate_rank: int = 8
    cclblock_num_regimes: int = 8
    cclblock_regime_temperature: float = 1.0
    cclblock_poly_order: int = 2
    cclblock_lie_generators: int = 4
    cclblock_grassmann_bank_size: int = 4
    cclblock_tucker_rank: int = 32
    cclblock_tucker_modes: int = 8
    cclblock_svs_rank: int = 64
    cclblock_svs_eps: float = 0.1
    cclblock_vq_codes: int = 8
    cclblock_vq_temperature: float = 1.0
    cclblock_dcu_warmup_steps: int = 0
    cclblock_fsi_rotations: int = 8
    cclblock_fsi_selector_dim: int = 64
    cclblock_aesp_strata: int = 4
    cclblock_aesp_delta_rank: int = 4
    cclblock_ckr_branches: int = 4
    cclblock_ckr_kernel_size: int = 64
    cclblock_ckr_pos_channels: int = 1
    cclblock_ckr_dual_optim: int = 0
    cclblock_ckr_content_bias: float = 0.0
    cclblock_giad_rank: int = 32
    cclblock_psg_kernel_size: int = 64
    cclblock_ss_dynamic_ratio: float = 0.25
    cclblock_ss_branches: int = 2
    cclblock_ss_kernel_size: int = 64
    cclblock_lokr_branches: int = 8
    cclblock_lokr_rank: int = 16
    cclblock_ckr_temp_start: float = 2.0
    cclblock_ckr_temp_end: float = 0.3
    cclblock_com_kernel_size: int = 32
    cclblock_ckr_ortho_init: int = 0
    cclblock_ckr_branch_dropout: float = 0.0
    cclblock_ckr_diversity_lambda: float = 0.0
    cclblock_ckr_layer_selective: int = 0
    cclblock_pgr_kernel_size: int = 64
    cclblock_cil_kernel_size: int = 64
    cclblock_prb_kernel_size: int = 64
    p18_layer_drop: float = 0.0
    p18_dynamic_activation: int = 0
    p18_mixture_norm: int = 0
    p18_causal_attn_bias: int = 0
    p18_aux_sim_lambda: float = 0.0
    p18_gradient_penalty: float = 0.0
    p18_per_channel_scale: int = 0
    p19_residual_gate: int = 0
    p19_head_importance: int = 0
    p19_residual_mix_groups: int = 0
    p19_attn_logit_bias: int = 0
    p19_residual_decay: int = 0
    p19_grad_equilibrium: float = 0.0
    p19_spectral_reparam: int = 0
    p19_weight_anticollapse: float = 0.0
    p19_ve_bias: int = 0
    p19_weight_noise: float = 0.0
    p20_hrcs_scale: int = 0
    p20_lswr_scale: int = 0
    p20_lswr_planes: int = 8
    p20_lrcfb_branches: int = 0
    p20_lrcfb_narrow: int = 0
    p20_lrcfb_learned: int = 0
    p20_lrcfb_topk: int = 0
    p20_dgcr_branches: int = 0
    p20_dgcr_aux_weight: float = 0.01
    p20_mone_experts: int = 0
    p20_mone_topk: int = 0
    p20_mone_narrow: int = 1
    p20_mone_frozen: int = 0
    p20_ncea_branches: int = 0
    p20_ncea_eps: float = 0.1
    p20_adwi: int = 0
    p20_pwu_branches: int = 0
    p20_pwu_phase: int = 1
    p20_fsvd_gate: int = 0
    p20_wbfc_clusters: int = 0
    p20_wbfc_active: int = 0
    p21_per_experts: int = 0
    p21_per_topk: int = 0
    p21_per_learned: int = 0
    p21_per_attn: int = 0
    p22_attn_moe_route: str = 'none'
    p23_tiny_expert: int = 0
    p23_n_experts: int = 64
    p23_topk: int = 16
    p23_learned_route: int = 0
    p23_std_moe_experts: int = 0
    p23_std_moe_topk: int = 1
    p23_std_moe_aux_weight: float = 0.01
    p23_lokr: int = 0
    p23_lokr_rank: int = 4
    p23_use_shared_block_router: int = 0
    p23_linear_moe_experts: int = 0
    p23_linear_moe_topk: int = 0
    p23_quantile_route: int = 0
    remix_shared_context_gates: int = 0
    p24_use_sliced_weight: int = 0
    p24_sliced_weight_reduction_scale: int = 8
    p24_sliced_weight_min_select: int = 128
    p24_sliced_weight_scope: str = "per_token"
    p24_sliced_weight_balance_coeff: float = 0.01
    p24_quantile_route: int = 0
    p24_use_folded_mod: int = 0
    p24_folded_mod_reduction_scale: int = 8
    p24_folded_mod_min_dim: int = 128
    p24_folded_mod_scope: str = "per_layer"
    p24_folded_mod_gate_act: str = "sigmoid"
    p24_use_sequence_gated_linear: int = 0
    p24_sequence_gated_scope: str = "per_layer"
    p24_sequence_gated_act: str = "sigmoid"
    remix_disable_ln_basis: int = 0
    dense_intermediate_ln: int = 0
    use_mst: bool = False
    mst_n_subs: int = 8
    mst_sub_dim: int = 64
    mst_head_dim: int = 0
    mst_input_mode: str = 'fixed_slice'
    mst_rotated_slice_learned: bool = False
    mst_routing_mode: str = 'soft_weighted'
    mst_routing_topk: int = 4
    mst_routing_aux_weight: float = 0.01
    mst_diversity_weight: float = 0.0
    mst_ffn_mode: str = 'standard'
    mst_transition_mode: str = 'parallel'
    mst_final_mode: str = 'aggregate_proj'
    mst_final_topk: int = -1
    mst_ffn_shared_up: int = 0
    mst_ffn_inner_dim: int = 0
    mst_sub_dropout: float = 0.0
    mst_transition_every: int = 1
    mst_ffa_temperature: float = 1.0
    mst_global_residual: int = 0
    mst_hybrid_dense: int = 0
    mst_cross_sub_kv: int = 0
    mst_sub_aux_weight: float = 0.0
    mst_progressive_merge: int = 0
    mst_multi_scale_windows: int = 0
    mst_delta_residual: int = 0
    mst_sub_layers: int = 1
    mst_grad_equalize: int = 0
    mst_block_diagonal_muon: int = 0
    mst_transition_width_mult: float = 1.0
    mst_sub_lr_scale: float = 1.0
    mst_shared_expert: int = 0
    mst_router_entropy_weight: float = 0.0
    mst_shared_kv_attn: int = 0
    mst_contrastive_diversity_weight: float = 0.0
    mst_transition_nonlinear: int = 0
    mst_transition_gated: int = 0
    mst_transition_mlp: int = 0
    mst_cross_sub_gate: int = 0
    mst_hyper_connect: int = 0
    mst_cross_kv_inject: int = 0
    mst_slice_transition: int = 0
    mst_lookback_layers: int = 0
    mst_bilinear_transition: int = 0
    mst_cross_sub_qmod: int = 0
    mst_feature_cycle: int = 0
    mst_mean_transition: int = 0

RESEARCH_ALLOWED_KEYS = set() # empty, not strictly validated or check config attributes

def norm(x):
    return F.rms_norm(x, (x.size(-1),)).to(x.dtype)

class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = min(self.n_embd, 32)
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, token_active=None, eet_frozen_kv=False, frozen_k=None, frozen_v=None):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if token_active is not None and eet_frozen_kv and frozen_k is not None and frozen_v is not None:
            active_mask = token_active.unsqueeze(-1).unsqueeze(-1)
            k = torch.where(active_mask, k, frozen_k)
            v = torch.where(active_mask, v, frozen_v)

        self._last_k = k
        self._last_v = v

        if kv_cache is None:
            if token_active is not None:
                causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                active_queries = token_active.unsqueeze(1).unsqueeze(3)
                
                if not eet_frozen_kv:
                    active_keys = token_active.unsqueeze(1).unsqueeze(2)
                    combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) & active_keys & active_queries
                else:
                    combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) & active_queries
                
                y = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    attn_mask=combined_mask
                ).transpose(1, 2)
            else:
                y = flash_attn.flash_attn_func(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q.to(torch.bfloat16), k_cache, v_cache,
                k=k.to(torch.bfloat16), v=v.to(torch.bfloat16),
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, token_active=None, eet_frozen_kv=False, frozen_k=None, frozen_v=None):
        attn_out = self.attn(norm(x), ve, cos_sin, window_size, kv_cache,
                             token_active=token_active, eet_frozen_kv=eet_frozen_kv,
                             frozen_k=frozen_k, frozen_v=frozen_v)
        x = x + attn_out
        block_out = self.mlp(norm(x))
        x = x + block_out
        return x

    def forward_attn_only(self, x, ve, cos_sin, window_size, kv_cache):
        attn_out = self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        return x + attn_out

    def forward_ffn_only(self, x):
        block_out = self.mlp(norm(x))
        return x + block_out

class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        # Pad vocab size to a multiple of pad_vocab_size_to for better GPU performance
        padded_vocab_size = math.ceil(config.vocab_size / pad_vocab_size_to) * pad_vocab_size_to

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(padded_vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        ))
        if config.use_pos_embed:
            self.transformer.update(dict(
                wpe = nn.Embedding(config.sequence_len, config.n_embd)
            ))

        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})

        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.embedding_model = None
        self._use_residual_decay = False
        self.depth_decay_raw = None
        self.residual_mixers = None

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if "wpe" in self.transformer:
            torch.nn.init.normal_(self.transformer.wpe.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=200000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = 256
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    @property
    def max_seq_len(self):
        return self.config.sequence_len

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        wpe_numel = self.transformer.wpe.weight.numel() if "wpe" in self.transformer else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + wpe_numel + value_embeds_numel +
                           self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        total_flops = 6 * (nparams - nparams_exclude) + attn_flops
        return total_flops, total_flops, nparams

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = sum(p.numel() for p in self.transformer.wpe.parameters()) if "wpe" in self.transformer else 0
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + wpe + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'wpe': wpe,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'research': 0,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5, disable_mu_p=False, mu_p_scale_override=-1.0, gate_lr_scale=0.3):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        struct_matrix_params = []
        struct_adamw_params = []
        eet_router_matrix_params = []
        eet_router_adamw_params = []

        for name, p in self.named_parameters():
            if "eet_router" in name:
                (eet_router_matrix_params if p.ndim == 2 else eet_router_adamw_params).append(p)
            elif name.startswith("resid_lambdas") or name.startswith("x0_lambdas"):
                pass # special scalars
            elif name.startswith("transformer.wte") or name.startswith("transformer.wpe"):
                pass # embeddings
            elif name.startswith("lm_head"):
                pass # lm_head
            elif "value_embeds" in name:
                pass # value embeddings
            else:
                if p.ndim == 2:
                    struct_matrix_params.append(p)
                else:
                    struct_adamw_params.append(p)

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        if "wpe" in self.transformer:
            embedding_params += list(self.transformer.wpe.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        all_params = (struct_matrix_params + struct_adamw_params +
                      embedding_params + lm_head_params + value_embeds_params +
                      resid_params + x0_params +
                      eet_router_matrix_params + eet_router_adamw_params)

        covered_ids = {id(p) for p in all_params}
        orphan_params = [p for p in self.parameters() if id(p) not in covered_ids]
        for p in orphan_params:
            (struct_matrix_params if p.ndim == 2 else struct_adamw_params).append(p)

        all_params = (struct_matrix_params + struct_adamw_params +
                      embedding_params + lm_head_params + value_embeds_params +
                      resid_params + x0_params +
                      eet_router_matrix_params + eet_router_adamw_params)

        assert len(list(self.parameters())) == len(all_params), "Parameter count mismatch"

        if mu_p_scale_override > 0.0:
            dmodel_lr_scale = mu_p_scale_override
        elif disable_mu_p:
            dmodel_lr_scale = 1.0
        else:
            dmodel_lr_scale = (model_dim / 768) ** -0.5

        _eet_model_lr_mult = getattr(self.config, 'eet_model_lr_mult', 1.0)
        _orig_matrix_lr = matrix_lr
        if _eet_model_lr_mult != 1.0 and getattr(self.config, 'use_eet', False):
            matrix_lr *= _eet_model_lr_mult
            unembedding_lr *= _eet_model_lr_mult
            embedding_lr *= _eet_model_lr_mult
            scalar_lr *= _eet_model_lr_mult

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=struct_adamw_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        # Handle depth scaling if requested
        _use_depth_lr = getattr(self.config, 'eet_depth_lr_scale', False) and getattr(self.config, 'use_eet', False)
        if _use_depth_lr:
            import math as _math
            _n_layer = self.config.n_layer
            _min_exit = getattr(self.config, 'eet_min_exit_layer', 1)
            _n_rl = _n_layer - 1 - _min_exit
            _target_frac = getattr(self.config, 'eet_target_active_frac', 0.125)
            _schedule = getattr(self.config, 'eet_capacity_schedule', 'bell')
            if _schedule == 'uniform':
                _exit_fracs = [(1.0 - _target_frac) / _n_rl] * _n_rl
            elif _schedule == 'linear':
                _ws = [k + 1 for k in range(_n_rl)]
                _tw = sum(_ws)
                _exit_fracs = [w / _tw * (1.0 - _target_frac) for w in _ws]
            elif _schedule == 'bell':
                _mid = (_n_rl - 1) / 2.0
                _sigma = max(1.0, _n_rl / 4.0)
                _ws = [_math.exp(-((_k - _mid) / _sigma) ** 2) for _k in range(_n_rl)]
                _tw = sum(_ws)
                _exit_fracs = [w / _tw * (1.0 - _target_frac) for w in _ws]
            else:
                _psf = 1.0 - _target_frac ** (1.0 / _n_rl)
                _exit_fracs = [_psf] * _n_rl

            _survivor = [1.0] * _n_layer
            _surv = 1.0
            _rl_idx = 0
            for layer_i in range(_n_layer):
                _survivor[layer_i] = _surv
                if layer_i in range(_min_exit, _n_layer - 1) and _rl_idx < len(_exit_fracs):
                    _surv -= _exit_fracs[_rl_idx]
                    _rl_idx += 1
            _layer_lr_scale = [min(10.0, 1.0 / max(s, 0.05)) for s in _survivor]

            _layer_param_ids = {}
            for layer_i, block in enumerate(self.transformer.h):
                for p in block.parameters():
                    _layer_param_ids[id(p)] = layer_i

            _depth_grouped = {}
            _non_layer_params = []
            for p in struct_matrix_params:
                li = _layer_param_ids.get(id(p))
                if li is not None:
                    key = (li, p.shape)
                    _depth_grouped.setdefault(key, []).append(p)
                else:
                    _non_layer_params.append(p)

            for (li, shape), gp in sorted(_depth_grouped.items()):
                param_groups.append(dict(
                    kind='muon', params=gp, lr=matrix_lr * _layer_lr_scale[li],
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
            if _non_layer_params:
                for shape in sorted({p.shape for p in _non_layer_params}):
                    gp = [p for p in _non_layer_params if p.shape == shape]
                    param_groups.append(dict(
                        kind='muon', params=gp, lr=matrix_lr,
                        momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                    ))
        else:
            for shape in sorted({p.shape for p in struct_matrix_params}):
                group_params = [p for p in struct_matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))

        eet_router_lr_mult = getattr(self.config, 'eet_router_lr_mult', 5.0)
        eet_router_lr = _orig_matrix_lr * gate_lr_scale * eet_router_lr_mult
        if eet_router_matrix_params:
            for shape in sorted({p.shape for p in eet_router_matrix_params}):
                group_params = [p for p in eet_router_matrix_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=eet_router_lr,
                    momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                ))
        if eet_router_adamw_params:
            param_groups.append(dict(
                kind='adamw', params=eet_router_adamw_params, lr=eet_router_lr,
                betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        T_total = T0 + T

        if T_total > self.cos.size(1):
            new_len = max(T_total, self.cos.size(1) * 2)
            head_dim = self.config.n_embd // self.config.n_head
            cos, sin = self._precompute_rotary_embeddings(new_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total]

        x = self.transformer.wte(idx)
        if "wpe" in self.transformer:
            positions = torch.arange(T0, T_total, device=idx.device)
            x = x + self.transformer.wpe(positions)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)

        x = norm(x)
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = 20 * torch.tanh(logits / 20)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            cos_sin = self.cos[:, :ids.size(1)], self.sin[:, :ids.size(1)]
            x = self.transformer.wte(ids)
            if "wpe" in self.transformer:
                positions = torch.arange(0, ids.size(1), device=device)
                x = x + self.transformer.wpe(positions)
            x = x.to(COMPUTE_DTYPE)
            x = norm(x)
            x0 = x
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](ids).to(x.dtype) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], None)
            x = norm(x)
            logits = self.lm_head(x[:, -1])
            logits = logits[..., :self.config.vocab_size]
            logits = logits.float()
            logits = 20 * torch.tanh(logits / 20)
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=rng)
            ids = torch.cat([ids, next_token], dim=1)
        return ids[0].tolist()

# Compatibility placeholders to prevent import errors in legacy scripts
class RemixedLinear: pass
class DualGateLinear: pass
class CausalKernelLinear: pass
class ModulationDiagnostics: pass

