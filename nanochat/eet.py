"""
Early Exit Transformer (EET) Architecture.

Prior-informed adaptive computation with exit-to-final-layer design.
Tokens can exit early based on semantic routing priors (frequency, POS, domain),
but ALL tokens are re-integrated at the final layer via frozen KV injection
(Option B) or masked attention (Option A).

Three-phase training:
  Phase 1 (Dense Warmup):   All tokens process all layers, router observes only.
  Phase 2 (Exploration):    Reconstruction loss + efficiency penalty, routing active.
  Phase 3 (Committed):      Fixed routing with increasing efficiency pressure.

Design grounded in Token Rank Stability (TRS) findings:
  - High-frequency tokens stabilize early (ρ = −0.32 to −0.45)
  - POS categories have consistent exit ordering
  - Code requires deeper processing than natural language
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.common import COMPUTE_DTYPE, print0, is_ddp_initialized
from nanochat.gpt import GPT, GPTConfig, Linear, Block, norm, has_ve
from torch.utils.checkpoint import checkpoint


def compute_layer_loss(h_k, p_k, targets, lm_head, config, eet_quality_lambda):
    """Checkpointed helper to calculate lm_head projection and cross entropy at layer k.
    Saves massive VRAM by avoiding storing large logits tensors in forward activations.
    """
    logits_k = lm_head(h_k)
    logits_k = logits_k[..., :config.vocab_size].float()
    logits_k = 20 * torch.tanh(logits_k / 20)
    
    loss_k = F.cross_entropy(
        logits_k.view(-1, logits_k.size(-1)),
        targets.view(-1),
        ignore_index=-1,
        reduction='none'
    ).view(h_k.size(0), h_k.size(1))
    
    combined_k = loss_k * p_k.detach() + eet_quality_lambda * (loss_k.detach() * p_k)
    return combined_k.mean(), (loss_k * p_k.detach()).mean(), (loss_k.detach() * p_k).mean()


def compute_efficiency_and_diversity(p_exits, n_exits, freq_bias, config, eet_lambda_e):
    """Compute per-token frequency-scaled efficiency loss + exit diversity pressure.
    
    Returns (total_loss, diagnostics_dict) where:
      - efficiency_loss: per-token weighted expected exit depth
      - diversity_loss: negative std of per-token expected exit (forces token differentiation)
      
    Memory efficient: only creates (B, T) intermediates, no vocab-sized tensors.
    """
    device = p_exits[0].device
    layer_indices = torch.arange(n_exits, device=device, dtype=torch.float32)
    
    # Compute per-token expected exit layer: (B, T)
    expected_exit = sum(
        p_exits[k] * layer_indices[k]
        for k in range(n_exits)
    )
    
    freq_alpha = getattr(config, 'eet_freq_efficiency_alpha', 0.0)
    diversity_lambda = getattr(config, 'eet_diversity_lambda', 0.0)
    
    # --- Per-token frequency-scaled efficiency ---
    if freq_alpha > 0 and freq_bias is not None:
        # freq_bias is (B, T), values in [0, 1] where 1 = most frequent
        # Scale: frequent tokens get (1 + alpha) weight, rare tokens get ~1.0 weight
        freq_weight = 1.0 + freq_alpha * freq_bias.float()  # (B, T)
        efficiency_loss = (expected_exit * freq_weight).mean() / n_exits
    else:
        efficiency_loss = expected_exit.mean() / n_exits
    
    total_loss = eet_lambda_e * efficiency_loss
    
    diag = {'efficiency': efficiency_loss, 'expected_exit': expected_exit.mean()}
    
    # --- Exit diversity pressure ---
    if diversity_lambda > 0:
        # Penalize low variance in expected exit depth across tokens
        # Negative std = we want to MAXIMIZE variance (minimize negative std)
        exit_std = expected_exit.std()
        diversity_loss = -exit_std
        total_loss = total_loss + diversity_lambda * diversity_loss
        diag['diversity'] = exit_std
    
    return total_loss, diag



# ---------------------------------------------------------------------------
# Routing Priors
# ---------------------------------------------------------------------------

class FrequencyPrior(nn.Module):
    """Log-frequency exit bias: common tokens → early exit, rare → late.

    Checks for pre-computed freq_table.pt in tokenizer_dir. If missing,
    computes from training data and caches.
    """

    def __init__(self, vocab_size: int, tokenizer_dir: str = None, device: str = 'cpu'):
        super().__init__()
        freq_table = self._load_or_compute(vocab_size, tokenizer_dir, device)
        # Normalize to [0, 1]: 1 = most frequent (early exit), 0 = rarest
        log_freq = torch.log1p(freq_table)
        log_freq = log_freq / (log_freq.max() + 1e-8)
        self.register_buffer('freq_bias', log_freq)  # (vocab_size,)

    def _load_or_compute(self, vocab_size: int, tokenizer_dir: str = None, device: str = 'cpu') -> torch.Tensor:
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

        cache_path = os.path.join(tokenizer_dir, "freq_table.pt")
        if os.path.exists(cache_path):
            print0(f"[EET] Loading frequency table from {cache_path}")
            return torch.load(cache_path, weights_only=True, map_location='cpu')

        print0(f"[EET] Computing frequency table from training data...")
        # Explicit device='cpu' to avoid meta device context contamination
        freq = torch.zeros(vocab_size, dtype=torch.float32, device='cpu')
        try:
            from nanochat.dataset import resolve_data_dir, list_parquet_files
            from nanochat.tokenizer import get_tokenizer
            import pyarrow.parquet as pq
            import numpy as np
            data_dir = resolve_data_dir()
            shards = list_parquet_files(data_dir=data_dir)
            tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
            
            for idx, shard_path in enumerate(shards):
                print0(f"[EET] Processing shard {idx+1}/{len(shards)}: {os.path.basename(shard_path)}")
                table = pq.read_table(shard_path, columns=["text"])
                texts = table.column("text").to_pylist()
                texts = [t for t in texts if t]
                
                # Batched parallel tokenization (tiktoken Rust implementation)
                all_token_ids_list = tokenizer.encode(texts)
                
                # Flatten and count in PyTorch on CPU
                if all_token_ids_list:
                    flat_np = np.concatenate([np.array(tokens, dtype=np.int32) for tokens in all_token_ids_list])
                    flat_tokens_t = torch.from_numpy(flat_np).long()
                    freq += torch.bincount(flat_tokens_t, minlength=vocab_size)[:vocab_size]
            
            freq_cpu = freq
            print0(f"[EET] Frequency table computed from {len(shards)} shards")
        except Exception as e:
            print0(f"[EET] Warning: could not compute freq table ({e}), using uniform")
            freq_cpu = torch.ones(vocab_size, dtype=torch.float32)

        os.makedirs(tokenizer_dir, exist_ok=True)
        torch.save(freq_cpu, cache_path)
        print0(f"[EET] Frequency table saved to {cache_path}")
        return freq_cpu

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns per-token frequency bias. Shape: (B, T)."""
        return self.freq_bias[input_ids]


class POSPrior(nn.Module):
    """POS-category exit bias: function words → early, content words → late.

    Category mapping (exit bias, higher = earlier exit):
      punctuation/special: 1.0
      function words:      0.8
      determiners/preps:   0.6
      adjectives/adverbs:  0.4
      content words:       0.2
      numbers/symbols:     0.1
    """

    POS_EXIT_SCORES = {
        'PUNCT': 1.0, 'SPACE': 1.0, 'SYM': 0.9,
        'DET': 0.8, 'CCONJ': 0.8, 'SCONJ': 0.8, 'ADP': 0.7, 'PART': 0.7,
        'PRON': 0.6, 'AUX': 0.6, 'INTJ': 0.6,
        'ADJ': 0.4, 'ADV': 0.4,
        'VERB': 0.3, 'NOUN': 0.2, 'PROPN': 0.2,
        'NUM': 0.1, 'X': 0.3,
    }

    def __init__(self, vocab_size: int, tokenizer_dir: str = None):
        super().__init__()
        pos_table = self._load_or_compute(vocab_size, tokenizer_dir)
        self.register_buffer('pos_bias', pos_table)  # (vocab_size,)

    def _load_or_compute(self, vocab_size: int, tokenizer_dir: str = None) -> torch.Tensor:
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

        cache_path = os.path.join(tokenizer_dir, "pos_categories.pt")
        if os.path.exists(cache_path):
            print0(f"[EET] Loading POS categories from {cache_path}")
            return torch.load(cache_path, weights_only=True, map_location='cpu')

        print0(f"[EET] Computing POS categories via spaCy...")
        # Explicit device='cpu' to avoid meta device context contamination
        pos_scores = torch.full((vocab_size,), 0.5, dtype=torch.float32, device='cpu')
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            except OSError:
                print0("[EET] Downloading missing spaCy model 'en_core_web_sm'...")
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            from nanochat.tokenizer import get_tokenizer
            tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)

            # Decode each token and tag it
            batch_texts = []
            batch_ids = []
            for tok_id in range(vocab_size):
                try:
                    text = tokenizer.decode([tok_id])
                    if text.strip():
                        batch_texts.append(text.strip())
                        batch_ids.append(tok_id)
                except Exception:
                    continue

            # Process in batches for efficiency
            BATCH = 1000
            for i in range(0, len(batch_texts), BATCH):
                chunk_texts = batch_texts[i:i + BATCH]
                chunk_ids = batch_ids[i:i + BATCH]
                docs = list(nlp.pipe(chunk_texts))
                for doc, tid in zip(docs, chunk_ids):
                    if len(doc) > 0:
                        pos = doc[0].pos_
                        pos_scores[tid] = self.POS_EXIT_SCORES.get(pos, 0.5)

            print0(f"[EET] POS categories computed for {len(batch_ids)} tokens")
        except Exception as e:
            print0(f"[EET] Warning: could not compute POS table ({e}), using uniform 0.5")

        os.makedirs(tokenizer_dir, exist_ok=True)
        torch.save(pos_scores, cache_path)
        print0(f"[EET] POS categories saved to {cache_path}")
        return pos_scores

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns per-token POS exit bias. Shape: (B, T)."""
        return self.pos_bias[input_ids]


# ---------------------------------------------------------------------------
# Exit Router (3 variants)
# ---------------------------------------------------------------------------

class AttentionRouter(nn.Module):
    """Lightweight self-attention router that allows tokens to observe surrounding context 
    before making exit routing decisions.
    """
    def __init__(self, n_embd: int, out_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.out_proj = nn.Linear(n_embd, out_dim, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, T, C)
        B, T, C = x.size()
        
        # Self-attention projections
        q = self.q_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # Causal attention using scaled dot product attention (highly optimized in modern PyTorch)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Concat heads
        
        return self.out_proj(y)


class EarlyExitRouter(nn.Module):
    """Per-layer exit router producing per-token exit probability.

    Three architecture variants:
      'linear': Single linear projection d → 1
      'mlp1':   d → hidden → 1 (one hidden layer + ReLU)
      'mlp2':   d → hidden → hidden → 1 (two hidden layers + ReLU)
      'attention' / 'attn': Multi-head causal self-attention + linear projection d → 1

    Adds frequency and POS prior biases to the raw logit before sigmoid.
    """

    def __init__(self, n_embd: int, router_type: str = 'mlp2',
                 hidden_dim: int = 0):
        super().__init__()
        self.router_type = router_type
        hidden = hidden_dim if hidden_dim > 0 else n_embd // 4

        if router_type == 'linear':
            self.net = Linear(n_embd, 1, bias=True)
        elif router_type == 'mlp1':
            self.net = nn.Sequential(
                Linear(n_embd, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, 1, bias=True),
            )
        elif router_type == 'mlp2':
            self.net = nn.Sequential(
                Linear(n_embd, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, 1, bias=True),
            )
        elif router_type in ('attention', 'attn'):
            self.net = AttentionRouter(n_embd, 1)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    def forward(self, h: torch.Tensor,
                freq_bias: torch.Tensor = None,
                pos_bias: torch.Tensor = None,
                freq_alpha: float = 0.0,
                pos_beta: float = 0.0,
                temp: float = 1.0) -> torch.Tensor:
        """Compute exit probability for each token.

        Args:
            h: hidden states (B, T, d)
            freq_bias: frequency prior (B, T), higher = more likely to exit
            pos_bias: POS prior (B, T), higher = more likely to exit
            freq_alpha: weight for frequency prior
            pos_beta: weight for POS prior
            temp: temperature for sigmoid annealing

        Returns:
            exit_prob: (B, T) in [0, 1]
        """
        # Compute in float32 for gradient precision — the per-token
        # differentiation signal (~1e-7) is below bf16 minimum (6e-5).
        # Router params are float32 (set in init_weights), so this
        # auto-upcasts h. Cast output back to model dtype.
        h_f32 = h.float()
        logit = self.net(h_f32).squeeze(-1)  # (B, T) float32
        if freq_bias is not None and freq_alpha > 0:
            logit = logit + freq_alpha * freq_bias.float()
        if pos_bias is not None and pos_beta > 0:
            logit = logit + pos_beta * pos_bias.float()
        
        # Apply temperature scaling
        logit = logit / temp
            
        # Soft-clamp logits to prevent sigmoid saturation while keeping gradients flowing.
        # tanh ensures exit probs stay strictly in (0.0067, 0.9933) with non-zero derivative.
        logit = 5.0 * torch.tanh(logit / 5.0)
        return torch.sigmoid(logit).to(h.dtype)

    def get_logit(self, h: torch.Tensor,
                  freq_bias: torch.Tensor = None,
                  pos_bias: torch.Tensor = None,
                  freq_alpha: float = 0.0,
                  pos_beta: float = 0.0) -> torch.Tensor:
        """Compute pre-activation exit logit. Used for Gumbel-Softmax."""
        h_f32 = h.float()
        logit = self.net(h_f32)  # (B, T, 1) float32
        if freq_bias is not None and freq_alpha > 0:
            logit = logit + freq_alpha * freq_bias.float().unsqueeze(-1)
        if pos_bias is not None and pos_beta > 0:
            logit = logit + pos_beta * pos_bias.float().unsqueeze(-1)
        # Soft-clamp logits to keep gradients flowing under extreme activations
        logit = 5.0 * torch.tanh(logit / 5.0)
        return logit.to(h.dtype)


class GlobalExitRouter(nn.Module):
    """Upfront global router that maps initial sequence embeddings directly to exit predictions.

    Three architecture variants:
      'linear': Single linear projection d → n_exits
      'mlp1':   d → hidden → n_exits (one hidden layer + LeakyReLU)
      'mlp2':   d → hidden → hidden → n_exits (two hidden layers + LeakyReLU)
      'attention' / 'attn': Multi-head causal self-attention + linear projection d → n_exits
    """

    def __init__(self, n_embd: int, n_exits: int, router_type: str = 'mlp2',
                 hidden_dim: int = 0):
        super().__init__()
        self.router_type = router_type
        hidden = hidden_dim if hidden_dim > 0 else n_embd // 4

        if router_type == 'linear':
            self.net = Linear(n_embd, n_exits, bias=True)
        elif router_type == 'mlp1':
            self.net = nn.Sequential(
                Linear(n_embd, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, n_exits, bias=True),
            )
        elif router_type == 'mlp2':
            self.net = nn.Sequential(
                Linear(n_embd, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, hidden, bias=True),
                nn.LeakyReLU(0.01),
                Linear(hidden, n_exits, bias=True),
            )
        elif router_type in ('attention', 'attn'):
            self.net = AttentionRouter(n_embd, n_exits)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    def forward(self, h: torch.Tensor,
                freq_bias: torch.Tensor = None,
                pos_bias: torch.Tensor = None,
                freq_alpha: float = 0.0,
                pos_beta: float = 0.0) -> torch.Tensor:
        """Compute upfront pre-activation logits for Gumbel/softmax exit layer distribution.

        Args:
            h: hidden states/embeddings (B, T, d)
            freq_bias: frequency prior (B, T), higher = more likely to exit
            pos_bias: POS prior (B, T), higher = more likely to exit
            freq_alpha: weight for frequency prior
            pos_beta: weight for POS prior

        Returns:
            logits: (B, T, n_exits) float32 pre-activation logits
        """
        h_f32 = h.float()
        logits = self.net(h_f32)  # (B, T, n_exits) float32
        n_exits = logits.size(-1)

        # Apply frequency prior: early exits get stronger positive bias, later exits get negative/weaker bias
        if freq_bias is not None and freq_alpha > 0:
            exit_indices = torch.arange(n_exits, device=h.device, dtype=torch.float32)
            decay = 1.0 - exit_indices / max(n_exits - 1, 1)
            logits = logits + freq_alpha * freq_bias.float().unsqueeze(-1) * decay

        # Apply POS prior
        if pos_bias is not None and pos_beta > 0:
            exit_indices = torch.arange(n_exits, device=h.device, dtype=torch.float32)
            decay = 1.0 - exit_indices / max(n_exits - 1, 1)
            logits = logits + pos_beta * pos_bias.float().unsqueeze(-1) * decay

        # Soft-clamp logits to keep gradients flowing under extreme activations
        logits = 5.0 * torch.tanh(logits / 5.0)
        return logits.to(h.dtype)


# ---------------------------------------------------------------------------
# Per-Exit Low-Rank Adapter
# ---------------------------------------------------------------------------

class ExitAdapter(nn.Module):
    """Low-rank residual adapter: x + up(gelu(down(x))).

    Maps early-exit representations to the lm_head's expected input space.
    Zero-initialized so it starts as identity (no-op at init).
    """

    def __init__(self, n_embd: int, rank: int = 16):
        super().__init__()
        self.down = Linear(n_embd, rank, bias=False)
        self.up = Linear(rank, n_embd, bias=False)
        # Zero-init up projection so adapter starts as identity
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x.float()))).to(x.dtype)

# ---------------------------------------------------------------------------
# TunedLens Translator
# ---------------------------------------------------------------------------

class TunedLensTranslator(nn.Module):
    """Per-layer affine translator: maps intermediate hidden state to
    final-layer prediction space for reconstruction loss.

    rank=0: full d→d linear (identity-init via zero weights)
    rank>0: d→rank→d bottleneck
    """

    def __init__(self, n_embd: int, rank: int = 0):
        super().__init__()
        if rank > 0:
            self.down = Linear(n_embd, rank, bias=False)
            self.up = Linear(rank, n_embd, bias=False)
        else:
            self.proj = Linear(n_embd, n_embd, bias=True)
        self.rank = rank

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.rank > 0:
            return self.up(self.down(h))
        return self.proj(h)


# ---------------------------------------------------------------------------
# Phase Scheduler
# ---------------------------------------------------------------------------

class EETPhaseScheduler:
    """Step-aware training phase manager for the three-phase EET schedule.

    Phase 1 (Dense Warmup):  steps [0, W)       — no routing
    Phase 2 (Exploration):   steps [W, W+E)     — reconstruction + efficiency loss
    Phase 3 (Committed):     steps [W+E, total) — increasing efficiency pressure
    """

    def __init__(self, total_steps: int, warmup_frac: float = 0.02,
                 explore_frac: float = 0.15,
                 reconstruct_lambda: float = 1.0,
                 efficiency_lambda_start: float = 0.01,
                 efficiency_lambda_end: float = 0.1):
        self.total_steps = max(total_steps, 1)
        self.warmup_end = int(warmup_frac * self.total_steps)
        if warmup_frac + explore_frac >= 1.0:
            self.explore_end = self.total_steps
        else:
            self.explore_end = self.warmup_end + int(explore_frac * self.total_steps)
        self.reconstruct_lambda = reconstruct_lambda
        self.efficiency_lambda_start = efficiency_lambda_start
        self.efficiency_lambda_end = efficiency_lambda_end

    def get_phase(self, step: int) -> dict:
        """Returns current phase info.

        Keys: phase (1/2/3), do_route (bool), lambda_r (float), lambda_e (float)
        """
        if step < self.warmup_end:
            return {'phase': 1, 'do_route': False, 'lambda_r': 0.0, 'lambda_e': 0.0}
        elif step < self.explore_end:
            # Linear ramp-up of efficiency penalty during exploration, scaled by 0.01 to prevent dominating quality losses
            progress = (step - self.warmup_end) / max(self.explore_end - self.warmup_end, 1)
            lambda_e = self.efficiency_lambda_start * progress * 0.01
            return {
                'phase': 2, 'do_route': True,
                'lambda_r': self.reconstruct_lambda,
                'lambda_e': lambda_e,
            }
        else:
            # Gradually anneal efficiency up from the explore end
            progress = (step - self.explore_end) / max(self.total_steps - self.explore_end, 1)
            progress = min(progress, 1.0)
            lambda_e = (self.efficiency_lambda_start * 0.01 +
                        progress * (self.efficiency_lambda_end - self.efficiency_lambda_start * 0.01))
            return {
                'phase': 3, 'do_route': True,
                'lambda_r': 0.0,  # no reconstruction loss in committed phase
                'lambda_e': lambda_e,
            }


# ---------------------------------------------------------------------------
# EarlyExitGPT — extends GPT with early exit forward pass
# ---------------------------------------------------------------------------

class EarlyExitGPT(GPT):
    """GPT with prior-informed early exit.

    Extends GPT.__init__ to add per-layer routers, translators, and priors.
    Overrides forward() to implement masking-based early exit with frozen KV.
    """

    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__(config, pad_vocab_size_to)

        n_layer = config.n_layer
        n_embd = config.n_embd
        router_hidden = config.eet_router_hidden if config.eet_router_hidden > 0 else n_embd // 4

        # Exit router(s) setup
        if getattr(config, 'eet_global_router', False):
            # Upfront single global exit router
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            if getattr(config, 'eet_ffn_skip', False):
                # A³D: one FFN decision per layer
                n_exits = n_layer
            else:
                n_exits = len(routing_layers) + 1
            self.eet_routers = nn.ModuleList([
                GlobalExitRouter(n_embd, n_exits, router_type=config.eet_router_type,
                                 hidden_dim=router_hidden)
            ])
        else:
            # Per-layer routers (layers 0..N-2; final layer has no router)
            self.eet_routers = nn.ModuleList([
                EarlyExitRouter(n_embd, router_type=config.eet_router_type,
                                hidden_dim=router_hidden)
                for _ in range(n_layer - 1)
            ])

        # Per-layer TunedLens translators — only for 'reconstruct' variant
        if config.eet_loss_variant == 'reconstruct':
            self.eet_translators = nn.ModuleList([
                TunedLensTranslator(n_embd, rank=config.eet_translator_rank)
                for _ in range(n_layer - 1)
            ])
        else:
            self.eet_translators = nn.ModuleList()  # empty — no wasted params

        # Per-exit low-rank adapters (maps each exit depth's representation to lm_head space)
        adapter_rank = getattr(config, 'eet_exit_adapter_rank', 0)
        if adapter_rank > 0 and getattr(config, 'eet_global_router', False):
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            n_adapters = len(routing_layers) + 1  # routing exits + final layer
            self.eet_exit_adapters = nn.ModuleList([
                ExitAdapter(n_embd, rank=adapter_rank) for _ in range(n_adapters)
            ])
        else:
            self.eet_exit_adapters = nn.ModuleList()

        # Per-exit depth-conditional affine (γ, β) — cheap manifold alignment before LM head
        if getattr(config, 'eet_depth_affine', False) and getattr(config, 'eet_global_router', False):
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            n_affine = len(routing_layers) + 1  # routing exits + final layer
            self.exit_gamma = nn.ParameterList([
                nn.Parameter(torch.ones(n_embd)) for _ in range(n_affine)
            ])
            self.exit_beta = nn.ParameterList([
                nn.Parameter(torch.zeros(n_embd)) for _ in range(n_affine)
            ])
        else:
            self.exit_gamma = nn.ParameterList()
            self.exit_beta = nn.ParameterList()

        # Departure summary injection: learned gate per exit slot
        if getattr(config, 'eet_departure_summary', False) and getattr(config, 'eet_global_router', False):
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            n_gates = len(routing_layers)
            self.departure_gate = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(n_gates)
            ])
        else:
            self.departure_gate = nn.ParameterList()

        # Learned exit schedule: replace fixed bell/uniform with learnable fractions
        if getattr(config, 'eet_learned_schedule', False) and getattr(config, 'eet_global_router', False):
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            n_sched = len(routing_layers) + 1  # exit slots + final
            self.exit_schedule_logits = nn.Parameter(torch.zeros(n_sched))
        else:
            self.exit_schedule_logits = None

        self.n_exits = n_exits if getattr(config, 'eet_global_router', False) else n_layer
        if getattr(config, 'eet_route_consistency_lambda', 0.0) > 0:
            self.register_buffer(
                'vocab_route_ema',
                torch.ones(config.vocab_size, self.n_exits) / self.n_exits
            )
        else:
            self.vocab_route_ema = None
        if getattr(config, 'eet_depth_weight_type', 'none') == 'ema':
            self.register_buffer(
                'exit_freq_ema',
                torch.ones(self.n_exits) / self.n_exits
            )

        # Priors (eagerly loaded/computed in __init__ to prevent torch.compile graph breaks/OOM)
        tokenizer_dir = getattr(config, '_tokenizer_dir', None)
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")

        if config.eet_freq_prior_alpha > 0:
            self._freq_prior = FrequencyPrior(config.vocab_size, tokenizer_dir)
        else:
            self._freq_prior = None

        if config.eet_pos_prior_beta > 0:
            self._pos_prior = POSPrior(config.vocab_size, tokenizer_dir)
        else:
            self._pos_prior = None

        # Accumulators for Phase 1 dense token cross-entropy calibration
        self.register_buffer('token_ce_sum', torch.zeros(config.vocab_size, dtype=torch.float32))
        self.register_buffer('token_ce_count', torch.zeros(config.vocab_size, dtype=torch.float32))
        self.register_buffer('token_difficulty', torch.zeros(config.vocab_size, dtype=torch.float32))
        self.register_buffer('eet_phase_tracker', torch.tensor([1], dtype=torch.int32))
        self.eet_current_phase = 1

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # Restore eet_current_phase from loaded buffer
        tracker_key = prefix + 'eet_phase_tracker'
        if tracker_key in state_dict:
            self.eet_current_phase = int(state_dict[tracker_key][0].item())

    @torch.no_grad()
    def init_weights(self):
        """Initialize base GPT weights + EET-specific parameters."""
        super().init_weights()

        # to_empty() replaces ALL tensor storage (including registered buffers)
        # with uninitialized garbage or leaves them as meta tensors.
        # We must re-load from disk and re-register on the correct device.
        config = self.config
        tokenizer_dir = getattr(config, '_tokenizer_dir', None)
        if tokenizer_dir is None:
            from nanochat.common import get_base_dir
            tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
        target_device = next(self.parameters()).device

        if self._freq_prior is not None:
            freq_table = self._freq_prior._load_or_compute(
                config.vocab_size, tokenizer_dir, 'cpu'
            )
            log_freq = torch.log1p(freq_table)
            log_freq = log_freq / (log_freq.max() + 1e-8)
            self._freq_prior.register_buffer('freq_bias', log_freq.to(target_device))

        if self._pos_prior is not None:
            pos_table = self._pos_prior._load_or_compute(
                config.vocab_size, tokenizer_dir
            )
            self._pos_prior.register_buffer('pos_bias', pos_table.to(target_device))

        # Routers: init with enough scale so output VARIES across tokens from step 1.
        # std=0.01 was too small — produced ~0.001 output variation → constant softmax.
        # std=0.1 gives ~0.01 softmax variation → enough for argmax to differentiate.
        for router in self.eet_routers:
            if router.router_type == 'linear':
                nn.init.normal_(router.net.weight, std=0.1)
                nn.init.constant_(router.net.bias, 0.0)
            else:
                # Scale last linear in MLP chain
                last_linear = list(router.net.modules())[-1]
                if isinstance(last_linear, (Linear, nn.Linear)):
                    nn.init.normal_(last_linear.weight, std=0.1)
                    nn.init.constant_(last_linear.bias, 0.0)

        # Keep router params in float32 for gradient precision.
        # The per-token differentiation signal (~1e-7) rounds to zero in bf16.
        for router in self.eet_routers:
            router.float()

        # Translators: identity-like init (only when present)
        for translator in self.eet_translators:
            if translator.rank > 0:
                nn.init.zeros_(translator.down.weight)
                nn.init.zeros_(translator.up.weight)
            else:
                nn.init.zeros_(translator.proj.weight)
                nn.init.zeros_(translator.proj.bias)

    @torch.compiler.disable
    @torch.no_grad()
    def _compute_dense_logits(self, idx):
        """Run dense forward pass (all blocks, no routing) and return dense hidden states."""
        B, T = idx.shape
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = x.to(torch.bfloat16 if idx.device.type == 'cuda' else torch.float32)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x_input = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = None
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx).to(x_input.dtype)
            x = block(x_input, ve, cos_sin, self.window_sizes[i], None)
        return norm(x)

    def _compute_dense_distillation_loss(self, x_final, dense_x, targets):
        """Compute chunked KL-divergence distillation loss between EET final states and dense final states."""
        B, T, C = x_final.shape
        x_flat = x_final.view(-1, C)
        dense_flat = dense_x.view(-1, C)
        targets_flat = targets.view(-1)
        
        loss_distill = torch.tensor(0.0, device=x_final.device, dtype=torch.float32)
        count = 0
        chunk_size = 128
        
        valid_mask = (targets_flat != -1)
        if not valid_mask.any():
            return loss_distill
            
        x_flat = x_flat[valid_mask]
        dense_flat = dense_flat[valid_mask]
        
        n_tokens = x_flat.size(0)
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            
            with torch.no_grad():
                dense_logits = self.lm_head(dense_flat[start:end])
                dense_logits = dense_logits[..., :self.config.vocab_size].float()
                dense_logits = 20.0 * torch.tanh(dense_logits / 20.0)
                dense_probs = torch.softmax(dense_logits / 2.0, dim=-1)
                
            student_logits = self.lm_head(x_flat[start:end])
            student_logits = student_logits[..., :self.config.vocab_size].float()
            student_logits = 20.0 * torch.tanh(student_logits / 20.0)
            student_log_probs = torch.log_softmax(student_logits / 2.0, dim=-1)
            
            kl = F.kl_div(student_log_probs, dense_probs, reduction='sum')
            loss_distill = loss_distill + kl
            count = count + (end - start)
            
        if count > 0:
            loss_distill = 4.0 * (loss_distill / count)
        return loss_distill

    @torch.compiler.disable
    @torch.no_grad()
    def _compute_dense_per_token_ce(self, idx, targets):
        """Run dense forward (all blocks, no routing) and return per-token CE.
        
        Used by the two-pass ce_guided trick: compare CE at full depth vs CE
        at exit depth to identify tokens that were hurt by early routing.
        
        Memory-efficient: chunks the lm_head + CE computation so we never
        allocate the full (B*T, vocab_size) logits tensor at once.
        """
        B, T = idx.shape
        config = self.config

        # Rotary embeddings
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Embedding
        x = self.transformer.wte(idx)
        x = x.to(torch.bfloat16 if idx.device.type == 'cuda' else torch.float32)
        x = norm(x)
        x0 = x

        # Run all blocks densely
        for i, block in enumerate(self.transformer.h):
            x0_w = self.x0_lambdas[i]
            x_input = self.resid_lambdas[i] * x + x0_w * x0

            ve = None
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx).to(x_input.dtype)

            x = block(x_input, ve, cos_sin, self.window_sizes[i], None)

        x = norm(x)

        # Chunked lm_head + CE: process 128 tokens at a time to avoid OOM
        # from materializing full (B*T, vocab_size) logits tensor
        x_flat = x.view(-1, x.shape[-1])        # (B*T, C)
        targets_flat = targets.view(-1)           # (B*T,)
        per_token_ce = torch.zeros(B * T, device=idx.device, dtype=torch.float32)
        chunk_size = 128
        for start in range(0, B * T, chunk_size):
            end = min(start + chunk_size, B * T)
            chunk_logits = self.lm_head(x_flat[start:end])
            chunk_logits = chunk_logits[..., :config.vocab_size].float()
            chunk_logits = 20.0 * torch.tanh(chunk_logits / 20.0)
            per_token_ce[start:end] = F.cross_entropy(
                chunk_logits, targets_flat[start:end],
                ignore_index=-1, reduction='none'
            )
            del chunk_logits

        return per_token_ce.view(B, T)

    @torch.no_grad()
    def finalize_token_difficulty(self):
        """Finalizes the token difficulty lookup tensor using DDP-all-reduced sums/counts if available."""
        import torch.distributed as dist
        
        # If in distributed setting, all-reduce the accumulated sums and counts across all ranks
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.token_ce_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.token_ce_count, op=dist.ReduceOp.SUM)
            
        avg_ce = self.token_ce_sum / (self.token_ce_count + 1e-8)
        observed_mask = (self.token_ce_count > 0)
        
        if observed_mask.any():
            observed_ce = avg_ce[observed_mask]
            min_ce = observed_ce.min()
            max_ce = observed_ce.max()
            norm_ce = (avg_ce - min_ce) / (max_ce - min_ce + 1e-8)
            norm_ce = norm_ce.clamp(0.0, 1.0)
            # Unobserved tokens get default 1.0 (safest: max difficulty, exit late)
            norm_ce[~observed_mask] = 1.0
            self.token_difficulty.copy_(norm_ce)
        else:
            # Fallback to all ones if no tokens observed
            self.token_difficulty.fill_(1.0)

    @torch.no_grad()
    def calibrate_token_difficulty(self, calibration_batches, target_phase=2):
        """Runs one forward pass over a set of calibration batches to compute and freeze the token difficulty lookup.
        
        Runs in eager mode with micro-batching to guarantee OOM-free calibration.
        """
        device = next(self.parameters()).device
        
        # Free up cache memory before starting calibration
        torch.cuda.empty_cache()
        
        # Reset local sums and counts
        self.token_ce_sum.zero_()
        self.token_ce_count.zero_()
        
        # We run the dense model to compute logits and per-token CE
        self.eval()  # ensure eval mode for stability
        for batch in calibration_batches:
            idx_full = batch[0].to(device)
            targets_full = batch[1].to(device)
            
            # Chunk the batch size to 1 to avoid OOM from large (B, T, V) logits tensors
            B = idx_full.size(0)
            for chunk_start in range(B):
                idx = idx_full[chunk_start : chunk_start + 1]
                targets = targets_full[chunk_start : chunk_start + 1]
                
                # Forward pass as dense
                logits = super().forward(idx, None)  # (1, T, V) — extremely memory light!
                
                # Compute per-token CE
                per_token_ce = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    ignore_index=-1, reduction='none'
                )  # (T,)
                
                # Accumulate
                valid_mask = (targets.view(-1) != -1)
                valid_targets = targets.view(-1)[valid_mask]
                valid_ce = per_token_ce[valid_mask]
                self.token_ce_sum.index_add_(0, valid_targets, valid_ce.float())
                self.token_ce_count.index_add_(0, valid_targets, torch.ones_like(valid_targets, dtype=torch.float32))
                
        self.train()  # restore train mode
        torch.cuda.empty_cache()  # free up logits memory
        
        # Finalize (includes all-reduce if in DDP)
        self.finalize_token_difficulty()
        self.eet_current_phase = target_phase
        self.eet_phase_tracker[0] = target_phase

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                eet_do_route=False, eet_phase=1, eet_lambda_r=0.0, eet_lambda_e=0.0,
                eet_gumbel_temp=1.0, eet_step=0, eet_total_steps=1, eet_dense_x=None,
                eet_target_active_frac=None):
        """Forward pass with early exit routing.

        Phase scheduling is done OUTSIDE this method (in base_train.py) to
        avoid torch.compile creating a single graph covering all phases.
        Args eet_do_route (bool) and eet_phase (int) are stable compile guards
        that cause at most 3 compiled graph variants. eet_lambda_r/e are Python
        floats converted to scalar tensors inside to avoid per-step recompiles.
        """
        B, T = idx.size()
        config = self.config
        loss_variant = config.eet_loss_variant

        do_route = eet_do_route
        if not self.training and config.use_eet and self.eet_current_phase in {2, 3}:
            do_route = True
            eet_phase = 3

        # A³D dispatch: completely separate code path to avoid polluting
        # the compute_skip torch.compile graph with dead branches
        if getattr(config, 'eet_ffn_skip', False) and do_route:
            return self._forward_a3d(
                idx, targets, kv_cache, loss_reduction,
                eet_phase=eet_phase, eet_lambda_r=eet_lambda_r, eet_lambda_e=eet_lambda_e,
                eet_gumbel_temp=eet_gumbel_temp, eet_step=eet_step, eet_total_steps=eet_total_steps,
            )

        # Phase 1 (no routing): delegate to parent GPT.forward() to get the
        # exact same torch.compile graph as dense — avoiding OOM from the
        # compiler generating a less memory-efficient graph for the EET
        # forward's additional dead-code paths (frozen_h, reconstruction, etc.)
        if not do_route:
            loss_or_logits = super().forward(idx, targets, kv_cache, loss_reduction)
            if self.training and targets is not None:
                self._eet_diagnostics = {
                    'phase': eet_phase,
                    'active_frac': 1.0,
                    'total_exit_frac': 0.0,
                }
            return loss_or_logits

        # --- Standard GPT embedding ---
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        T_total = T0 + T
        if T_total > self.cos.size(1):
            new_len = max(T_total, self.cos.size(1) * 2)
            head_dim = config.n_embd // config.n_head
            cos, sin = self._precompute_rotary_embeddings(new_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total]

        if self.embedding_model is None:
            x = self.transformer.wte(idx)
        else:
            x, _ = self.embedding_model(idx)
        if "wpe" in self.transformer:
            positions = torch.arange(T0, T_total, device=idx.device)
            x = x + self.transformer.wpe(positions)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x  # save for x0 residual

        freq_bias = self._freq_prior(idx) if self._freq_prior is not None else None
        pos_bias = self._pos_prior(idx) if self._pos_prior is not None else None

        # --- Early exit forward pass ---
        # Use soft blending during Phase 2 training to provide gradients to the
        # router and transformer weights. Phase 3 training runs the hard routing
        # path, but computes parallel soft router probabilities for gradient propagation
        # (hybrid path) so the router continues learning from quality/efficiency loss.
        use_gumbel = (config.eet_gumbel_temp_start > 0.0)
        compute_skip = getattr(config, 'eet_compute_skip', False)
        is_global_router = getattr(config, 'eet_global_router', False)
        # Compute is_soft_training first so the auto-enable checks below can reference it
        is_soft_training = (eet_do_route and self.training and eet_phase == 2) or (use_gumbel and eet_do_route and self.training and not compute_skip)
        is_layer_weighted = (loss_variant == 'layer_weighted')

        # Global router requires compute_skip to physically drop tokens.
        # Without it the code falls through to soft blending (no compute savings).
        if is_global_router and not compute_skip and do_route and not is_soft_training and not is_layer_weighted:
            print0("[EET] WARNING: eet_global_router=True but eet_compute_skip=False. "
                   "Enabling compute_skip automatically.")
            compute_skip = True

        # The threshold-based hard-exit path never drops tokens at init (fresh router
        # outputs ~0, never crosses eet_exit_threshold). compute_skip uses rank-based
        # top-K which always drops exactly the right number of tokens regardless of
        # router magnitude. Auto-enable whenever we're doing hard routing.
        if do_route and not compute_skip and not is_soft_training and not is_layer_weighted:
            print0("[EET] WARNING: eet_compute_skip=False with hard routing. "
                   "Enabling automatically — set eet_compute_skip=True in config to suppress.")
            compute_skip = True

        token_active = torch.ones(B, T, dtype=torch.bool, device=x.device)
        frozen_h = torch.zeros_like(x)
        reconstruction_losses = []
        total_exit_frac = torch.tensor(0.0, device=x.device)
        n_routed_layers = 0

        blocks = list(self.transformer.h)
        n_layer = len(blocks)
        need_exit_tracking = (eet_phase == 2 and not is_global_router and
                              loss_variant not in ('reconstruct', 'quality', 'ce_guided') and not is_layer_weighted)
        
        if need_exit_tracking:
            exit_hidden = torch.zeros_like(x)                # (B, T, D)
            exit_layers = torch.full((B, T), n_layer - 1, dtype=torch.long, device=x.device)
            if loss_variant == 'entropy_surprise':
                prev_exit_hidden = torch.zeros_like(x)       # for surprise term

        prev_x = x.detach()  # hidden state before first block

        if is_global_router and not compute_skip:
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            n_exits = len(routing_layers) + 1
            
            # Predict exit logits upfront from input embedding x0
            global_router = self.eet_routers[0]
            router_logits = global_router(
                x0,
                freq_bias=freq_bias,
                pos_bias=pos_bias,
                freq_alpha=config.eet_freq_prior_alpha,
                pos_beta=config.eet_pos_prior_beta
            )  # (B, T, n_exits)
            if self.exit_schedule_logits is not None:
                router_logits = router_logits + self.exit_schedule_logits
            
            # Gumbel vs Standard Soft vs Hard
            if use_gumbel:
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(router_logits.float()).clamp(1e-9)) + 1e-9
                )
                noisy_logits = (router_logits.float() + gumbel_noise) / eet_gumbel_temp
                soft_weights = torch.softmax(noisy_logits, dim=-1).to(x0.dtype)  # (B, T, n_exits)
                
                if config.eet_gumbel_hard:
                    hard_weights = torch.zeros_like(soft_weights)
                    hard_weights.scatter_(
                        -1,
                        soft_weights.argmax(dim=-1, keepdim=True),
                        1.0
                    )
                    routing_weights = hard_weights - soft_weights.detach() + soft_weights
                else:
                    routing_weights = soft_weights
            elif is_soft_training or is_layer_weighted:
                if is_soft_training:
                    routing_weights = torch.softmax(router_logits.float() / eet_gumbel_temp, dim=-1).to(x0.dtype)
                else:
                    routing_weights = torch.softmax(router_logits.float(), dim=-1).to(x0.dtype)
                soft_weights = routing_weights
            else:
                # Hard routing (Phase 3 or inference)
                soft_weights = torch.softmax(router_logits.float(), dim=-1).to(x0.dtype)
                exit_layer_hard = router_logits.argmax(dim=-1)
                hard_weights = torch.zeros_like(router_logits)
                hard_weights.scatter_(
                    -1,
                    exit_layer_hard.unsqueeze(-1),
                    1.0
                )
                if self.training:
                    routing_weights = hard_weights.to(x0.dtype) - soft_weights.detach() + soft_weights
                else:
                    routing_weights = hard_weights.to(x0.dtype)
                    soft_weights = routing_weights
                
            if self.training and getattr(config, 'eet_use_override', 0) == 1:
                step_val = eet_step
                total_val = eet_total_steps
                if not isinstance(step_val, torch.Tensor):
                    step_val = torch.tensor(step_val, device=x0.device, dtype=torch.float32)
                if not isinstance(total_val, torch.Tensor):
                    total_val = torch.tensor(total_val, device=x0.device, dtype=torch.float32)
                progress = step_val.float() / torch.clamp(total_val.float(), min=1.0)
                p_start = getattr(config, 'eet_override_prob_start', 0.5)
                p_end = getattr(config, 'eet_override_prob_end', 0.1)
                p_override = torch.maximum(torch.tensor(p_end, device=x0.device), torch.tensor(p_start, device=x0.device) * (1.0 - progress))
                
                override_mask = (torch.rand(B, T, device=x0.device) < p_override).unsqueeze(-1)
                forced = torch.zeros_like(routing_weights)
                forced[:, :, -1] = 1.0
                
                routing_weights = torch.where(override_mask, forced, routing_weights)
                soft_weights = torch.where(override_mask, forced, soft_weights)
                
            # Collect states densely
            candidate_states = []
            for i, block in enumerate(blocks):
                x0_w = self.x0_lambdas[i]
                if self._use_residual_decay and self.depth_decay_raw is not None:
                    decay_base = torch.sigmoid(self.depth_decay_raw)
                    x0_w = x0_w * (decay_base ** i)
                x_input = self.resid_lambdas[i] * x + x0_w * x0
                
                ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
                x_new = block(x_input, ve, cos_sin, self.window_sizes[i], kv_cache)
                
                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x_new.dtype)
                    mixed = self.residual_mixers[i](x_new.transpose(1, 2)).transpose(1, 2)
                    x_new = x_new + gamma * mixed
                x = x_new
                
                if i in routing_layers or i == n_layer - 1:
                    candidate_states.append(norm(x))
            
            # Blend hidden states using routing weights
            if len(self.eet_exit_adapters) > 0 or len(self.exit_gamma) > 0:
                adapted_states = []
                for idx_ad, state in enumerate(candidate_states):
                    adapted = state
                    if len(self.eet_exit_adapters) > 0:
                        adapted = self.eet_exit_adapters[idx_ad](adapted)
                    if len(self.exit_gamma) > 0:
                        gamma = self.exit_gamma[idx_ad].to(adapted.dtype)
                        beta = self.exit_beta[idx_ad].to(adapted.dtype)
                        adapted = adapted * gamma + beta
                    adapted_states.append(adapted)
                stacked = torch.stack(adapted_states, dim=2)
            else:
                stacked = torch.stack(candidate_states, dim=2)  # (B, T, n_exits, D)
            x_final = (routing_weights.unsqueeze(-1) * stacked).sum(dim=2)  # (B, T, D)
            
            p_exits = [soft_weights[:, :, k] for k in range(soft_weights.size(-1))]
            self._last_exit_probs = torch.stack(p_exits, dim=-1)
            
            # Diagnostics
            exit_layer_hard = routing_weights.argmax(dim=-1).float()
            soft_active = torch.zeros(B, T, dtype=x.dtype, device=x.device)
            for slot_idx, p_exit_i in enumerate(p_exits[:-1]):
                soft_active = soft_active + p_exit_i * ((config.eet_min_exit_layer + slot_idx + 1) / n_layer)
            soft_active = soft_active + p_exits[-1] * 1.0
            
            avg_active = soft_active.mean()
            total_exit_frac = (exit_layer_hard < len(routing_layers)).float().mean()
            
            if targets is not None:
                if loss_variant == 'layer_weighted':
                    # Global Layer Weighted Loss
                    loss_accumulator = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                    for k, (h_k, p_k) in enumerate(zip(candidate_states, p_exits)):
                        loss_k_mean, _, _ = checkpoint(
                            compute_layer_loss,
                            h_k, p_k, targets, self.lm_head, config, config.eet_quality_lambda,
                            use_reentrant=False
                        )
                        loss_accumulator = loss_accumulator + loss_k_mean
                        
                    eff_div_loss, _ = compute_efficiency_and_diversity(
                        p_exits, len(p_exits), freq_bias, config, eet_lambda_e
                    )
                    loss = loss_accumulator + eff_div_loss
                else:
                    logits = self.lm_head(x_final)
                    logits = logits[..., :config.vocab_size].float()
                    logits = 20 * torch.tanh(logits / 20)
                    
                    # For ce_guided: compute per-token CE from the SAME F.cross_entropy
                    # call (reduction='none' then manual mean). Zero extra memory vs 'mean'.
                    if loss_variant == 'ce_guided' and loss_reduction == 'mean':
                        per_token_ce_flat = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), targets.view(-1),
                            ignore_index=-1, reduction='none',
                        )  # (B*T,)
                        # Mask ignored tokens for proper mean
                        valid_mask_flat = (targets.view(-1) != -1).float()

                        dw_type = getattr(config, 'eet_depth_weight_type', 'none')
                        use_depth_grad_scale = getattr(config, 'eet_depth_grad_scale', False)

                        if dw_type != 'none' or use_depth_grad_scale:
                            # 1. Get exit depth for every token in the batch
                            exit_depth = routing_weights.argmax(dim=-1)  # (B, T)
                            exit_depth_flat = exit_depth.view(-1)  # (B*T,)

                            # 2. Update EMA buffer if using 'ema' strategy and training
                            if dw_type == 'ema' and self.training:
                                with torch.no_grad():
                                    for k in range(self.n_exits):
                                        freq_k = (exit_depth_flat == k).float().mean()
                                        if is_ddp_initialized():
                                            dist.all_reduce(freq_k, op=dist.ReduceOp.SUM)
                                            freq_k = freq_k / dist.get_world_size()
                                        self.exit_freq_ema[k] = 0.99 * self.exit_freq_ema[k] + 0.01 * freq_k

                            # 3. Compute depth weighting (existing depth_weight_type)
                            if dw_type == 'linear':
                                max_w = getattr(config, 'eet_depth_weight_max', 2.5)
                                depth_weight_flat = 1.0 + (exit_depth_flat.float() / (self.n_exits - 1)) * (max_w - 1.0)
                            elif dw_type == 'ema':
                                raw_w = 1.0 / self.exit_freq_ema.clamp(min=0.05)
                                norm_w = raw_w / raw_w.mean()
                                depth_weight_flat = norm_w[exit_depth_flat]
                            elif dw_type == 'sqrt':
                                layer_indices = torch.arange(self.n_exits, device=idx.device).float()
                                raw_layer_w = torch.sqrt(layer_indices + 1.0)
                                norm_layer_w = raw_layer_w / raw_layer_w.mean()
                                depth_weight_flat = norm_layer_w[exit_depth_flat]
                            else:
                                depth_weight_flat = torch.ones_like(exit_depth_flat, dtype=torch.float32)

                            # 3b. Option B: depth gradient scaling from capacity schedule
                            # Weight = 1/active_fraction at exit layer (capped at 10×, normalized)
                            if use_depth_grad_scale and hasattr(self, '_last_target_fractions'):
                                tgt_fracs = self._last_target_fractions  # [exit_frac_0, ..., exit_frac_N, final_frac]
                                # Compute surviving fraction at each exit slot
                                n_slots = len(tgt_fracs)
                                surv_at_slot = []
                                _s = 1.0
                                for ef in tgt_fracs:
                                    _s -= ef
                                    surv_at_slot.append(max(_s, 0.05))
                                surv_at_slot[-1] = surv_at_slot[-2] if len(surv_at_slot) > 1 else 0.05  # final layer uses last surviving frac
                                inv_surv = [min(10.0, 1.0 / s) for s in surv_at_slot]
                                # Normalize so mean weight = 1.0
                                _mean_w = sum(inv_surv) / len(inv_surv)
                                inv_surv_norm = [w / _mean_w for w in inv_surv]
                                grad_scale_t = torch.tensor(inv_surv_norm, device=idx.device, dtype=torch.float32)
                                # Clamp exit_depth to valid range
                                clamped_depth = exit_depth_flat.clamp(0, len(inv_surv_norm) - 1)
                                grad_scale_per_token = grad_scale_t[clamped_depth]
                                depth_weight_flat = depth_weight_flat * grad_scale_per_token

                            # 4. Multiply with per-token cross entropy before reduction
                            weighted_ce = per_token_ce_flat * depth_weight_flat.detach()
                            loss = (weighted_ce * valid_mask_flat).sum() / valid_mask_flat.sum().clamp(min=1.0)
                        else:
                            loss = (per_token_ce_flat * valid_mask_flat).sum() / valid_mask_flat.sum().clamp(min=1.0)

                        per_token_ce = per_token_ce_flat.detach().view(targets.shape)  # (B, T) — detached, no grad
                    else:
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), targets.view(-1),
                            ignore_index=-1, reduction=loss_reduction,
                        )
                        per_token_ce = None
                    
                    if loss_variant == 'quality' and loss_reduction == 'mean':
                        if len(p_exits) > 1:
                            adv_tensor, vmask = self._compute_quality_advantages(
                                candidate_states, p_exits, targets, config
                            )
                            quality_loss = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                            for qi, p_exit_i in enumerate(p_exits):
                                quality_loss = quality_loss - (p_exit_i.float() * adv_tensor[qi]).mean()
                            loss = loss + config.eet_quality_lambda * quality_loss

                            if config.eet_quality_entropy_bonus > 0:
                                entropy_bonus = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                                for p_exit_i in p_exits:
                                    p_f = p_exit_i.float().clamp(min=1e-8)
                                    entropy_bonus = entropy_bonus - (p_f * torch.log(p_f)).mean()
                                loss = loss - config.eet_quality_entropy_bonus * entropy_bonus
                    elif loss_variant == 'entropy_surprise' and loss_reduction == 'mean':
                        _detach_aux = getattr(config, 'eet_detach_aux_from_backbone', False)
                        _stacked = stacked.detach() if _detach_aux else stacked
                        _rw = routing_weights.detach() if _detach_aux else routing_weights
                        _xf = x_final.detach() if _detach_aux else x_final
                        variant_loss, variant_diag = self._compute_entropy_surprise_loss(
                            _xf, _stacked, config, routing_weights=_rw
                        )
                        loss = loss + variant_loss
                    elif loss_variant == 'ce_guided' and loss_reduction == 'mean':
                        _detach_aux = getattr(config, 'eet_detach_aux_from_backbone', False)
                        _stacked = stacked.detach() if _detach_aux else stacked
                        _rw = routing_weights.detach() if _detach_aux else routing_weights
                        _rl = router_logits  # router_logits should keep grad (trains router)
                        variant_loss, variant_diag = self._compute_ce_guided_loss(
                            per_token_ce, targets, _stacked, _rw, config,
                            router_logits=_rl,
                        )
                        loss = loss + variant_loss
                    elif loss_variant == 'reconstruct' and loss_reduction == 'mean':
                        reconstruction_losses = []
                        for idx, p_exit_i in enumerate(p_exits[:-1]):
                            block_idx = config.eet_min_exit_layer + idx
                            h_i = candidate_states[idx]
                            translated = self.eet_translators[block_idx](h_i.detach())
                            reconstruction_losses.append((translated, p_exit_i))
                        
                        recon_loss_term = self._compute_reconstruction_loss(
                            logits, reconstruction_losses, eet_lambda_r, config.vocab_size
                        )
                        loss = loss + recon_loss_term

                    # Efficiency + diversity for global router quality path
                    # Skip for ce_guided: the CE classification loss already routes
                    # easy→early, hard→late. Efficiency pressure creates a coherent
                    # "always exit early" signal that overwhelms per-token differentiation.
                    if loss_reduction == 'mean' and loss_variant != 'ce_guided':
                        eff_div_loss, _ = compute_efficiency_and_diversity(
                            p_exits, len(p_exits), freq_bias, config, eet_lambda_e
                        )
                        loss = loss + eff_div_loss
                        
                    # Quality entropy bonus for any routing loss variant (e.g. ce_guided) to encourage diverse routing
                    if getattr(config, 'eet_quality_entropy_bonus', 0.0) > 0 and loss_reduction == 'mean':
                        entropy_bonus = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                        for p_exit_i in p_exits:
                            p_f = p_exit_i.float().clamp(min=1e-8)
                            entropy_bonus = entropy_bonus - (p_f * torch.log(p_f)).mean()
                        loss = loss - config.eet_quality_entropy_bonus * entropy_bonus
                                
                if getattr(config, 'eet_commitment_beta', 0.0) > 0:
                    commit_loss = self._compute_commitment_loss(
                        candidate_states, p_exits, beta=config.eet_commitment_beta
                    )
                    loss = loss + commit_loss
                    
                # Route consistency loss
                if getattr(config, 'eet_route_consistency_lambda', 0.0) > 0 and self.training and len(p_exits) > 0:
                    if hasattr(self, 'vocab_route_ema') and self.vocab_route_ema is not None:
                        idx_flat = idx.view(-1)
                        valid_idx = (idx_flat >= 0) & (idx_flat < config.vocab_size)
                        if valid_idx.any():
                            idx_valid = idx_flat[valid_idx]
                            stacked_p = torch.stack(p_exits, dim=-1)  # (B, T, n_exits)
                            soft_weights_valid = stacked_p.view(-1, stacked_p.size(-1))[valid_idx]
                            batch_ema = self.vocab_route_ema[idx_valid]
                            consistency_loss = F.mse_loss(soft_weights_valid, batch_ema)
                            loss = loss + config.eet_route_consistency_lambda * consistency_loss
                            with torch.no_grad():
                                new_ema_vals = 0.99 * batch_ema + 0.01 * soft_weights_valid.detach()
                                self.vocab_route_ema.index_copy_(0, idx_valid, new_ema_vals)

                # Concurrent dense distillation loss
                if eet_dense_x is not None and targets is not None and self.training:
                    distill_loss = self._compute_dense_distillation_loss(x_final, eet_dense_x, targets)
                    loss = loss + config.eet_dense_distill_lambda * distill_loss
                    
                if self.training:
                    self._eet_diagnostics = {
                        'phase': eet_phase,
                        'active_frac': avg_active,
                        'total_exit_frac': total_exit_frac,
                    }
                return loss
            else:
                logits = self.lm_head(x_final)
                logits = logits[..., :config.vocab_size].float()
                logits = 20 * torch.tanh(logits / 20)
                return logits

        if is_layer_weighted:
            # --- Direction 2: Per-Token Layer-Weighted Loss dense loop ---
            all_layer_norms = []
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            
            for i, block in enumerate(blocks):
                x0_w = self.x0_lambdas[i]
                if self._use_residual_decay and self.depth_decay_raw is not None:
                    decay_base = torch.sigmoid(self.depth_decay_raw)
                    x0_w = x0_w * (decay_base ** i)
                x_input = self.resid_lambdas[i] * x + x0_w * x0
                
                ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
                x_new = block(x_input, ve, cos_sin, self.window_sizes[i], kv_cache)
                
                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x_new.dtype)
                    mixed = self.residual_mixers[i](x_new.transpose(1, 2)).transpose(1, 2)
                    x_new = x_new + gamma * mixed
                x = x_new
                
                if i in routing_layers or i == n_layer - 1:
                    all_layer_norms.append(norm(x))
            
            x = all_layer_norms[-1]
            
            # Compute exit probabilities
            exit_probs_per_layer = []
            for idx, r_layer_idx in enumerate(routing_layers):
                h_k = all_layer_norms[idx]
                router = self.eet_routers[r_layer_idx]
                # detach h_k from backbone so router gradient doesn't affect backbone weights
                p = router(
                    h_k.detach(),
                    freq_bias=freq_bias,
                    pos_bias=pos_bias,
                    freq_alpha=config.eet_freq_prior_alpha,
                    pos_beta=config.eet_pos_prior_beta,
                )
                exit_probs_per_layer.append(p)
                
            # Compute cumulative exit probabilities for diagnostics and commitment loss
            p_exits = []
            p_reach = torch.ones(B, T, dtype=x.dtype, device=x.device)
            for p_exit_i in exit_probs_per_layer:
                p_exits.append(p_reach * p_exit_i)
                p_reach = p_reach * (1.0 - p_exit_i)
            p_exits.append(p_reach)
            
            # Stack soft exit probabilities for diagnostics
            if len(p_exits) > 0:
                self._last_exit_probs = torch.stack(p_exits, dim=-1)
                
            # Diagnostics for logging
            soft_active = torch.zeros(B, T, dtype=x.dtype, device=x.device)
            for idx, p_exit_i in enumerate(p_exits[:-1]):
                soft_active = soft_active + p_exit_i * ((config.eet_min_exit_layer + idx + 1) / n_layer)
            soft_active = soft_active + p_exits[-1] * 1.0
            avg_active = soft_active.mean()
            total_exit_frac = 1.0 - avg_active
            n_routed_layers = len(routing_layers)

        elif is_soft_training and not compute_skip:
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            
            if use_gumbel:
                # --- Direction 1: Gumbel-Softmax pathway ---
                router_logits_list = []
                candidate_states = []
                
                for i, block in enumerate(blocks):
                    x0_w = self.x0_lambdas[i]
                    if self._use_residual_decay and self.depth_decay_raw is not None:
                        decay_base = torch.sigmoid(self.depth_decay_raw)
                        x0_w = x0_w * (decay_base ** i)
                    x_input = self.resid_lambdas[i] * x + x0_w * x0
                    
                    ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
                    x_new = block(x_input, ve, cos_sin, self.window_sizes[i], kv_cache)
                    
                    if self.residual_mixers is not None:
                        gamma = self.residual_mix_gamma[i].to(x_new.dtype)
                        mixed = self.residual_mixers[i](x_new.transpose(1, 2)).transpose(1, 2)
                        x_new = x_new + gamma * mixed
                    x = x_new
                    
                    if i in routing_layers:
                        router = self.eet_routers[i]
                        exit_logit_i = router.get_logit(
                            norm(x),
                            freq_bias=freq_bias,
                            pos_bias=pos_bias,
                            freq_alpha=config.eet_freq_prior_alpha,
                            pos_beta=config.eet_pos_prior_beta,
                        )  # (B, T, 1)
                        router_logits_list.append(exit_logit_i)
                        candidate_states.append(norm(x))
                    elif i == n_layer - 1:
                        candidate_states.append(norm(x))
                
                # Stack logits for all exit points including final layer
                # final layer always has logit = 0 (neutral prior)
                final_logit = torch.zeros(
                    *router_logits_list[0].shape,
                    device=x.device, dtype=x.dtype
                )
                all_logits = torch.cat(
                    router_logits_list + [final_logit], dim=-1
                )  # (B, T, n_exits)
                if self.exit_schedule_logits is not None:
                    all_logits = all_logits + self.exit_schedule_logits
                
                # Sample Gumbel noise in float32 for numerical stability
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(all_logits.float()).clamp(1e-9)) + 1e-9
                )
                noisy_logits = (all_logits.float() + gumbel_noise) / eet_gumbel_temp
                soft_weights = torch.softmax(noisy_logits, dim=-1).to(x.dtype)  # (B, T, n_exits)
                
                if config.eet_gumbel_hard:
                    # Straight-Through Estimator (STE)
                    hard_weights = torch.zeros_like(soft_weights)
                    hard_weights.scatter_(
                        -1,
                        soft_weights.argmax(dim=-1, keepdim=True),
                        1.0
                    )
                    routing_weights = hard_weights - soft_weights.detach() + soft_weights
                else:
                    routing_weights = soft_weights
                    
                if self.training and getattr(config, 'eet_use_override', 0) == 1:
                    step_val = eet_step
                    total_val = eet_total_steps
                    if not isinstance(step_val, torch.Tensor):
                        step_val = torch.tensor(step_val, device=x.device, dtype=torch.float32)
                    if not isinstance(total_val, torch.Tensor):
                        total_val = torch.tensor(total_val, device=x.device, dtype=torch.float32)
                    progress = step_val.float() / torch.clamp(total_val.float(), min=1.0)
                    p_start = getattr(config, 'eet_override_prob_start', 0.5)
                    p_end = getattr(config, 'eet_override_prob_end', 0.1)
                    p_override = torch.maximum(torch.tensor(p_end, device=x.device), torch.tensor(p_start, device=x.device) * (1.0 - progress))
                    
                    override_mask = (torch.rand(B, T, device=x.device) < p_override).unsqueeze(-1)
                    forced = torch.zeros_like(routing_weights)
                    forced[:, :, -1] = 1.0
                    
                    routing_weights = torch.where(override_mask, forced, routing_weights)
                    soft_weights = torch.where(override_mask, forced, soft_weights)
                    
                # Blend hidden states
                if len(self.eet_exit_adapters) > 0 or len(self.exit_gamma) > 0:
                    adapted_states = []
                    for idx_ad, state in enumerate(candidate_states):
                        adapted = state
                        if len(self.eet_exit_adapters) > 0:
                            adapted = self.eet_exit_adapters[idx_ad](adapted)
                        if len(self.exit_gamma) > 0:
                            gamma = self.exit_gamma[idx_ad].to(adapted.dtype)
                            beta = self.exit_beta[idx_ad].to(adapted.dtype)
                            adapted = adapted * gamma + beta
                        adapted_states.append(adapted)
                    stacked = torch.stack(adapted_states, dim=2)
                else:
                    stacked = torch.stack(candidate_states, dim=2)  # (B, T, n_exits, D)
                x_final = (routing_weights.unsqueeze(-1) * stacked).sum(dim=2)  # (B, T, D)
                
                p_exits = [soft_weights[:, :, k] for k in range(soft_weights.size(-1))]
                self._last_exit_probs = torch.stack(p_exits, dim=-1)
                
                # Diagnostics
                exit_layer_hard = routing_weights.argmax(dim=-1).float()
                soft_active = torch.zeros(B, T, dtype=x.dtype, device=x.device)
                for idx, p_exit_i in enumerate(p_exits[:-1]):
                    soft_active = soft_active + p_exit_i * ((config.eet_min_exit_layer + idx + 1) / n_layer)
                soft_active = soft_active + p_exits[-1] * 1.0
                
                x = x_final
                exit_hidden = x_final
                avg_active = soft_active.mean()
                total_exit_frac = (exit_layer_hard < len(routing_layers)).float().mean()
                n_routed_layers = len(routing_layers)
            else:
                # --- Original Phase 2 Soft Blending loop ---
                exit_probs_list = []
                candidate_states = []
                candidate_prev_states = []
                
                for i, block in enumerate(blocks):
                    x0_w = self.x0_lambdas[i]
                    if self._use_residual_decay and self.depth_decay_raw is not None:
                        decay_base = torch.sigmoid(self.depth_decay_raw)
                        x0_w = x0_w * (decay_base ** i)
                    x_input = self.resid_lambdas[i] * x + x0_w * x0
                    
                    prev_x_val = x_input
                    
                    ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
                    x_new = block(x_input, ve, cos_sin, self.window_sizes[i], kv_cache)
                    
                    if self.residual_mixers is not None:
                        gamma = self.residual_mix_gamma[i].to(x_new.dtype)
                        mixed = self.residual_mixers[i](x_new.transpose(1, 2)).transpose(1, 2)
                        x_new = x_new + gamma * mixed
                    x = x_new
                    
                    if i in routing_layers:
                        exit_prob_i = self.eet_routers[i](
                            x.detach() if self.training else x,
                            freq_bias=freq_bias, pos_bias=pos_bias,
                            freq_alpha=config.eet_freq_prior_alpha,
                            pos_beta=config.eet_pos_prior_beta,
                            temp=eet_gumbel_temp if is_soft_training else 1.0
                        )
                        exit_probs_list.append(exit_prob_i)
                        candidate_states.append(norm(x))
                        candidate_prev_states.append(norm(prev_x_val))
                    elif i == n_layer - 1:
                        candidate_states.append(norm(x))
                        candidate_prev_states.append(norm(prev_x_val))
                
                # --- Original soft blending post-processing ---
                p_exits = []
                p_reach = torch.ones(B, T, dtype=x.dtype, device=x.device)
                soft_h = torch.zeros_like(x)
                soft_prev_h = torch.zeros_like(x)
                soft_active = torch.zeros(B, T, dtype=x.dtype, device=x.device)

                for idx, (r_i, h_i, prev_h_i) in enumerate(zip(exit_probs_list, candidate_states[:-1], candidate_prev_states[:-1])):
                    p_exit_i = r_i * p_reach
                    p_exits.append(p_exit_i)
                    soft_h = soft_h + p_exit_i.unsqueeze(-1) * h_i
                    soft_prev_h = soft_prev_h + p_exit_i.unsqueeze(-1) * prev_h_i
                    soft_active = soft_active + p_exit_i * ((config.eet_min_exit_layer + idx + 1) / n_layer)
                    
                    if loss_variant == 'reconstruct':
                        block_idx = config.eet_min_exit_layer + idx
                        translated = self.eet_translators[block_idx](h_i.detach())
                        reconstruction_losses.append((translated, p_exit_i))
                        
                    p_reach = p_reach * (1.0 - r_i)

                soft_h = soft_h + p_reach.unsqueeze(-1) * candidate_states[-1]
                soft_prev_h = soft_prev_h + p_reach.unsqueeze(-1) * candidate_prev_states[-1]
                soft_active = soft_active + p_reach * 1.0
                p_exits.append(p_reach)
                
                self._last_exit_probs = torch.stack(p_exits, dim=-1)
                
                x = soft_h
                exit_hidden = soft_h
                prev_exit_hidden = soft_prev_h
                avg_active = soft_active.mean()
                total_exit_frac = 1.0 - avg_active
                n_routed_layers = len(routing_layers)

        elif compute_skip and do_route:
            # --- MoD-style Fixed-Capacity Compute-Skip Path ---
            # Tokens physically skip blocks via gather/scatter. Fixed top-K capacity
            # per routable layer keeps shapes static for torch.compile.
            C = x.size(-1)
            routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
            n_rl = len(routing_layers)
            routing_set = set(routing_layers)
            target_frac = eet_target_active_frac if eet_target_active_frac is not None else getattr(config, 'eet_target_active_frac', 0.125)

            # NOTE: capacities are computed AFTER the global router runs (below),
            # so this list is populated later when is_global_router=True.
            # For per-layer routers, fall back to linear decay.
            capacities = []

            cos_full, sin_full = cos_sin  # (1, T, rope_dim)
            rope_dim = cos_full.size(-1)

            # Start with all tokens active (sorted indices)
            active_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1).contiguous()  # (B, T)
            K_cur = T
            K_before_reentry = T
            rl_counter = 0  # index into capacities / routing_layers

            # Track soft and hard exit probabilities for efficiency loss and diagnostics
            p_exits_soft = []
            p_exits = []
            p_reach = torch.ones(B, T, dtype=x.dtype, device=x.device)
            is_active = torch.ones(B, T, dtype=torch.bool, device=x.device)

            candidate_states = []
            hard_candidate_states = []

            if is_global_router:
                # Upfront single global exit router
                global_router = self.eet_routers[0]

                # Post-attention router: run first N blocks under no_grad
                # to give the router contextual input instead of raw embeddings
                _rab = getattr(config, 'eet_router_after_block', 0)
                if _rab > 0:
                    with torch.no_grad():
                        _x_ctx = x.clone()
                        for _pre_i in range(min(_rab, n_layer)):
                            _x0_w = self.x0_lambdas[_pre_i]
                            _x_inp = self.resid_lambdas[_pre_i] * _x_ctx + _x0_w * x0
                            _ve = None
                            if str(_pre_i) in self.value_embeds:
                                _ve = self.value_embeds[str(_pre_i)](idx).to(_x_inp.dtype)
                            _x_ctx = blocks[_pre_i](_x_inp, _ve, cos_sin, self.window_sizes[_pre_i], None)
                    router_input = _x_ctx  # (B, T, C) with attention context, detached
                else:
                    router_input = x0

                router_logits = global_router(
                    router_input,
                    freq_bias=freq_bias,
                    pos_bias=pos_bias,
                    freq_alpha=config.eet_freq_prior_alpha,
                    pos_beta=config.eet_pos_prior_beta
                )  # (B, T, n_exits)
                if self.exit_schedule_logits is not None:
                    router_logits = router_logits + self.exit_schedule_logits
                
                # Gumbel vs Standard Soft vs Hard
                if use_gumbel:
                    gumbel_noise = -torch.log(
                        -torch.log(torch.rand_like(router_logits.float()).clamp(1e-9)) + 1e-9
                    )
                    noisy_logits = (router_logits.float() + gumbel_noise) / eet_gumbel_temp
                    soft_weights = torch.softmax(noisy_logits, dim=-1).to(x0.dtype)  # (B, T, n_exits)
                    
                    if config.eet_gumbel_hard:
                        hard_weights = torch.zeros_like(soft_weights)
                        hard_weights.scatter_(
                            -1,
                            soft_weights.argmax(dim=-1, keepdim=True),
                            1.0
                        )
                        routing_weights = hard_weights - soft_weights.detach() + soft_weights
                    else:
                        routing_weights = soft_weights
                elif is_soft_training or is_layer_weighted:
                    if is_soft_training:
                        routing_weights = torch.softmax(router_logits.float() / eet_gumbel_temp, dim=-1).to(x0.dtype)
                    else:
                        routing_weights = torch.softmax(router_logits.float(), dim=-1).to(x0.dtype)
                    soft_weights = routing_weights
                else:
                    # Hard routing (Phase 3 or inference)
                    soft_weights = torch.softmax(router_logits.float(), dim=-1).to(x0.dtype)
                    exit_layer_hard = router_logits.argmax(dim=-1)
                    hard_weights = torch.zeros_like(router_logits)
                    hard_weights.scatter_(
                        -1,
                        exit_layer_hard.unsqueeze(-1),
                        1.0
                    )
                    if self.training:
                        routing_weights = hard_weights.to(x0.dtype) - soft_weights.detach() + soft_weights
                    else:
                        routing_weights = hard_weights.to(x0.dtype)
                        soft_weights = routing_weights

                if self.training and getattr(config, 'eet_use_override', 0) == 1:
                    step_val = eet_step
                    total_val = eet_total_steps
                    if not isinstance(step_val, torch.Tensor):
                        step_val = torch.tensor(step_val, device=x0.device, dtype=torch.float32)
                    if not isinstance(total_val, torch.Tensor):
                        total_val = torch.tensor(total_val, device=x0.device, dtype=torch.float32)
                    progress = step_val.float() / torch.clamp(total_val.float(), min=1.0)
                    p_start = getattr(config, 'eet_override_prob_start', 0.5)
                    p_end = getattr(config, 'eet_override_prob_end', 0.1)
                    p_override = torch.maximum(torch.tensor(p_end, device=x0.device), torch.tensor(p_start, device=x0.device) * (1.0 - progress))
                    
                    override_mask = (torch.rand(B, T, device=x0.device) < p_override).unsqueeze(-1)
                    forced = torch.zeros_like(routing_weights)
                    forced[:, :, -1] = 1.0
                    
                    routing_weights = torch.where(override_mask, forced, routing_weights)
                    soft_weights = torch.where(override_mask, forced, soft_weights)

                # --- Fixed capacities (compile-stable, config-driven) ---
                # Capacities are pure Python ints computed from config scalars only —
                # no tensor .item() calls, no per-batch data. This means torch.compile
                # sees the same K values every step and never recompiles.
                #
                # eet_capacity_schedule: 'uniform' | 'linear' | 'geometric'
                #   uniform:   equal tokens exit at every slot
                #   linear:    exits ramp up linearly (fewer early, more late)
                #   geometric: exits ramp geometrically (mirrors old behaviour but fixed)
                #
                # Per-slot exit fractions can be overridden via eet_exit_fracs:
                #   a list of n_exits floats (including final layer) that sum to ~1.0.
                #   e.g. [0.00, 0.18, 0.28, 0.20, 0.15, 0.12, 0.07]
                exit_fracs_cfg = getattr(config, 'eet_exit_fracs', None)
                if exit_fracs_cfg is not None and len(exit_fracs_cfg) in (n_rl, n_rl + 1):
                    # User-supplied per-slot fractions (n_rl routing slots, optionally + final)
                    routing_exit_fracs = list(exit_fracs_cfg[:n_rl])
                elif exit_fracs_cfg is not None:
                    print0(f"[EET] WARNING: eet_exit_fracs has {len(exit_fracs_cfg)} values "
                           f"but need {n_rl} or {n_rl+1} (n_rl={n_rl}). Falling back to schedule.")
                else:
                    schedule = getattr(config, 'eet_capacity_schedule', 'bell')
                    if schedule == 'uniform':
                        # Equal fraction exits at every routing slot
                        per_slot = (1.0 - target_frac) / n_rl
                        routing_exit_fracs = [per_slot] * n_rl
                    elif schedule == 'linear':
                        # Ramp up: earlier slots exit less, later slots exit more
                        # weights[k] = k+1, normalized so they sum to (1-target_frac)
                        weights = [k + 1 for k in range(n_rl)]
                        total_w = sum(weights)
                        routing_exit_fracs = [w / total_w * (1.0 - target_frac) for w in weights]
                    elif schedule == 'bell':
                        import math
                        # Symmetric Gaussian distribution centered in the middle of the routing layers
                        mid = (n_rl - 1) / 2.0
                        sigma = max(1.0, n_rl / 4.0)
                        weights = [math.exp(-((k - mid) / sigma) ** 2) for k in range(n_rl)]
                        total_w = sum(weights)
                        routing_exit_fracs = [w / total_w * (1.0 - target_frac) for w in weights]
                    else:  # geometric
                        # Each slot exits a fixed fraction of *remaining* tokens
                        # such that after n_rl slots, target_frac remain.
                        per_slot_frac = 1.0 - target_frac ** (1.0 / n_rl)
                        routing_exit_fracs = [per_slot_frac] * n_rl

                # Convert to cumulative survivor capacities (Python ints)
                survivor = 1.0
                capacities = []
                for ef in routing_exit_fracs:
                    survivor -= ef
                    capacities.append(max(1, int(survivor * T)))
                
                # Store target fractions distribution (slots + final layer)
                self._last_target_fractions = routing_exit_fracs + [1.0 - sum(routing_exit_fracs)]

            else:
                # Per-layer router fallback: same fixed schedule logic, no global soft_weights
                exit_fracs_cfg = getattr(config, 'eet_exit_fracs', None)
                if exit_fracs_cfg is not None and len(exit_fracs_cfg) in (n_rl, n_rl + 1):
                    routing_exit_fracs = list(exit_fracs_cfg[:n_rl])
                elif exit_fracs_cfg is not None:
                    print0(f"[EET] WARNING: eet_exit_fracs has {len(exit_fracs_cfg)} values "
                           f"but need {n_rl} or {n_rl+1} (n_rl={n_rl}). Falling back to schedule.")
                else:
                    schedule = getattr(config, 'eet_capacity_schedule', 'bell')
                    if schedule == 'uniform':
                        per_slot = (1.0 - target_frac) / n_rl
                        routing_exit_fracs = [per_slot] * n_rl
                    elif schedule == 'linear':
                        weights = [k + 1 for k in range(n_rl)]
                        total_w = sum(weights)
                        routing_exit_fracs = [w / total_w * (1.0 - target_frac) for w in weights]
                    elif schedule == 'bell':
                        import math
                        mid = (n_rl - 1) / 2.0
                        sigma = max(1.0, n_rl / 4.0)
                        weights = [math.exp(-((k - mid) / sigma) ** 2) for k in range(n_rl)]
                        total_w = sum(weights)
                        routing_exit_fracs = [w / total_w * (1.0 - target_frac) for w in weights]
                    else:  # geometric
                        per_slot_frac = 1.0 - target_frac ** (1.0 / n_rl)
                        routing_exit_fracs = [per_slot_frac] * n_rl
                survivor = 1.0
                for ef in routing_exit_fracs:
                    survivor -= ef
                    capacities.append(max(1, int(survivor * T)))
                self._last_target_fractions = routing_exit_fracs + [1.0 - sum(routing_exit_fracs)]

            # Pre-allocate x_final to store the representation of each token at the moment of exit.
            # Start x_final as a copy of x0. When a token exits, we scatter its latest state from x_active into x_final.
            x_final = x0.clone()
            # x_active holds the current active token representations.
            x_active = x.clone()

            # Diagnostic active counts per block
            active_counts = []

            for i, block in enumerate(blocks):
                if i == n_layer - 1:
                    K_before_reentry = K_cur

                # --- Layer 8 Reentry: restore all tokens at final layer ---
                if i == n_layer - 1 and getattr(config, 'eet_reenter_final', False):
                    # Scatter current active tokens into x_final before resetting active_idx
                    x_final = x_final.scatter(1, active_idx.unsqueeze(-1).expand(-1, -1, C), x_active)
                    
                    active_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1).contiguous()
                    K_cur = T
                    is_active = torch.ones(B, T, dtype=torch.bool, device=x.device)
                    x_active = x_final.clone()

                # Track active counts
                active_counts.append(K_cur)

                # Gather from full constants based on the current active_idx of shape (B, K_cur)
                # This is fast and has static shape (B, K_cur, C) / (B, K_cur, 1, 1)
                idx3 = active_idx.unsqueeze(-1)  # (B, K, 1)
                x0_active = torch.gather(x0, 1, idx3.expand(-1, -1, C))      # (B, K, C)

                # Residual mixing (only on active tokens)
                x0_w = self.x0_lambdas[i]
                if self._use_residual_decay and self.depth_decay_raw is not None:
                    decay_base = torch.sigmoid(self.depth_decay_raw)
                    x0_w = x0_w * (decay_base ** i)
                x_input = self.resid_lambdas[i] * x_active + x0_w * x0_active

                # Gather RoPE for active positions
                idx4 = idx3.unsqueeze(-1)  # (B, K, 1, 1)
                cos_act = torch.gather(cos_full.expand(B, -1, -1, -1), 1, idx4.expand(-1, -1, 1, rope_dim))
                sin_act = torch.gather(sin_full.expand(B, -1, -1, -1), 1, idx4.expand(-1, -1, 1, rope_dim))

                # Gather value embeddings
                ve = None
                if str(i) in self.value_embeds:
                    ve_full = self.value_embeds[str(i)](idx).to(x_input.dtype)
                    ve_dim = ve_full.size(-1)
                    ve = torch.gather(ve_full, 1, idx3.expand(-1, -1, ve_dim))

                # Run block on active tokens only
                x_out = block(x_input, ve, (cos_act, sin_act), self.window_sizes[i], kv_cache)

                # Residual mixer (active tokens only).
                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x_active.dtype)
                    mixed_active = self.residual_mixers[i](
                        x_out.transpose(1, 2)
                    ).transpose(1, 2)  # (B, K_cur, C)
                    x_out = x_out + gamma * mixed_active

                x_active = x_out

                # Construct state for commitment/quality loss if needed at this layer.
                # Skip in Phase 3 compute_skip when not needed — the scatter+norm is expensive
                # (full B,T,C operation) and runs at every routing layer for no benefit.
                need_candidate_states = is_soft_training or loss_variant == 'quality'
                if need_candidate_states and (i in routing_set or i == n_layer - 1):
                    x_at_layer_i = x_final.scatter(1, idx3.expand(-1, -1, C), x_active)
                    state_to_append = norm(x_at_layer_i)
                    candidate_states.append(state_to_append.detach())
                    hard_candidate_states.append(state_to_append.detach())

                # --- Routing: top-K tokens stay, rest exit ---
                if i in routing_set and rl_counter < n_rl:
                    K_next = capacities[rl_counter]

                    if is_global_router:
                        # Soft/hard probabilities for the current exit slot
                        p_exits_soft.append(soft_weights[:, :, rl_counter])
                        p_exits.append(routing_weights[:, :, rl_counter])

                        # Use per-slot exit probability directly (no cumulative scoring).
                        # This is the same pattern as the per-layer MoD router:
                        # higher exit_score → token wants to exit → lower continue_score.
                        exit_score_full = soft_weights[:, :, rl_counter]                        # (B, T)
                        exit_score = torch.gather(exit_score_full, 1, active_idx)                # (B, K_cur)
                        continue_score = 1.0 - exit_score                                       # (B, K_cur)

                        _, sorted_idx = torch.sort(continue_score, dim=-1, descending=True)

                        # Highest continue score stays
                        keep_local = sorted_idx[:, :K_next]
                        keep_local, _ = torch.sort(keep_local, dim=-1)

                        # Lowest continue score exits
                        exit_local = sorted_idx[:, K_next:]
                        exit_local, _ = torch.sort(exit_local, dim=-1)

                        # Exiting tokens: scatter their FULL (unweighted) state into x_final
                        exit_idx_global = torch.gather(active_idx, 1, exit_local)
                        x_exited = torch.gather(x_active, 1, exit_local.unsqueeze(-1).expand(-1, -1, C))
                        # Per-exit adapter: map this depth's representation to lm_head space
                        if len(self.eet_exit_adapters) > 0:
                            x_exited = self.eet_exit_adapters[rl_counter](x_exited)
                        if len(self.exit_gamma) > 0:
                            gamma = self.exit_gamma[rl_counter].to(x_exited.dtype)
                            beta = self.exit_beta[rl_counter].to(x_exited.dtype)
                            x_exited = x_exited * gamma + beta
                        # Backbone gradient isolation: detach exiting tokens so the main CE
                        # loss only trains the backbone through final-layer tokens. This
                        # prevents the conflicting gradient objective where early layers are
                        # pulled toward "prediction-ready" representations (for exit tokens)
                        # AND "good intermediate features" (for continuing tokens).
                        if getattr(config, 'eet_detach_exit_from_backbone', False):
                            x_exited = x_exited.detach()
                        x_final = x_final.scatter(1, exit_idx_global.unsqueeze(-1).expand(-1, -1, C), x_exited)

                        # Continuing tokens: update active_idx and x_active
                        active_idx_next = torch.gather(active_idx, 1, keep_local)  # (B, K_next)
                        x_active_kept = torch.gather(x_active, 1, keep_local.unsqueeze(-1).expand(-1, -1, C))
                        if getattr(config, 'eet_departure_summary', False) and len(self.departure_gate) > 0:
                            if K_cur - K_next > 0:
                                exit_mean = x_exited.mean(dim=1, keepdim=True)
                                gate = torch.sigmoid(self.departure_gate[rl_counter]).to(x_active_kept.dtype)
                                x_active_kept = x_active_kept + gate * exit_mean

                        # MoD-style score weighting: multiply the block's contribution by
                        # the continue weight from routing_weights (STE version) so that
                        # the task loss gradient can flow back to the router.
                        # Without this, top-K is non-differentiable and the router only
                        # gets signal from the efficiency loss (causing collapse).
                        # Formula: x = x_input + w * (x_out - x_input)
                        #   w=1 → full block output (token should continue)
                        #   w=0 → skip block (token should have exited)
                        if getattr(config, 'eet_router_task_grad', True):
                            rw = routing_weights[:, :, rl_counter]
                            cw_full = 1.0 - rw           # (B, T) with STE grad
                            cw_active = torch.gather(cw_full, 1, active_idx)            # (B, K_cur)
                            cw_kept = torch.gather(cw_active, 1, keep_local)            # (B, K_next)
                            x_input_kept = torch.gather(x_input, 1, keep_local.unsqueeze(-1).expand(-1, -1, C))
                            x_active = x_input_kept + cw_kept.unsqueeze(-1) * (x_active_kept - x_input_kept)
                        else:
                            x_active = x_active_kept

                        is_active_next = torch.zeros(B, T, dtype=torch.bool, device=x.device).scatter(1, active_idx_next, True)
                        is_active = is_active_next
                        active_idx = active_idx_next
                    else:
                        # Construct temporary full state to evaluate the per-layer router
                        x_at_layer_i = x_final.scatter(1, idx3.expand(-1, -1, C), x_active)
                        
                        # Run router on the FULL hidden state x (which has static shape B, T, C)
                        # and detach it to prevent router gradients from flowing to backbone
                        exit_prob_full = self.eet_routers[i](
                            norm(x_at_layer_i).detach() if self.training else norm(x_at_layer_i),
                            freq_bias=freq_bias, pos_bias=pos_bias,
                            freq_alpha=config.eet_freq_prior_alpha,
                            pos_beta=config.eet_pos_prior_beta,
                        )  # (B, T)

                        p_exits_soft.append(p_reach * exit_prob_full)
                        p_reach = p_reach * (1.0 - exit_prob_full)

                        # Gather the exit probability only for active tokens to make the top-K decision
                        exit_prob = torch.gather(exit_prob_full, 1, active_idx)  # (B, K_cur)

                        # Top-K by continue score (highest continue score/lowest exit prob stays)
                        continue_score = 1.0 - exit_prob  # (B, K_cur)
                        _, sorted_idx = torch.sort(continue_score, dim=-1, descending=True)

                        # Highest continue score stays
                        keep_local = sorted_idx[:, :K_next]
                        keep_local, _ = torch.sort(keep_local, dim=-1)

                        # Lowest continue score exits
                        exit_local = sorted_idx[:, K_next:]
                        exit_local, _ = torch.sort(exit_local, dim=-1)

                        # Exiting tokens: scatter their final state into x_final
                        exit_idx_global = torch.gather(active_idx, 1, exit_local)
                        x_exited = torch.gather(x_active, 1, exit_local.unsqueeze(-1).expand(-1, -1, C))
                        # Backbone gradient isolation (same as global router path)
                        if getattr(config, 'eet_detach_exit_from_backbone', False):
                            x_exited = x_exited.detach()
                        x_final = x_final.scatter(1, exit_idx_global.unsqueeze(-1).expand(-1, -1, C), x_exited)

                        # Continuing tokens: update active_idx and x_active
                        active_idx_next = torch.gather(active_idx, 1, keep_local)  # (B, K_next)
                        x_active = torch.gather(x_active, 1, keep_local.unsqueeze(-1).expand(-1, -1, C))

                        is_active_next = torch.zeros(B, T, dtype=torch.bool, device=x.device).scatter(1, active_idx_next, True)
                        exit_hard = is_active & ~is_active_next
                        p_exits.append(exit_hard.float())

                        is_active = is_active_next
                        active_idx = active_idx_next

                    K_cur = K_next
                    rl_counter += 1

            # Scatter the remaining active tokens into x_final at the very end
            # Apply the final exit adapter (last adapter = deepest exit)
            if len(self.eet_exit_adapters) > 0:
                x_active = self.eet_exit_adapters[-1](x_active)
            if len(self.exit_gamma) > 0:
                gamma = self.exit_gamma[-1].to(x_active.dtype)
                beta = self.exit_beta[-1].to(x_active.dtype)
                x_active = x_active * gamma + beta
            x_final = x_final.scatter(1, active_idx.unsqueeze(-1).expand(-1, -1, C), x_active)
            x = x_final

            # Record final exit probs
            if is_global_router:
                p_exits_soft.append(soft_weights[:, :, -1])
                p_exits.append(routing_weights[:, :, -1])
            else:
                p_exits_soft.append(p_reach)
                p_exits.append(is_active.float())

            if len(p_exits_soft) > 0:
                self._last_exit_probs = torch.stack(p_exits_soft, dim=-1)

            n_routed_layers = n_rl
            # Diagnostics: compute average active tokens across all layers (including final layer)
            avg_active = sum(active_counts) / (n_layer * T)
            total_exit_frac = (1.0 - (K_before_reentry / T)) * n_rl

            # Store enforced capacity info for final diagnostics
            self._last_enforced_capacities = capacities  # list of K values per routing slot
            self._last_active_counts = active_counts      # list of K values per block
            self._last_T = T

        else:
            # --- Hard Early Exit path (Phase 3 / Eval / Inference) ---
            hard_candidate_states = []
            
            n_kv_head = self.transformer.h[0].attn.n_kv_head
            head_dim = self.transformer.h[0].attn.head_dim
            frozen_k = torch.zeros(B, T, n_kv_head, head_dim, device=x.device, dtype=x.dtype)
            frozen_v = torch.zeros(B, T, n_kv_head, head_dim, device=x.device, dtype=x.dtype)
            
            for i, block in enumerate(blocks):
                # Layer 8 Reentry variant: force all tokens to re-enter at the final layer
                if i == n_layer - 1 and getattr(config, 'eet_reenter_final', False) and do_route:
                    token_active = torch.ones(B, T, dtype=torch.bool, device=x.device)
                
                x0_w = self.x0_lambdas[i]
                if self._use_residual_decay and self.depth_decay_raw is not None:
                    decay_base = torch.sigmoid(self.depth_decay_raw)
                    x0_w = x0_w * (decay_base ** i)
                x_input = self.resid_lambdas[i] * x + x0_w * x0
                
                prev_x_val = x_input
                ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
                
                block_active = token_active if do_route else None
                x_new = block(
                    x_input, ve, cos_sin, self.window_sizes[i], kv_cache,
                    token_active=block_active,
                    eet_frozen_kv=config.eet_frozen_kv,
                    frozen_k=frozen_k,
                    frozen_v=frozen_v
                )

                active_mask = token_active.unsqueeze(-1)
                x = torch.where(active_mask, x_new, frozen_h) if do_route else x_new

                if self.training and eet_phase == 3 and do_route:
                    if i < n_layer - 1 and i >= config.eet_min_exit_layer:
                        hard_candidate_states.append(norm(x).detach())
                    elif i == n_layer - 1:
                        hard_candidate_states.append(norm(x).detach())

                if i < n_layer - 1 and do_route and i >= config.eet_min_exit_layer:
                    exit_prob = self.eet_routers[i](
                        x.detach() if self.training else x,
                        freq_bias=freq_bias, pos_bias=pos_bias,
                        freq_alpha=config.eet_freq_prior_alpha,
                        pos_beta=config.eet_pos_prior_beta,
                    )

                    exit_candidates = (exit_prob > config.eet_exit_threshold) & token_active
                    n_max_exits = int(config.eet_max_frozen_kv_frac * B * T)
                    probs_for_cap = exit_prob * exit_candidates.float()
                    _, top_idx = probs_for_cap.view(-1).topk(n_max_exits, largest=True, sorted=False)
                    budget_mask = torch.zeros(B * T, dtype=torch.bool, device=x.device)
                    budget_mask.scatter_(0, top_idx, True)
                    exit_mask = exit_candidates & budget_mask.view(B, T)

                    frozen_h = torch.where(
                        exit_mask.unsqueeze(-1),
                        x.detach(),
                        frozen_h,
                    )
                    
                    # Capture computed keys and values for newly exited tokens
                    if config.eet_frozen_kv:
                        exit_mask_expanded = exit_mask.unsqueeze(-1).unsqueeze(-1) # (B, T, 1, 1)
                        frozen_k = torch.where(exit_mask_expanded, block.attn._last_k, frozen_k)
                        frozen_v = torch.where(exit_mask_expanded, block.attn._last_v, frozen_v)

                    token_active = token_active & ~exit_mask
                    total_exit_frac = total_exit_frac + exit_mask.float().mean()
                    n_routed_layers += 1

                    if eet_phase == 2:
                        if loss_variant == 'reconstruct':
                            translated = self.eet_translators[i](x.detach())
                            reconstruction_losses.append((translated, exit_mask.float()))
                        elif need_exit_tracking:
                            exit_hidden = torch.where(exit_mask.unsqueeze(-1), x.detach(), exit_hidden)
                            exit_layers = torch.where(exit_mask, torch.tensor(i, device=x.device), exit_layers)
                            if loss_variant == 'entropy_surprise':
                                prev_exit_hidden = torch.where(exit_mask.unsqueeze(-1), prev_x_val, prev_exit_hidden)

                if need_exit_tracking:
                    prev_x = x.detach()

                if self.residual_mixers is not None:
                    gamma = self.residual_mix_gamma[i].to(x.dtype)
                    mixed = self.residual_mixers[i](x.transpose(1, 2)).transpose(1, 2)
                    x = x + gamma * mixed

            if do_route:
                x = torch.where(token_active.unsqueeze(-1), x, frozen_h)

            if need_exit_tracking:
                never_exited = token_active.unsqueeze(-1)
                exit_hidden = torch.where(never_exited, x.detach(), exit_hidden)
                if loss_variant == 'entropy_surprise':
                    prev_exit_hidden = torch.where(never_exited, prev_x, prev_exit_hidden)

            x = norm(x)
            avg_active = token_active.float().mean()

            p_exits_soft = []
            if self.training and eet_phase == 3 and do_route:
                p_reach = torch.ones(B, T, dtype=x.dtype, device=x.device)
                routing_layers = list(range(config.eet_min_exit_layer, n_layer - 1))
                for idx, r_layer_idx in enumerate(routing_layers):
                    h_layer = hard_candidate_states[idx]
                    router = self.eet_routers[r_layer_idx]
                    exit_prob_i = router(
                        h_layer,
                        freq_bias=freq_bias, pos_bias=pos_bias,
                        freq_alpha=config.eet_freq_prior_alpha,
                        pos_beta=config.eet_pos_prior_beta,
                    )
                    p_exits_soft.append(p_reach * exit_prob_i)
                    p_reach = p_reach * (1.0 - exit_prob_i)
                p_exits_soft.append(p_reach)
                
                if len(p_exits_soft) > 0:
                    self._last_exit_probs = torch.stack(p_exits_soft, dim=-1)

        # --- LM head ---
        softcap = 20
        logits = self.lm_head(x)
        logits = logits[..., :config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            if is_layer_weighted:
                loss, loss_diags = self._compute_layer_weighted_loss(
                    all_layer_norms, exit_probs_per_layer, targets, config, eet_lambda_e,
                    freq_bias=freq_bias
                )
                if getattr(config, 'eet_commitment_beta', 0.0) > 0:
                    commit_loss = self._compute_commitment_loss(
                        all_layer_norms, p_exits, beta=config.eet_commitment_beta
                    )
                    loss = loss + commit_loss
                
                if self.training:
                    self._eet_diagnostics = {
                        'phase': eet_phase,
                        'active_frac': avg_active,
                        'total_exit_frac': total_exit_frac,
                    }
                return loss

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )

            # Store router input + per_token_ce for external ce_guided (outside compiled forward)
            if loss_variant == 'ce_guided' and is_global_router and self.training:
                # router_input is x0 or post-attention context (set in compute_skip path)
                self._last_x0_for_ce = (router_input if compute_skip else x0).detach()
                with torch.no_grad():
                    ptce = F.cross_entropy(
                        logits.detach().view(-1, logits.size(-1)), targets.view(-1),
                        ignore_index=-1, reduction='none',
                    ).view(B, T)
                self._last_per_token_ce = ptce

            # Commitment Loss for Gumbel/soft-blending
            if is_soft_training and getattr(config, 'eet_commitment_beta', 0.0) > 0:
                commit_loss = self._compute_commitment_loss(
                    candidate_states, p_exits, beta=config.eet_commitment_beta
                )
                loss = loss + commit_loss

            # --- Early Exit Variant/Auxiliary losses (Phase 2 & 3) ---
            if eet_phase in (2, 3) and loss_reduction == 'mean' and not use_gumbel:
                if eet_phase == 2:
                    if loss_variant == 'reconstruct' and reconstruction_losses:
                        recon_loss_term = self._compute_reconstruction_loss(
                            logits, reconstruction_losses, eet_lambda_r, config.vocab_size
                        )
                        loss = loss + recon_loss_term
                    elif loss_variant == 'entropy_surprise' and need_exit_tracking:
                        _detach_aux = getattr(config, 'eet_detach_aux_from_backbone', False)
                        _eh = exit_hidden.detach() if _detach_aux else exit_hidden
                        _peh = prev_exit_hidden.detach() if (_detach_aux and prev_exit_hidden is not None) else prev_exit_hidden
                        variant_loss, variant_diag = self._compute_entropy_surprise_loss(
                            _eh, _peh, config
                        )
                        loss = loss + variant_loss
                    elif loss_variant == 'adversarial' and need_exit_tracking:
                        variant_loss, variant_diag = self._compute_adversarial_loss(
                            exit_hidden, logits, targets, config
                        )
                        loss = loss + variant_loss
                        
                # Quality REINFORCE loss with entropy bonus applies to both Phase 2 and 3
                if loss_variant == 'quality':
                    curr_p_exits = p_exits if eet_phase == 2 else p_exits_soft
                    curr_candidate_states = candidate_states if eet_phase == 2 else hard_candidate_states
                    
                    if len(curr_p_exits) > 1:
                        adv_tensor, vmask = self._compute_quality_advantages(
                            curr_candidate_states, curr_p_exits, targets, config
                        )
                        quality_loss = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                        for qi, p_exit_i in enumerate(curr_p_exits):
                            quality_loss = quality_loss - (p_exit_i.float() * adv_tensor[qi]).mean()
                        loss = loss + config.eet_quality_lambda * quality_loss

                        if config.eet_quality_entropy_bonus > 0:
                            entropy_bonus = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                            for p_exit_i in curr_p_exits:
                                p_f = p_exit_i.float().clamp(min=1e-8)
                                entropy_bonus = entropy_bonus - (p_f * torch.log(p_f)).mean()
                            loss = loss - config.eet_quality_entropy_bonus * entropy_bonus



            # Gumbel auxiliary loss: uses REINFORCE quality loss & entropy bonus continuously.
            # Skip under compute_skip — candidate_states require expensive full-sequence
            # scatter+norm at every routing layer, and _compute_quality_advantages runs
            # the LM head on each candidate, negating all compute savings.
            if use_gumbel and loss_reduction == 'mean' and not compute_skip:
                if len(p_exits) > 1:
                    adv_tensor, vmask = self._compute_quality_advantages(
                        candidate_states, p_exits, targets, config
                    )
                    quality_loss = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                    for qi, p_exit_i in enumerate(p_exits):
                        quality_loss = quality_loss - (p_exit_i.float() * adv_tensor[qi]).mean()
                    loss = loss + config.eet_quality_lambda * quality_loss

                    if config.eet_quality_entropy_bonus > 0:
                        entropy_bonus = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
                        for p_exit_i in p_exits:
                            p_f = p_exit_i.float().clamp(min=1e-8)
                            entropy_bonus = entropy_bonus - (p_f * torch.log(p_f)).mean()
                        loss = loss - config.eet_quality_entropy_bonus * entropy_bonus

            # --- Efficiency loss + diversity pressure ---
            if (do_route or is_soft_training) and n_routed_layers > 0 and loss_reduction == 'mean':
                if self.training and eet_phase == 3 and not use_gumbel:
                    eff_div_loss, _ = compute_efficiency_and_diversity(
                        p_exits_soft, len(p_exits_soft), freq_bias, config, eet_lambda_e
                    )
                    loss = loss + eff_div_loss
                else:
                    eff_div_loss, _ = compute_efficiency_and_diversity(
                        p_exits, len(p_exits), freq_bias, config, eet_lambda_e
                    )
                    loss = loss + eff_div_loss

                # Capacity Alignment / Load-Balancing Loss
                cal_lambda = getattr(config, 'eet_capacity_alignment_lambda', 0.0)
                if self.training and cal_lambda > 0 and hasattr(self, '_last_target_fractions'):
                    target_dist = torch.tensor(self._last_target_fractions, device=loss.device, dtype=torch.float32)
                    curr_probs = p_exits_soft if (eet_phase == 3 and not use_gumbel) else p_exits
                    if len(curr_probs) == len(target_dist):
                        mean_probs = torch.stack([p.mean().float() for p in curr_probs])
                        capacity_loss = F.mse_loss(mean_probs, target_dist)
                        loss = loss + cal_lambda * capacity_loss.to(loss.dtype)

            # Store diagnostics
            if self.training:
                self._eet_diagnostics = {
                    'phase': eet_phase,
                    'active_frac': avg_active,
                    'total_exit_frac': total_exit_frac if is_soft_training else (total_exit_frac / max(n_routed_layers, 1)),
                }

            return loss
        else:
            return logits
    @torch.compiler.disable
    def _forward_a3d(self, idx, targets=None, kv_cache=None, loss_reduction='mean',
                     eet_phase=1, eet_lambda_r=0.0, eet_lambda_e=0.0,
                     eet_gumbel_temp=1.0, eet_step=0, eet_total_steps=1):
        """A³D: Attention-Anchored Adaptive Depth (FFN-only skip).

        Completely separate from the main forward to avoid polluting the
        torch.compile graph of the compute_skip path.
        All tokens go through attention at every layer (context preserved).
        Only FFN is selectively applied to top-K tokens per layer.
        """
        B, T = idx.size()
        config = self.config

        # --- Standard GPT embedding (same as main forward) ---
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        T_total = T0 + T
        if T_total > self.cos.size(1):
            new_len = max(T_total, self.cos.size(1) * 2)
            head_dim = config.n_embd // config.n_head
            cos, sin = self._precompute_rotary_embeddings(new_len, head_dim)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        cos_sin = self.cos[:, T0:T_total], self.sin[:, T0:T_total]

        if self.embedding_model is None:
            x = self.transformer.wte(idx)
        else:
            x, _ = self.embedding_model(idx)
        if "wpe" in self.transformer:
            x = x + self.transformer.wpe(torch.arange(T0, T_total, device=idx.device))

        x0 = x

        blocks = list(self.transformer.h)
        n_layer = len(blocks)
        C = x.size(-1)

        ffn_target_frac = getattr(config, 'eet_ffn_target_frac', 0.50)
        K_ffn = max(1, int(ffn_target_frac * T))

        # Global router: outputs per-layer FFN importance
        global_router = self.eet_routers[0]

        # Frequency / position bias for router (lazy-loaded)
        freq_bias = getattr(self, '_freq_bias_cache', None)
        pos_bias = getattr(self, '_pos_bias_cache', None)

        router_logits = global_router(
            x0,
            freq_bias=freq_bias,
            pos_bias=pos_bias,
            freq_alpha=config.eet_freq_prior_alpha,
            pos_beta=config.eet_pos_prior_beta
        )  # (B, T, n_layer)

        # Temperature-scaled sigmoid for FFN probabilities
        temp = max(getattr(config, 'eet_gumbel_temp_end', 0.1), 0.01)
        if self.training and eet_step is not None and eet_total_steps is not None:
            t_start = getattr(config, 'eet_gumbel_temp_start', 1.0)
            t_end = getattr(config, 'eet_gumbel_temp_end', 0.1)
            step_f = float(eet_step) if not isinstance(eet_step, (int, float)) else eet_step
            total_f = float(eet_total_steps) if not isinstance(eet_total_steps, (int, float)) else eet_total_steps
            progress = min(1.0, step_f / max(1, total_f))
            temp = t_start + (t_end - t_start) * progress

        ffn_probs = torch.sigmoid(router_logits.float() / temp).to(x.dtype)  # (B, T, n_layer)
        self._last_exit_probs = ffn_probs

        ffn_counts = []

        for i, block in enumerate(blocks):
            # Residual mixing
            x0_w = self.x0_lambdas[i]
            if self._use_residual_decay and self.depth_decay_raw is not None:
                decay_base = torch.sigmoid(self.depth_decay_raw)
                x0_w = x0_w * (decay_base ** i)
            x_input = self.resid_lambdas[i] * x + x0_w * x0

            # Phase A: Attention on ALL T tokens (context preserved)
            ve = self.value_embeds[str(i)](idx).to(x_input.dtype) if str(i) in self.value_embeds else None
            x = block.forward_attn_only(x_input, ve, cos_sin, self.window_sizes[i], kv_cache)

            # Phase B: FFN on top-K tokens by FFN importance
            layer_ffn_prob = ffn_probs[:, :, i]  # (B, T)
            _, sorted_idx = torch.sort(layer_ffn_prob, dim=-1, descending=True)
            ffn_idx = sorted_idx[:, :K_ffn]
            ffn_idx, _ = torch.sort(ffn_idx, dim=-1)  # maintain position order

            # Gather tokens for FFN
            idx3 = ffn_idx.unsqueeze(-1)  # (B, K_ffn, 1)
            x_ffn_in = torch.gather(x, 1, idx3.expand(-1, -1, C))

            # Run FFN on gathered subset
            x_ffn_out = block.forward_ffn_only(x_ffn_in)

            # Task gradient: weight FFN contribution by router probability (STE)
            if self.training and getattr(config, 'eet_router_task_grad', True):
                ffn_weight = torch.gather(layer_ffn_prob, 1, ffn_idx)  # (B, K_ffn)
                ffn_delta = x_ffn_out - x_ffn_in
                x_ffn_out = x_ffn_in + ffn_weight.unsqueeze(-1) * ffn_delta

            # Scatter FFN output back into full tensor
            x = x.scatter(1, idx3.expand(-1, -1, C), x_ffn_out)

            # Residual mixer (on full tensor)
            if self.residual_mixers is not None:
                gamma = self.residual_mix_gamma[i].to(x.dtype)
                mixed = self.residual_mixers[i](x.transpose(1, 2)).transpose(1, 2)
                x = x + gamma * mixed

            ffn_counts.append(K_ffn)

        x = norm(x)

        # Store diagnostics
        self._last_active_counts = ffn_counts
        self._last_T = T
        self._eet_diagnostics = {
            'phase': eet_phase,
            'active_frac': ffn_target_frac,
            'total_exit_frac': 1.0 - ffn_target_frac,
            'ffn_counts': ffn_counts,
            'a3d': True,
        }

        # Compute loss
        if targets is not None and loss_reduction == 'mean':
            x_flat = x.view(-1, x.shape[-1])
            targets_flat = targets.view(-1)
            per_token_ce = torch.zeros(B * T, device=idx.device, dtype=torch.float32)
            chunk_size = 128
            for start in range(0, B * T, chunk_size):
                end = min(start + chunk_size, B * T)
                logits_chunk = self.lm_head(x_flat[start:end])
                logits_chunk = logits_chunk[..., :config.vocab_size].float()
                logits_chunk = 20 * torch.tanh(logits_chunk / 20)
                per_token_ce[start:end] = F.cross_entropy(
                    logits_chunk, targets_flat[start:end], reduction='none'
                )
            loss = per_token_ce.mean()

            # A³D efficiency loss: penalize overall FFN usage
            if self.training:
                eet_lambda_e_t = eet_lambda_e if isinstance(eet_lambda_e, torch.Tensor) else torch.tensor(eet_lambda_e, device=loss.device)
                ffn_usage = ffn_probs.mean()
                loss = loss + eet_lambda_e_t * ffn_usage

            return loss
        else:
            logits = self.lm_head(x)
            logits = logits[..., :config.vocab_size].float()
            logits = 20 * torch.tanh(logits / 20)
            return logits

    @torch.compiler.disable
    def _compute_layer_weighted_loss(self, all_layer_norms, exit_probs_per_layer, targets, config, eet_lambda_e, freq_bias=None):
        """
        Direction 2: Per-Token Layer-Weighted Loss.
        Each layer contributes to next-token prediction weighted by its exit probability.
        Bypasses standard routing by computing expected loss over all candidate exit points in a memory-efficient single loop.
        """
        B, T = targets.shape
        
        # Compute cumulative exit probabilities
        p_exits = []
        p_reach = torch.ones(B, T, dtype=all_layer_norms[0].dtype, device=targets.device)
        for p_exit_i in exit_probs_per_layer:
            p_exits.append(p_reach * p_exit_i)
            p_reach = p_reach * (1.0 - p_exit_i)
        p_exits.append(p_reach)
        
        loss_accumulator = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
        total_backbone_loss_val = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
        total_router_loss_val = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
        
        for k, (h_k, p_k) in enumerate(zip(all_layer_norms, p_exits)):
            # Backbone prediction and loss at layer k computed using activation checkpointing
            loss_k_mean, backbone_loss_mean, router_loss_mean = checkpoint(
                compute_layer_loss,
                h_k, p_k, targets, self.lm_head, config, config.eet_quality_lambda,
                use_reentrant=False
            )
            loss_accumulator = loss_accumulator + loss_k_mean
            total_backbone_loss_val = total_backbone_loss_val + backbone_loss_mean
            total_router_loss_val = total_router_loss_val + router_loss_mean
            
        # Frequency-scaled efficiency + diversity pressure
        eff_div_loss, eff_diag = compute_efficiency_and_diversity(
            p_exits, len(p_exits), freq_bias, config, eet_lambda_e
        )
        loss = loss_accumulator + eff_div_loss
        
        return loss, {
            'weighted_lm': total_backbone_loss_val,
            'router': total_router_loss_val,
            'efficiency': eff_diag['efficiency'],
            'expected_exit': eff_diag['expected_exit'],
        }

    @torch.compiler.disable
    def _compute_commitment_loss(self, all_layer_norms, exit_probs, beta=0.25):
        """
        Direction 3: Commitment Loss.
        Push early-exit layer representations toward the full-depth representation.
        """
        h_full = all_layer_norms[-1].detach()  # freeze targets
        commit_loss = torch.tensor(0.0, device=h_full.device, dtype=torch.float32)
        
        for k, (h_k, p_k) in enumerate(zip(all_layer_norms[:-1], exit_probs[:-1])):
            mse_k = F.mse_loss(h_k, h_full, reduction='none').mean(dim=-1)  # (B, T)
            weighted_mse = (p_k.detach() * mse_k).mean()
            commit_loss = commit_loss + weighted_mse
            
        return beta * commit_loss

    @torch.compiler.disable
    def _compute_reconstruction_loss(self, logits, reconstruction_losses, eet_lambda_r, vocab_size):
        recon_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        target_logits = logits.detach().view(-1, logits.size(-1))
        for translated_h, exit_weight in reconstruction_losses:
            flat_mask = (exit_weight > 0).view(-1)
            
            # Index only exited tokens
            exited_h = translated_h.view(-1, translated_h.size(-1))[flat_mask]
            exited_target_logits = target_logits[flat_mask]
            flat_weights = exit_weight.view(-1)[flat_mask]
            
            # Project only exited tokens to vocabulary size (saving huge memory)
            trans_logits = self.lm_head(norm(exited_h))
            trans_logits = trans_logits[..., :vocab_size].float()
            
            # Compute KL divergence on exited tokens
            t_probs = F.softmax(exited_target_logits, dim=-1)
            t_log_probs = F.log_softmax(exited_target_logits, dim=-1)
            e_log_probs = F.log_softmax(trans_logits, dim=-1)
            kl_exited = (t_probs * (t_log_probs - e_log_probs)).sum(-1)
            recon_loss = recon_loss + (kl_exited * flat_weights).sum() / flat_weights.sum().clamp(min=1e-5)
            
        return eet_lambda_r * recon_loss / max(len(reconstruction_losses), 1)

    @torch.compiler.disable
    def _compute_entropy_surprise_loss(self, exit_hidden, prev_exit_hidden_or_stacked, config,
                                       routing_weights=None):
        """Variant A: Entropy + Surprise loss (eager mode, memory-efficient chunking).

        Entropy: topk approximation of vocab entropy at exit-layer hidden states.
                 Low entropy = confident → safe to exit.
        Surprise: relative norm of update between exit layer and previous layer.
                  High surprise = representation still changing → unsafe to exit.

        For global router: pass stacked=(B,T,n_exits,D) + routing_weights=(B,T,n_exits).
        For per-layer router: pass prev_exit_hidden=(B,T,D), routing_weights=None.

        Returns (loss_term, diagnostics_dict).
        """
        B, T, D = exit_hidden.shape
        topk_vocab = min(config.eet_topk_vocab, config.vocab_size)
        unembed = self.lm_head.weight  # (padded_vocab, D)

        # --- Entropy: chunked projection to cap peak at ~1 GB ---
        flat_h = exit_hidden.reshape(-1, D)  # (N, D)
        N = flat_h.shape[0]
        chunk_size = 8192
        entropy_sum = torch.tensor(0.0, device=exit_hidden.device, dtype=torch.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_logits = flat_h[start:end].float() @ unembed[:config.vocab_size].float().T
            topk_logits = chunk_logits.topk(topk_vocab, dim=-1).values
            log_p = F.log_softmax(topk_logits, dim=-1)
            entropy_sum = entropy_sum - (log_p.exp() * log_p).sum()
        entropy_loss = entropy_sum / N

        # --- Surprise: relative norm of representation update ---
        if routing_weights is not None:
            # Global router path: compute surprise from consecutive candidate states
            # prev_exit_hidden_or_stacked is stacked (B, T, n_exits, D)
            stacked = prev_exit_hidden_or_stacked
            # Shift: prev for exit i is exit i-1 (first exit uses itself → surprise=0)
            prev_stacked = torch.cat([stacked[:, :, :1], stacked[:, :, :-1]], dim=2)
            prev_exit_hidden = (routing_weights.unsqueeze(-1) * prev_stacked).sum(dim=2)
        else:
            # Per-layer router path: prev_exit_hidden passed directly
            prev_exit_hidden = prev_exit_hidden_or_stacked

        update = exit_hidden - prev_exit_hidden  # (B, T, D)
        surprise = update.norm(dim=-1) / (prev_exit_hidden.norm(dim=-1) + 1e-8)  # (B, T)
        surprise_loss = surprise.mean()

        # Entropy: minimize (exit when confident)
        # Surprise: SUBTRACT (penalize exiting when representation still changing)
        loss_term = (
            config.eet_entropy_lambda * entropy_loss
            - config.eet_surprise_lambda * surprise_loss
        )

        diag = {
            'entropy': entropy_loss.detach(),
            'surprise': surprise_loss.detach(),
        }
        return loss_term, diag

    def _compute_ce_guided_loss(self, per_token_ce, targets, stacked, routing_weights, config,
                                 router_logits=None):
        """CE-Guided Routing Loss — direct classification on router logits.

        Bypasses the softmax Jacobian bottleneck that causes router collapse.
        Instead of: router_logits → softmax → expected_depth → MSE (gradients cancel),
        uses:       router_logits → F.cross_entropy(target_exit_bin) (gradients ~0.86/logit).

        Per-token CE is binned into n_exits difficulty classes via quantiles.
        F.cross_entropy gradient = softmax(logits) - one_hot(target), which is:
          - Large (~0.86 per logit vs ~1e-6 through expected_depth)
          - Per-token (different targets → different gradient directions)
          - Non-cancelling (doesn't average away in shared weights)

        Args:
            per_token_ce:    (B, T) — detached per-token CE from main loss
            targets:         (B, T) — target token IDs
            stacked:         (B, T, n_exits, D) — candidate exit states
            routing_weights: (B, T, n_exits) — router's soft/hard exit weights
            config:          GPTConfig
            router_logits:   (B, T, n_exits) — raw router logits (before softmax)
        """
        B, T = targets.shape
        n_exits = routing_weights.shape[-1]
        device = targets.device
        valid_mask = (targets != -1).float()
        n_valid = valid_mask.sum().clamp(min=1.0)

        # --- Direct classification loss on router logits ---
        if router_logits is not None:
            with torch.no_grad():
                # Bin per-token CE into n_exits difficulty bins via z-score + sigmoid
                # (compile-friendly: no torch.quantile or torch.bucketize)
                valid_ce = per_token_ce[targets != -1]
                if valid_ce.numel() > 1:
                    ce_mean = valid_ce.mean()
                    ce_std = valid_ce.std().clamp(min=0.1)
                    # sigmoid maps z-scores to [0, 1], scale to [0, n_exits-1]
                    normalized = torch.sigmoid((per_token_ce - ce_mean) / ce_std)
                    target_exit = (normalized * n_exits).long().clamp(0, n_exits - 1)
                else:
                    target_exit = torch.zeros(B, T, device=device, dtype=torch.long)
                # Mask ignored tokens
                target_exit = target_exit * (targets != -1).long()

            # F.cross_entropy directly on router logits — bypasses softmax Jacobian
            depth_loss = F.cross_entropy(
                router_logits.float().view(-1, n_exits),
                target_exit.view(-1),
                ignore_index=-100,  # won't match any valid target_exit
                reduction='none',
            )
            # Mask and average over valid tokens only
            depth_loss = (depth_loss.view(B, T) * valid_mask).sum() / n_valid
        else:
            # Fallback: no router_logits available (e.g. layer-weighted path)
            depth_loss = torch.tensor(0.0, device=device)

        # --- Surprise: representation stability from consecutive exits ---
        exit_norms = stacked[:, :, :-1].norm(dim=-1).clamp(min=1e-8)
        diffs = (stacked[:, :, 1:] - stacked[:, :, :-1]).norm(dim=-1)
        per_exit_surprise = diffs / exit_norms

        weighted_surprise = (routing_weights[:, :, 1:].float() * per_exit_surprise).sum(dim=-1)
        surprise_loss = (weighted_surprise * valid_mask).sum() / n_valid

        loss_term = (
            config.eet_ce_guided_lambda * depth_loss
            + config.eet_surprise_lambda * surprise_loss
        )

        # Diagnostics
        with torch.no_grad():
            exit_indices = torch.arange(n_exits, device=device, dtype=torch.float32)
            exit_indices = exit_indices / max(n_exits - 1, 1)
            expected_depth = (routing_weights.float() * exit_indices).sum(dim=-1)
            ce_mean = per_token_ce[targets != -1].mean() if (targets != -1).any() else per_token_ce.mean()
            ce_std = per_token_ce[targets != -1].std() if (targets != -1).any() else torch.tensor(0.0)
            target_depth = torch.sigmoid((per_token_ce - ce_mean) / ce_std.clamp(min=0.1))

        diag = {
            'depth_loss': depth_loss.detach(),
            'surprise': surprise_loss.detach(),
            'target_depth_mean': target_depth.mean().detach(),
            'expected_depth_mean': expected_depth.mean().detach(),
        }
        return loss_term, diag

    @torch.compiler.disable
    def _compute_adversarial_loss(self, exit_hidden, full_logits, targets, config):
        """Variant B: Adversarial gap + Entropy stabilizer (eager mode).

        Adversarial: CE gap between early-exit predictions and full-depth predictions.
                     Router minimizes quality degradation from early exit.
        Entropy: same topk entropy as Variant A — stabilizes adversarial signal early.

        Uses single forward pass: exit_hidden from the loop, full_logits already computed.
        Returns (loss_term, diagnostics_dict).
        """
        B, T, D = exit_hidden.shape
        topk_vocab = config.eet_topk_vocab
        unembed = self.lm_head.weight  # (padded_vocab, D)
        vocab_size = config.vocab_size
        softcap = 20

        # --- Adversarial: early-exit LM loss vs full-depth LM loss ---
        # Compute early-exit logits via chunked norm + lm_head projection
        flat_h = exit_hidden.reshape(-1, D)  # (N, D)
        N = flat_h.shape[0]
        chunk_size = 8192
        # CE losses
        flat_targets = targets.reshape(-1)  # (B*T,)
        loss_full = F.cross_entropy(
            full_logits.detach().view(-1, full_logits.size(-1)),
            flat_targets, ignore_index=-1,
        )

        # Compute unreduced cross entropy chunk-by-chunk to save massive peak memory (avoiding B*T*V tensor)
        total_ce_early = torch.tensor(0.0, device=exit_hidden.device, dtype=torch.float32)
        total_valid = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            h_normed = norm(flat_h[start:end])  # apply same layernorm as real forward
            chunk_logits = self.lm_head(h_normed)  # (chunk, padded_V)
            chunk_logits = chunk_logits[..., :vocab_size].float()
            chunk_logits = softcap * torch.tanh(chunk_logits / softcap)
            
            chunk_targets = flat_targets[start:end]
            chunk_ce = F.cross_entropy(chunk_logits, chunk_targets, ignore_index=-1, reduction='sum')
            total_ce_early = total_ce_early + chunk_ce
            total_valid += (chunk_targets != -1).sum().item()
        
        loss_early = total_ce_early / max(total_valid, 1)

        # Gap: how much worse is early exit vs full depth? Clamp at 0 (no reward for being better)
        adversarial_loss = (loss_early - loss_full).clamp(min=0)

        # --- Entropy stabilizer (same as Variant A, cheap signal) ---
        entropies = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_logits = flat_h[start:end].float() @ unembed[:vocab_size].float().T
            topk_logits = chunk_logits.topk(topk_vocab, dim=-1).values
            probs = F.softmax(topk_logits, dim=-1)
            log_probs = F.log_softmax(topk_logits, dim=-1)
            ent = -(probs * log_probs).sum(dim=-1)
            entropies.append(ent)
        entropy_loss = torch.cat(entropies).mean()

        loss_term = (
            config.eet_adv_lambda * adversarial_loss
            + config.eet_adv_entropy_lambda * entropy_loss
        )

        diag = {
            'adversarial': adversarial_loss.detach(),
            'entropy': entropy_loss.detach(),
            'lm_early': loss_early.detach(),
            'lm_full': loss_full.detach(),
        }
        return loss_term, diag

    @torch.compiler.disable
    def _compute_quality_advantages(self, candidate_states, p_exits, targets, config):
        """Compute per-layer, per-token advantages for REINFORCE quality loss.

        Runs OUTSIDE torch.compile (heavy lm_head projections + loops that
        would cause graph breaks or very slow compilation). Returns fully
        detached tensors that are used as reward signals — no gradient flows
        through this function.

        Returns:
            advantages: (n_exits, B, T) float32 detached tensor
            valid_mask: (B, T) float32 detached tensor
        """
        B, T = targets.shape
        vocab_size = config.vocab_size
        softcap = 20
        N = B * T
        chunk_size = 4096

        flat_targets = targets.reshape(-1)  # (B*T,)
        valid_mask = (flat_targets != -1).float().reshape(B, T)  # (B, T)

        # Compute per-layer quality (negative CE) — no gradient needed
        layer_qualities = []
        with torch.no_grad():
            for state_i in candidate_states:
                flat_h = state_i.reshape(N, -1)
                ce_per_token = torch.zeros(N, dtype=torch.float32, device=flat_h.device)

                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    chunk_h = flat_h[start:end]
                    chunk_logits = self.lm_head(chunk_h)
                    chunk_logits = chunk_logits[..., :vocab_size].float()
                    chunk_logits = softcap * torch.tanh(chunk_logits / softcap)
                    chunk_targets = flat_targets[start:end]
                    chunk_ce = F.cross_entropy(
                        chunk_logits, chunk_targets,
                        ignore_index=-1, reduction='none'
                    )
                    ce_per_token[start:end] = chunk_ce

                # quality = negative CE, clamped to prevent extreme values
                quality_i = -ce_per_token.reshape(B, T).clamp(-50.0, 0.0)
                layer_qualities.append(quality_i)

        # Stack: (n_exits, B, T)
        qualities = torch.stack(layer_qualities, dim=0)

        # Baseline: expected quality under current policy
        p_stack = torch.stack([p.detach().float() for p in p_exits], dim=0)
        baseline = (p_stack * qualities).sum(dim=0)  # (B, T)

        # Advantage: how much better is layer i than expected?
        advantages = qualities - baseline.unsqueeze(0)  # (n_exits, B, T)

        # Mask invalid tokens and detach everything
        advantages = (advantages * valid_mask.unsqueeze(0)).detach()

        return advantages, valid_mask.detach()

    def num_scaling_params(self):
        """Override to include EET params in scaling count without causing assertion errors."""
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        wpe = sum(p.numel() for p in self.transformer.wpe.parameters()) if "wpe" in self.transformer else 0
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        # EET: Add router and translator parameters to transformer_matrices (and keep track of them separately)
        eet_params = sum(p.numel() for p in self.eet_routers.parameters())
        eet_params += sum(p.numel() for p in self.eet_translators.parameters())
        eet_params += sum(p.numel() for p in self.eet_exit_adapters.parameters())
        eet_params += sum(p.numel() for p in self.exit_gamma.parameters())
        eet_params += sum(p.numel() for p in self.exit_beta.parameters())
        eet_params += sum(p.numel() for p in self.departure_gate.parameters())
        if self.exit_schedule_logits is not None:
            eet_params += self.exit_schedule_logits.numel()
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters()) + eet_params

        research = 0
        if self.embedding_model is not None:
            research += sum(p.numel() for p in self.embedding_model.parameters())
        if self.aux_head is not None:
            research += sum(p.numel() for p in self.aux_head.parameters())

        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        if self.depth_decay_raw is not None:
            scalars += self.depth_decay_raw.numel()
        if self.residual_mix_gamma is not None:
            for gamma_p in self.residual_mix_gamma:
                scalars += gamma_p.numel()
        if self.residual_mixers is not None:
            for mixer in self.residual_mixers:
                scalars += sum(p.numel() for p in mixer.parameters())

        total = wte + wpe + value_embeds + lm_head + transformer_matrices + research + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch in EarlyExitGPT"
        return {
            'wte': wte,
            'wpe': wpe,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'research': research,
            'scalars': scalars,
            'eet_routers_and_translators': eet_params,
            'total': total,
        }
