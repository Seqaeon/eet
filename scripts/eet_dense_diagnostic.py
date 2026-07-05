#!/usr/bin/env python3
"""
EET vs Dense Diagnostic Comparison.

Loads trained dense and EET checkpoints, runs them on identical validation data,
and produces comprehensive comparison metrics + diagnostic plots.

Usage:
    python -m scripts.eet_dense_diagnostic \
        --dense-ckpt-dir out/eet_p01/DENSE_D8/ckpt_base/base \
        --eet-ckpt-dir out/eet_p01/EET_D8/ckpt_base/base \
        --output-dir out/eet_p01/diagnostic_D8
"""
import argparse
import json
import os
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from nanochat.gpt import GPT, GPTConfig, norm
from nanochat.common import print0
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.checkpoint_manager import load_checkpoint, find_last_step

def load_model_for_diagnostic(ckpt_dir, device, tokenizer_dir=None):
    """Load a model from checkpoint directory for diagnostic analysis."""
    step = find_last_step(ckpt_dir)
    model_data, _, meta_data = load_checkpoint(ckpt_dir, step, device, load_optimizer=False)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    cfg_kw = meta_data["model_config"]
    if "window_pattern" not in cfg_kw:
        cfg_kw["window_pattern"] = "L"
    model_config = GPTConfig(**cfg_kw)

    if getattr(model_config, 'use_eet', False):
        from nanochat.eet import EarlyExitGPT
        ModelClass = EarlyExitGPT
    else:
        ModelClass = GPT

    with torch.device("meta"):
        model = ModelClass(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=False, assign=True)
    model.eval()
    return model, model_config, meta_data, step


def dense_forward_with_hooks(model, idx):
    """Run a dense forward pass (all blocks, no routing) on any model.
    Returns (final_hidden, layer_states) where layer_states is {i: tensor}."""
    B, T = idx.shape
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    h = model.transformer.wte(idx)
    h = h.to(torch.bfloat16 if idx.device.type == 'cuda' else torch.float32)
    h = norm(h)
    x0 = h
    layer_states = {}
    for i, block in enumerate(model.transformer.h):
        x0_w = model.x0_lambdas[i]
        x_input = model.resid_lambdas[i] * h + x0_w * x0
        ve = None
        if str(i) in model.value_embeds:
            ve = model.value_embeds[str(i)](idx).to(x_input.dtype)
        h = block(x_input, ve, cos_sin, model.window_sizes[i], None)
        layer_states[i] = h.detach().float()
    final_hidden = norm(h)
    return final_hidden.detach().float(), layer_states


def compute_logits_from_hidden(model, hidden, vocab_size, chunk=512):
    """Compute softcapped logits from hidden states in chunks."""
    B, T, C = hidden.shape
    flat = hidden.reshape(-1, C)
    n = flat.size(0)
    all_logits = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lg = model.lm_head(flat[s:e].to(next(model.lm_head.parameters()).dtype))
        lg = lg[..., :vocab_size].float()
        lg = 20.0 * torch.tanh(lg / 20.0)
        all_logits.append(lg)
    return torch.cat(all_logits, dim=0).view(B, T, vocab_size)


def compute_batch_metrics(dense_model, eet_model, x, y, vocab_size, accum):
    """Compute all comparison metrics for one batch, accumulate into accum dict."""
    B, T = x.shape
    valid = (y.view(-1) != -1)

    # 1) Dense forward — hidden states + logits
    with torch.no_grad():
        dense_final, dense_layers = dense_forward_with_hooks(dense_model, x)
        dense_logits = compute_logits_from_hidden(dense_model, dense_final, vocab_size)

    # 2) EET dense-mode forward — same arch, no routing
    with torch.no_grad():
        eet_dense_final, eet_layers = dense_forward_with_hooks(eet_model, x)
        eet_dense_logits = compute_logits_from_hidden(eet_model, eet_dense_final, vocab_size)

    # 3) EET routing-mode forward
    eet_routed_loss = None
    eet_diag = {}
    if getattr(eet_model.config, 'use_eet', False):
        with torch.no_grad():
            eet_routed_loss = eet_model(x, y, eet_do_route=True, eet_phase=3)
            if hasattr(eet_model, '_eet_diagnostics'):
                eet_diag = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                            for k, v in eet_model._eet_diagnostics.items()}

    # Flatten for per-token metrics
    d_logits = dense_logits.view(-1, vocab_size)[valid]
    e_logits = eet_dense_logits.view(-1, vocab_size)[valid]
    targets = y.view(-1)[valid]
    n_valid = int(valid.sum().item())

    # Per-token CE
    d_ce = F.cross_entropy(d_logits, targets, reduction='none')
    e_ce = F.cross_entropy(e_logits, targets, reduction='none')

    accum['dense_ce_sum'] += d_ce.sum().item()
    accum['eet_dense_ce_sum'] += e_ce.sum().item()
    accum['n_tokens'] += n_valid
    if eet_routed_loss is not None:
        accum['eet_routed_loss_sum'] += eet_routed_loss.item() * n_valid

    # CE gap histogram bins
    ce_gap = (e_ce - d_ce).cpu().numpy()
    accum['ce_gaps'].extend(ce_gap.tolist())

    # CE by sequence position
    positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1).reshape(-1)[valid].cpu().numpy()
    d_ce_np = d_ce.cpu().numpy()
    e_ce_np = e_ce.cpu().numpy()
    for pos, dc, ec in zip(positions, d_ce_np, e_ce_np):
        accum['pos_dense_ce'][int(pos)] += dc
        accum['pos_eet_ce'][int(pos)] += ec
        accum['pos_count'][int(pos)] += 1

    # Top-1 agreement
    d_top1 = d_logits.argmax(dim=-1)
    e_top1 = e_logits.argmax(dim=-1)
    accum['top1_agree'] += (d_top1 == e_top1).sum().item()

    # Mean rank of dense-top-1 in EET distribution
    e_sorted = e_logits.argsort(dim=-1, descending=True)
    ranks = (e_sorted == d_top1.unsqueeze(-1)).nonzero(as_tuple=True)[1]
    accum['rank_sum'] += ranks.float().sum().item()

    # Top-5 overlap (Jaccard)
    d_top5 = d_logits.topk(5, dim=-1).indices
    e_top5 = e_logits.topk(5, dim=-1).indices
    for i in range(n_valid):
        d_set = set(d_top5[i].cpu().tolist())
        e_set = set(e_top5[i].cpu().tolist())
        accum['top5_jaccard_sum'] += len(d_set & e_set) / len(d_set | e_set)

    # KL divergence (dense → eet_dense)
    d_lp = F.log_softmax(d_logits, dim=-1)
    e_lp = F.log_softmax(e_logits, dim=-1)
    d_p = d_lp.exp()
    kl = F.kl_div(e_lp, d_p, reduction='sum').item()
    accum['kl_sum'] += kl

    # Entropy of each model's distribution
    d_ent = -(d_p * d_lp).sum(dim=-1)
    e_p = e_lp.exp()
    e_ent = -(e_p * e_lp).sum(dim=-1)
    accum['dense_entropy_sum'] += d_ent.sum().item()
    accum['eet_entropy_sum'] += e_ent.sum().item()

    # Per-layer hidden state cosine similarity
    n_layers = len(dense_layers)
    for i in range(n_layers):
        dl = dense_layers[i].reshape(-1, dense_layers[i].size(-1))
        el = eet_layers[i].reshape(-1, eet_layers[i].size(-1))
        cos_sim = F.cosine_similarity(dl, el, dim=-1).mean().item()
        frob = (dl - el).norm(dim=-1).mean().item()
        accum[f'layer_{i}_cos_sim_sum'] += cos_sim
        accum[f'layer_{i}_frob_sum'] += frob
    accum['n_batches'] += 1

    # Routing diagnostics (EET only)
    if 'exit_fracs' in eet_diag:
        ef = eet_diag['exit_fracs']
        if isinstance(ef, torch.Tensor):
            ef = ef.tolist()
        if isinstance(ef, (list, tuple)):
            for sl, frac in enumerate(ef):
                f = frac.item() if torch.is_tensor(frac) else float(frac)
                accum[f'exit_frac_slot_{sl}_sum'] += f
            accum['exit_frac_n'] += 1


def compute_gradient_metrics(model, x, y, label, grad_accum, vocab_size):
    """Run a backward pass on one batch and record per-layer gradient norms."""
    model.train()
    model.zero_grad(set_to_none=True)
    is_eet = getattr(model.config, 'use_eet', False)
    if is_eet:
        loss = model(x, y, eet_do_route=True, eet_phase=3)
    else:
        loss = model(x, y)
    loss.backward()
    for i, block in enumerate(model.transformer.h):
        total_norm = 0.0
        count = 0
        for p in block.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                count += 1
        grad_accum[f'{label}_layer_{i}_grad_norm_sum'] += math.sqrt(total_norm) if count else 0.0
    grad_accum[f'{label}_grad_batches'] += 1
    model.eval()
    model.zero_grad(set_to_none=True)


def compute_weight_metrics(dense_model, eet_model, n_layers):
    """Compare weight matrices of shared parameters between dense and EET."""
    results = {}
    for i in range(n_layers):
        d_block = dense_model.transformer.h[i]
        e_block = eet_model.transformer.h[i]
        cos_sims = []
        norm_ratios = []
        for (dn, dp), (en, ep) in zip(d_block.named_parameters(), e_block.named_parameters()):
            if dn == en and dp.shape == ep.shape:
                df = dp.data.float().reshape(-1)
                ef = ep.data.float().reshape(-1)
                cs = F.cosine_similarity(df.unsqueeze(0), ef.unsqueeze(0)).item()
                cos_sims.append(cs)
                dn_norm = df.norm().item()
                en_norm = ef.norm().item()
                if dn_norm > 0:
                    norm_ratios.append(en_norm / dn_norm)
        results[f'layer_{i}_weight_cos_sim'] = np.mean(cos_sims) if cos_sims else 0.0
        results[f'layer_{i}_weight_norm_ratio'] = np.mean(norm_ratios) if norm_ratios else 1.0
    return results


def aggregate_metrics(accum, n_layers):
    """Convert accumulated sums into final averaged metrics."""
    nt = max(accum['n_tokens'], 1)
    nb = max(accum['n_batches'], 1)
    r = {}
    r['dense_mean_ce'] = accum['dense_ce_sum'] / nt
    r['eet_dense_mean_ce'] = accum['eet_dense_ce_sum'] / nt
    r['training_gap'] = r['eet_dense_mean_ce'] - r['dense_mean_ce']
    if accum.get('eet_routed_loss_sum', 0) > 0:
        r['eet_routed_mean_ce'] = accum['eet_routed_loss_sum'] / nt
        r['routing_gap'] = r['eet_routed_mean_ce'] - r['eet_dense_mean_ce']
        r['total_gap'] = r['eet_routed_mean_ce'] - r['dense_mean_ce']
    r['top1_agreement'] = accum['top1_agree'] / nt
    r['mean_rank_displacement'] = accum['rank_sum'] / nt
    r['top5_jaccard'] = accum['top5_jaccard_sum'] / nt
    r['mean_kl_div'] = accum['kl_sum'] / nt
    r['dense_mean_entropy'] = accum['dense_entropy_sum'] / nt
    r['eet_mean_entropy'] = accum['eet_entropy_sum'] / nt
    r['n_tokens'] = nt
    r['n_batches'] = nb

    # Per-layer
    r['per_layer_cos_sim'] = []
    r['per_layer_frob'] = []
    for i in range(n_layers):
        r['per_layer_cos_sim'].append(accum.get(f'layer_{i}_cos_sim_sum', 0) / nb)
        r['per_layer_frob'].append(accum.get(f'layer_{i}_frob_sum', 0) / nb)

    # Gradient norms
    for label in ['dense', 'eet']:
        gb = max(accum.get(f'{label}_grad_batches', 1), 1)
        norms = []
        for i in range(n_layers):
            norms.append(accum.get(f'{label}_layer_{i}_grad_norm_sum', 0) / gb)
        r[f'{label}_grad_norms'] = norms
    if r.get('dense_grad_norms') and r.get('eet_grad_norms'):
        r['grad_ratio'] = [e / max(d, 1e-10) for d, e in zip(r['dense_grad_norms'], r['eet_grad_norms'])]

    # Exit fracs
    efn = max(accum.get('exit_frac_n', 1), 1)
    exit_fracs = []
    for sl in range(n_layers + 2):
        k = f'exit_frac_slot_{sl}_sum'
        if k in accum:
            exit_fracs.append(accum[k] / efn)
    if exit_fracs:
        r['mean_exit_fracs'] = exit_fracs

    # Position CE
    pos_gap = {}
    for pos in sorted(accum['pos_count'].keys()):
        cnt = accum['pos_count'][pos]
        if cnt > 0:
            pos_gap[pos] = (accum['pos_eet_ce'][pos] - accum['pos_dense_ce'][pos]) / cnt
    r['position_ce_gap'] = pos_gap

    return r


def generate_plots(results, accum, output_dir, n_layers):
    """Generate all diagnostic plots and save as PNGs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Per-layer cosine similarity
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(n_layers))
    ax.bar(layers, results['per_layer_cos_sim'], color='steelblue')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Dense vs EET: Per-Layer Hidden State Cosine Similarity')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_layer_cosine_sim.png'), dpi=150)
    plt.close(fig)

    # 2. CE gap histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    gaps = np.array(accum['ce_gaps'])
    ax.hist(gaps, bins=100, color='coral', alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('CE Gap (EET − Dense)')
    ax.set_ylabel('Count')
    ax.set_title(f'Per-Token CE Gap Distribution (mean={gaps.mean():.4f})')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_ce_gap_histogram.png'), dpi=150)
    plt.close(fig)

    # 3. CE gap by sequence position
    pos_gap = results.get('position_ce_gap', {})
    if pos_gap:
        fig, ax = plt.subplots(figsize=(12, 5))
        positions = sorted(pos_gap.keys())
        # Bin positions into groups of 64 for smoother plot
        bin_size = 64
        binned_pos, binned_gap = [], []
        for start in range(0, max(positions) + 1, bin_size):
            vals = [pos_gap[p] for p in positions if start <= p < start + bin_size]
            if vals:
                binned_pos.append(start + bin_size // 2)
                binned_gap.append(np.mean(vals))
        ax.plot(binned_pos, binned_gap, color='darkred', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Position in Sequence')
        ax.set_ylabel('Mean CE Gap (EET − Dense)')
        ax.set_title('CE Gap by Sequence Position')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '3_ce_gap_by_position.png'), dpi=150)
        plt.close(fig)

    # 4. Gradient magnitude ratio
    if 'grad_ratio' in results:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(layers, results['grad_ratio'], color='mediumseagreen')
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Gradient Ratio (EET / Dense)')
        ax.set_title('Per-Layer Gradient Magnitude Ratio')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '4_grad_ratio.png'), dpi=150)
        plt.close(fig)

    # 5. Per-layer Frobenius norm of hidden state difference
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(layers, results['per_layer_frob'], color='darkorange')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Frobenius Norm of Difference')
    ax.set_title('Dense vs EET: Per-Layer Hidden State Divergence')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_layer_frob_divergence.png'), dpi=150)
    plt.close(fig)

    # 6. Weight cosine similarity per layer
    if 'weight_metrics' in results:
        wm = results['weight_metrics']
        w_cos = [wm.get(f'layer_{i}_weight_cos_sim', 1.0) for i in layers]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(layers, w_cos, color='slateblue')
        ax.set_ylim(min(w_cos) * 0.99, 1.001)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Weight Cosine Similarity')
        ax.set_title('Dense vs EET: Per-Layer Weight Cosine Similarity')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '6_weight_cosine_sim.png'), dpi=150)
        plt.close(fig)

    # 7. Exit fraction bar chart
    if 'mean_exit_fracs' in results:
        fig, ax = plt.subplots(figsize=(10, 5))
        ef = results['mean_exit_fracs']
        slots = list(range(len(ef)))
        ax.bar(slots, ef, color='teal')
        ax.set_xlabel('Exit Slot')
        ax.set_ylabel('Mean Exit Fraction')
        ax.set_title('EET: Mean Token Exit Fraction per Slot')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '7_exit_fractions.png'), dpi=150)
        plt.close(fig)

    # 8. Summary dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Top-left: CE comparison
    ax = axes[0, 0]
    names = ['Dense', 'EET-Dense', 'EET-Routed']
    vals = [results['dense_mean_ce'], results['eet_dense_mean_ce'],
            results.get('eet_routed_mean_ce', results['eet_dense_mean_ce'])]
    colors = ['steelblue', 'coral', 'darkred']
    bars = ax.bar(names, vals, color=colors)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{b.get_height():.4f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean CE Loss')
    ax.set_title('Loss Comparison')

    # Top-right: Top-1 agreement + Jaccard
    ax = axes[0, 1]
    ax.bar(['Top-1 Agree', 'Top-5 Jaccard'],
           [results['top1_agreement'], results['top5_jaccard']],
           color=['mediumseagreen', 'darkorange'])
    ax.set_ylim(0, 1.05)
    ax.set_title('Prediction Agreement')

    # Bottom-left: Entropy comparison
    ax = axes[1, 0]
    ax.bar(['Dense Entropy', 'EET Entropy'],
           [results['dense_mean_entropy'], results['eet_mean_entropy']],
           color=['steelblue', 'coral'])
    ax.set_title('Mean Logit Entropy')

    # Bottom-right: Key gaps
    ax = axes[1, 1]
    gap_names = ['Training Gap', 'Routing Gap', 'Total Gap']
    gap_vals = [results.get('training_gap', 0),
                results.get('routing_gap', 0),
                results.get('total_gap', 0)]
    cs = ['coral' if v > 0 else 'steelblue' for v in gap_vals]
    ax.barh(gap_names, gap_vals, color=cs)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('CE Gap')
    ax.set_title('Gap Decomposition')

    fig.suptitle('EET vs Dense: Diagnostic Summary', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, '8_summary_dashboard.png'), dpi=150)
    plt.close(fig)

    print0(f"Saved {8} diagnostic plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="EET vs Dense Diagnostic Comparison")
    parser.add_argument("--dense-ckpt-dir", required=True, help="Path to dense model checkpoint dir")
    parser.add_argument("--eet-ckpt-dir", required=True, help="Path to EET model checkpoint dir")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--tokenizer-dir", default=None, help="Tokenizer directory")
    parser.add_argument("--num-batches", type=int, default=50, help="Number of validation batches")
    parser.add_argument("--num-grad-batches", type=int, default=5, help="Number of batches for gradient analysis")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--sequence-len", type=int, default=2048, help="Sequence length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print0("=" * 64)
    print0("  EET vs Dense Diagnostic Comparison")
    print0(f"  Dense: {args.dense_ckpt_dir}")
    print0(f"  EET:   {args.eet_ckpt_dir}")
    print0(f"  Output: {args.output_dir}")
    print0("=" * 64)

    # Load models
    print0("\n[1/6] Loading dense model...")
    dense_model, dense_cfg, dense_meta, dense_step = load_model_for_diagnostic(
        args.dense_ckpt_dir, device, args.tokenizer_dir)
    print0(f"  Dense model loaded (step {dense_step}, {dense_cfg.n_layer} layers, dim {dense_cfg.n_embd})")

    print0("[2/6] Loading EET model...")
    eet_model, eet_cfg, eet_meta, eet_step = load_model_for_diagnostic(
        args.eet_ckpt_dir, device, args.tokenizer_dir)
    print0(f"  EET model loaded (step {eet_step}, {eet_cfg.n_layer} layers, dim {eet_cfg.n_embd})")

    n_layers = dense_cfg.n_layer
    vocab_size = dense_cfg.vocab_size

    # Load tokenizer and data
    print0("[3/6] Loading data...")
    tokenizer = get_tokenizer(tokenizer_dir=args.tokenizer_dir)
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.sequence_len, "val", device=device, data_dir=args.data_dir)

    # Initialize accumulators
    accum = defaultdict(float)
    accum['ce_gaps'] = []
    accum['pos_dense_ce'] = defaultdict(float)
    accum['pos_eet_ce'] = defaultdict(float)
    accum['pos_count'] = defaultdict(int)

    # Main metric loop
    print0(f"[4/6] Computing metrics over {args.num_batches} batches...")
    batch_count = 0
    for x, y in loader:
        if batch_count >= args.num_batches:
            break
        compute_batch_metrics(dense_model, eet_model, x, y, vocab_size, accum)
        batch_count += 1
        if batch_count % 10 == 0:
            print0(f"  Batch {batch_count}/{args.num_batches}")

    # Gradient analysis
    print0(f"[5/6] Gradient analysis over {args.num_grad_batches} batches...")
    grad_count = 0
    loader2 = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.sequence_len, "val", device=device, data_dir=args.data_dir)
    for x, y in loader2:
        if grad_count >= args.num_grad_batches:
            break
        compute_gradient_metrics(dense_model, x, y, 'dense', accum, vocab_size)
        compute_gradient_metrics(eet_model, x, y, 'eet', accum, vocab_size)
        grad_count += 1

    # Weight comparison
    print0("  Computing weight metrics...")
    weight_m = compute_weight_metrics(dense_model, eet_model, n_layers)

    # Aggregate
    results = aggregate_metrics(accum, n_layers)
    results['weight_metrics'] = weight_m
    results['dense_step'] = dense_step
    results['eet_step'] = eet_step
    results['dense_ckpt'] = args.dense_ckpt_dir
    results['eet_ckpt'] = args.eet_ckpt_dir

    # Generate plots
    print0("[6/6] Generating plots...")
    generate_plots(results, accum, args.output_dir, n_layers)

    # Convert position_ce_gap keys to strings for JSON
    if 'position_ce_gap' in results:
        results['position_ce_gap'] = {str(k): v for k, v in results['position_ce_gap'].items()}

    # Save JSON
    json_path = os.path.join(args.output_dir, 'diagnostic_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print0(f"  Saved results to {json_path}")

    # Print summary
    print0("\n" + "=" * 64)
    print0("  DIAGNOSTIC SUMMARY")
    print0("=" * 64)
    print0(f"  Dense mean CE:       {results['dense_mean_ce']:.4f}")
    print0(f"  EET-Dense mean CE:   {results['eet_dense_mean_ce']:.4f}")
    print0(f"  Training gap:        {results['training_gap']:+.4f}")
    if 'eet_routed_mean_ce' in results:
        print0(f"  EET-Routed mean CE:  {results['eet_routed_mean_ce']:.4f}")
        print0(f"  Routing gap:         {results.get('routing_gap', 0):+.4f}")
        print0(f"  Total gap:           {results.get('total_gap', 0):+.4f}")
    print0(f"  Top-1 agreement:     {results['top1_agreement']:.4f}")
    print0(f"  Top-5 Jaccard:       {results['top5_jaccard']:.4f}")
    print0(f"  Mean rank displace:  {results['mean_rank_displacement']:.2f}")
    print0(f"  Mean KL div:         {results['mean_kl_div']:.4f}")
    print0(f"  Tokens analyzed:     {results['n_tokens']:,}")
    print0("=" * 64)


if __name__ == "__main__":
    main()
