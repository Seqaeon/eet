"""
Shared utilities for nanochat sweep scripts.

Extracted from lr_sweep.py / warmup_sweep.py / research_compare.py /
actual_lr_research_sweep.py to eliminate copy-paste.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import torch


# ---------------------------------------------------------------------------
# DDP runner
# ---------------------------------------------------------------------------

def resolve_runner() -> list[str]:
    """Return torchrun command prefix, capped to available GPUs.

    Reads NPROC_PER_NODE (set by shell scripts) or NANOCHAT_NPROC as the
    requested worker count and caps it to the number of visible CUDA devices.
    """
    nproc_requested = int(os.environ.get("NPROC_PER_NODE", os.environ.get("NANOCHAT_NPROC", 8)))
    gpu_count = max(torch.cuda.device_count(), 1)
    nproc = min(nproc_requested, gpu_count)
    if nproc < nproc_requested:
        print(
            f"[sweep] Requested {nproc_requested} DDP workers but only "
            f"{gpu_count} GPU(s) available — using {nproc}."
        )
    torchrun = shutil.which("torchrun")
    if torchrun:
        return [torchrun, "--standalone", f"--nproc_per_node={nproc}"]
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
    ]


# ---------------------------------------------------------------------------
# Architecture sizing
# ---------------------------------------------------------------------------

def model_dims(depth: int, aspect_ratio: int = 0) -> tuple[int, int, int, int]:
    """Return (aspect_ratio, head_dim, model_dim, research_dim) for a given depth.

    research_dim is ~1/8th of model_dim, rounded up to the nearest head_dim
    multiple, capped at model_dim.
    """
    if aspect_ratio <= 0:
        aspect_ratio = 57 if depth == 9 else 64
    head_dim = 128
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    raw_research_dim = max(model_dim // 8, 1)
    research_dim = ((raw_research_dim + head_dim - 1) // head_dim) * head_dim
    research_dim = min(research_dim, model_dim)
    return aspect_ratio, head_dim, model_dim, research_dim


def estimate_tokens_from_base(
    depth: int,
    target_ratio: float = 10.5,
    tokenizer_dir: str | None = None,
) -> int:
    """Chinchilla-style optimal token count: target_ratio × scaling_params.

    Mirrors base_train.py exactly (transformer_matrices + lm_head).
    """
    from nanochat.gpt import GPT, GPTConfig

    vocab_size = 32768
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_dir=tokenizer_dir)
        vocab_size = tokenizer.get_vocab_size()
    except Exception:
        pass

    _, head_dim, model_dim, _ = model_dims(depth)
    num_heads = model_dim // head_dim
    config = GPTConfig(
        sequence_len=2048, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    )
    with torch.device("meta"):
        model = GPT(config)
    counts = model.num_scaling_params()
    scaling_params = counts["transformer_matrices"] + counts["lm_head"]
    return int(scaling_params * target_ratio)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def check_and_prepare_env(args, label: str = "sweep") -> None:
    """Ensure data shards and tokenizer exist, downloading/training if needed."""
    from nanochat.common import get_base_dir
    from nanochat.dataset import resolve_data_dir, list_parquet_files

    data_dir = getattr(args, "data_dir", None) or resolve_data_dir()
    tokenizer_dir = getattr(args, "tokenizer_dir", None) or os.path.join(get_base_dir(), "tokenizer")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    shards = list_parquet_files(data_dir=data_dir)
    if not shards:
        max_shards = getattr(args, "max_shards", -1)
        num_files = max_shards if max_shards and max_shards > 0 else 2
        cmd = [sys.executable, "-m", "nanochat.dataset", "-n", str(num_files), "--data-dir", data_dir]
        print(f"[{label}] Downloading data: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_pkl):
        cmd = [
            sys.executable, "-m", "scripts.tok_train",
            "--max-chars", "10000000",
            "--data-dir", data_dir,
            "--tokenizer-dir", tokenizer_dir,
        ]
        print(f"[{label}] Training tokenizer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
