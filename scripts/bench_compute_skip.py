#!/usr/bin/env python3
"""Benchmark: Dense GPT vs EET compute_skip forward pass.

Measures raw forward+backward wall-clock time for:
  1. Dense GPT (no EET)
  2. EET with compute_skip and target_active_frac=0.10

Usage:
  PYTHONPATH=. python scripts/bench_compute_skip.py [--depth 8] [--compile] [--warmup 10] [--iters 50]
"""
import argparse
import time
import torch
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Depth: {args.depth}, SeqLen: {args.seq_len}, BatchSize: {args.batch_size}")
    print(f"Compile: {args.compile}, Warmup: {args.warmup}, Iters: {args.iters}")
    print()

    from nanochat.gpt import GPTConfig, GPT
    from nanochat.eet import EarlyExitGPT

    # --- Dense model ---
    dense_config = GPTConfig(
        n_layer=args.depth,
        n_head=6, n_kv_head=6,
        n_embd=384,
        vocab_size=50304,
        sequence_len=args.seq_len,
        use_eet=False,
    )
    dense_model = GPT(dense_config)
    dense_model.to(device)
    dense_model.init_weights()
    dense_model.train()
    if args.compile:
        dense_model = torch.compile(dense_model)

    # --- EET model ---
    eet_config = GPTConfig(
        n_layer=args.depth,
        n_head=6, n_kv_head=6,
        n_embd=384,
        vocab_size=50304,
        sequence_len=args.seq_len,
        use_eet=True,
        eet_compute_skip=True,
        eet_global_router=True,
        eet_target_active_frac=0.10,
        eet_capacity_schedule='bell',
        eet_warmup_frac=0.0,
        eet_explore_frac=0.0,
        eet_loss_variant='ce_guided',
        eet_router_type='mlp1',
        eet_gumbel_temp_start=1.0,
        eet_gumbel_temp_end=0.1,
        eet_gumbel_hard=True,
        eet_min_exit_layer=1,
        eet_router_task_grad=True,
        eet_depth_weight_type='ema',
    )
    eet_model = EarlyExitGPT(eet_config)
    eet_model.to(device)
    eet_model.init_weights()
    eet_model.train()
    if args.compile:
        eet_model = torch.compile(eet_model)

    # Benchmark function
    def bench(model, name, is_eet=False):
        T = args.seq_len
        B = args.batch_size

        # Warmup
        for _ in range(args.warmup):
            x = torch.randint(0, 50304, (B, T), device=device)
            y = torch.randint(0, 50304, (B, T), device=device)
            if is_eet:
                loss = model(x, y,
                    eet_do_route=True,
                    eet_phase=3,
                    eet_lambda_r=torch.tensor(0.0, device=device),
                    eet_lambda_e=torch.tensor(0.1, device=device),
                    eet_gumbel_temp=torch.tensor(0.1, device=device),
                    eet_step=torch.tensor(100.0, device=device),
                    eet_total_steps=torch.tensor(1000.0, device=device),
                )
            else:
                loss = model(x, y)
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(args.iters):
            x = torch.randint(0, 50304, (B, T), device=device)
            y = torch.randint(0, 50304, (B, T), device=device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if is_eet:
                loss = model(x, y,
                    eet_do_route=True,
                    eet_phase=3,
                    eet_lambda_r=torch.tensor(0.0, device=device),
                    eet_lambda_e=torch.tensor(0.1, device=device),
                    eet_gumbel_temp=torch.tensor(0.1, device=device),
                    eet_step=torch.tensor(100.0, device=device),
                    eet_total_steps=torch.tensor(1000.0, device=device),
                )
            else:
                loss = model(x, y)
            loss.backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
            model.zero_grad()

        avg = sum(times) / len(times)
        med = sorted(times)[len(times) // 2]
        mn = min(times)
        mx = max(times)
        print(f"  {name:20s}: avg={avg:.1f}ms  med={med:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")
        return avg

    print("=" * 60)
    print("Benchmarking forward + backward")
    print("=" * 60)
    dense_avg = bench(dense_model, "Dense GPT", is_eet=False)
    eet_avg = bench(eet_model, "EET compute_skip", is_eet=True)
    
    diff = dense_avg - eet_avg
    pct = (diff / dense_avg) * 100
    print()
    print(f"  Dense avg:  {dense_avg:.1f}ms")
    print(f"  EET avg:    {eet_avg:.1f}ms")
    print(f"  Difference: {diff:+.1f}ms ({pct:+.1f}%)")
    if diff > 0:
        print(f"  → EET is {diff:.1f}ms FASTER than Dense")
    else:
        print(f"  → EET is {-diff:.1f}ms SLOWER than Dense")

if __name__ == "__main__":
    main()
