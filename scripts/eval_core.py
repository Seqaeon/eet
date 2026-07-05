"""
Standalone CORE evaluation script.

Loads one or more nanochat checkpoints directly from their directories and
reports per-task accuracy, centred score, and the aggregate CORE metric.

Unlike base_eval.py this script:
  * Accepts raw checkpoint directories (no nanochat model-tag indirection).
  * Can evaluate several checkpoints in one shot (pass multiple --checkpoint-dir
    values or use shell globs).
  * Writes a CSV per checkpoint to --output-dir (default: ./core_results/).
  * Prints a compact summary table at the end.

Examples
--------
# Single checkpoint, last step, all tasks, auto device
python -m scripts.eval_core --checkpoint-dir /path/to/checkpoints/d12

# Multiple checkpoints (shell glob)
python -m scripts.eval_core --checkpoint-dir /path/to/checkpoints/d*

# Specific step, fast approximate run, explicit tokeniser
python -m scripts.eval_core \\
    --checkpoint-dir /path/to/checkpoints/d24 \\
    --step 50000 \\
    --max-per-task 100 \\
    --tokenizer-dir /path/to/nanochat/tokenizer \\
    --output-dir ./core_results

# Multi-GPU via torchrun
torchrun --nproc_per_node=4 -m scripts.eval_core --checkpoint-dir /path/to/checkpoints/d24
"""
import os
import csv
import glob
import json
import time
import yaml
import random
import shutil
import zipfile
import tempfile
import argparse
import urllib.request
from pathlib import Path

import torch
from filelock import FileLock

from nanochat.common import (
    compute_init,
    compute_cleanup,
    autodetect_device_type,
    get_base_dir,
    download_file_with_lock,
)
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.core_eval import evaluate_task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print0(msg: str = "", **kwargs):
    """Print only on rank-0 to avoid duplicate output in distributed runs."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(msg, **kwargs)


def _ensure_eval_bundle() -> str:
    """Download and unpack the CORE eval bundle if it isn't already present."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")

    if os.path.exists(eval_bundle_dir):
        return eval_bundle_dir

    def _place(zip_path: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            extracted = os.path.join(tmpdir, "eval_bundle")
            shutil.move(extracted, eval_bundle_dir)
        print0(f"Eval bundle placed at: {eval_bundle_dir}")

    download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=_place)
    return eval_bundle_dir


def _load_core_config(eval_bundle_dir: str):
    """Return (tasks list, random_baselines dict) from the CORE bundle."""
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    meta_path   = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines: dict[str, float] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    return tasks, random_baselines


def _run_core(model, tokenizer, device, tasks, random_baselines,
              eval_bundle_dir: str, max_per_task: int = -1) -> dict:
    """
    Run all CORE tasks and return a results dictionary:
        {
          "results":          {label: accuracy},
          "centered_results": {label: centered},
          "core_metric":      float,
        }
    """
    data_base = os.path.join(eval_bundle_dir, "eval_data")
    results: dict[str, float] = {}
    centered_results: dict[str, float] = {}

    for task in tasks:
        t0 = time.time()
        label = task["label"]
        task_meta = {
            "task_type":             task["icl_task_type"],
            "dataset_uri":           task["dataset_uri"],
            "num_fewshot":           task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }

        print0(
            f"  {label:<35} "
            f"({task_meta['num_fewshot']}-shot, {task_meta['task_type']}) ... ",
            end="",
            flush=True,
        )

        data_path = os.path.join(data_base, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        random_bl = random_baselines.get(label, 0.0)
        centered  = (accuracy - 0.01 * random_bl) / (1.0 - 0.01 * random_bl)

        results[label]          = accuracy
        centered_results[label] = centered
        elapsed = time.time() - t0
        print0(f"acc={accuracy:.4f}  centred={centered:.4f}  ({elapsed:.1f}s)")

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}


def _write_csv(out: dict, csv_path: str, checkpoint_dir: str, step: int):
    """Write per-task results to a CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "step", "task", "accuracy", "centered"])
        for label in out["results"]:
            writer.writerow([
                checkpoint_dir,
                step,
                label,
                f"{out['results'][label]:.6f}",
                f"{out['centered_results'][label]:.6f}",
            ])
        writer.writerow([checkpoint_dir, step, "CORE", "", f"{out['core_metric']:.6f}"])
    print0(f"  → CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone CORE evaluation for nanochat checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint-dir",
        nargs="+",
        required=True,
        metavar="DIR",
        help=(
            "One or more checkpoint directories to evaluate. "
            "Each directory must contain model_<step>.pt + meta_<step>.json. "
            "Shell globs are supported (e.g. '/runs/d*')."
        ),
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: last available step).",
    )
    p.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to tokenizer directory (default: nanochat's built-in tokenizer).",
    )
    p.add_argument(
        "--max-per-task",
        type=int,
        default=-1,
        metavar="N",
        help="Max examples per task (-1 = all examples, default). Use ~100 for quick runs.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./core_results",
        metavar="DIR",
        help="Directory to write per-checkpoint CSV results (default: ./core_results).",
    )
    p.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["", "cuda", "cpu", "mps"],
        help="Device to use: cuda | cpu | mps (default: auto-detect).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile the model before evaluation (much faster on GPU, ~2-3x speedup).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Device / distributed setup ----------------------------------------
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # ---- Expand checkpoint dirs (support globs) ----------------------------
    raw_dirs: list[str] = []
    for pattern in args.checkpoint_dir:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            raw_dirs.extend(expanded)
        else:
            # treat as literal path even if glob matched nothing
            raw_dirs.append(pattern)

    checkpoint_dirs = [d for d in raw_dirs if os.path.isdir(d)]
    if not checkpoint_dirs:
        print0("ERROR: No valid checkpoint directories found.")
        print0(f"  Patterns given: {args.checkpoint_dir}")
        raise SystemExit(1)

    print0(f"\nFound {len(checkpoint_dirs)} checkpoint director(y/ies) to evaluate.")

    # ---- CORE bundle --------------------------------------------------------
    print0("\nEnsuring CORE eval bundle is present...")
    eval_bundle_dir = _ensure_eval_bundle()
    tasks, random_baselines = _load_core_config(eval_bundle_dir)
    print0(f"Loaded {len(tasks)} CORE tasks.")

    # ---- Summary table accumulator -----------------------------------------
    summary: list[dict] = []   # [{name, step, core_metric}]

    # ---- Evaluate each checkpoint ------------------------------------------
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        name = os.path.basename(checkpoint_dir)

        step = args.step
        if step is None:
            try:
                step = find_last_step(checkpoint_dir)
            except FileNotFoundError as e:
                print0(f"\nWARNING: Skipping {checkpoint_dir} — {e}")
                continue

        print0(f"\n{'='*70}")
        print0(f"Checkpoint : {checkpoint_dir}")
        print0(f"Step       : {step:,}")
        print0(f"{'='*70}")

        # Load model
        try:
            model, tokenizer, meta_data = build_model(
                checkpoint_dir, step, device, phase="eval",
                tokenizer_dir=args.tokenizer_dir,
            )
        except Exception as e:
            print0(f"ERROR loading checkpoint: {e}")
            continue

        # Compile for speed (huge difference on H100 / Ampere+)
        if args.compile:
            print0("  Compiling model with torch.compile ...")
            model = torch.compile(model)
            # Warmup pass to trigger compilation before timing starts
            print0("  Warmup forward pass ...", end="", flush=True)
            warmup_ids = torch.randint(0, 256, (1, 64), device=device)
            with torch.no_grad():
                _ = model(warmup_ids)
            del warmup_ids
            if device_type == "cuda":
                torch.cuda.synchronize()
            print0(" done.")

        # Run CORE
        out = _run_core(
            model, tokenizer, device,
            tasks, random_baselines, eval_bundle_dir,
            max_per_task=args.max_per_task,
        )

        core_metric = out["core_metric"]
        print0(f"\n  CORE metric: {core_metric:.4f}")

        # Write CSV (rank-0 only)
        if ddp_rank == 0:
            csv_name = f"{name}_step{step:06d}_core.csv"
            csv_path = os.path.join(args.output_dir, csv_name)
            _write_csv(out, csv_path, checkpoint_dir, step)

        summary.append({"name": name, "step": step, "core_metric": core_metric})

        # Free model memory before next checkpoint
        del model
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # ---- Summary table (rank-0 only) ---------------------------------------
    if ddp_rank == 0 and summary:
        print0(f"\n{'='*70}")
        print0("SUMMARY")
        print0(f"{'='*70}")
        print0(f"  {'Checkpoint':<30}  {'Step':>8}  {'CORE':>8}")
        print0(f"  {'-'*30}  {'-'*8}  {'-'*8}")
        for row in summary:
            print0(f"  {row['name']:<30}  {row['step']:>8,}  {row['core_metric']:>8.4f}")
        print0()

    compute_cleanup()


if __name__ == "__main__":
    main()
