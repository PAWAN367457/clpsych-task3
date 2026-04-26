#!/usr/bin/env python3
"""
CLPsych 2026 Task 3 — Master Pipeline
Usage:
  python run_pipeline.py --mock --evaluate               # smoke test, no model
  python run_pipeline.py --task 31 --data data/train.json --backend ollama --model llama3:8b
  python run_pipeline.py --task 32 --data data/train.json --backend ollama --model llama3:8b
  python run_pipeline.py --data data/train.json --backend ollama --model llama3:8b --evaluate
"""

import argparse
import sys
import os

# ── Ensure src/ is on the path regardless of CWD ─────────────
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_sequences_from_summary_file, load_posts, attach_posts_to_sequences,load_test_sequences
from task31_runner import run_task31, evaluate_summaries
from task32_runner import run_task32


def parse_args():
    p = argparse.ArgumentParser(description="CLPsych 2026 Task 3 Pipeline")
    p.add_argument("--timelines", default=None, help="Path to test_tasks12nolabels/ folder")
    p.add_argument("--data",     default="data/sample_train.json", help="Path to Task 3.1 training/test JSON")
    p.add_argument("--posts",    default=None, help="Optional: path to post-level JSON (Task 1/2 output)")
    p.add_argument("--task",     default="both", choices=["31", "32", "both"], help="Which task to run")
    p.add_argument("--backend",  default="mock", choices=["mock", "ollama", "hf"], help="LLM backend")
    p.add_argument("--model",    default="llama3:8b", help="Model name (Ollama) or HF model ID")
    p.add_argument("--mock",     action="store_true", help="Shorthand: set backend=mock")
    p.add_argument("--evaluate", action="store_true", help="Run evaluation against gold summaries")
    p.add_argument("--output",   default="outputs", help="Output directory")
    args = p.parse_args()
    if args.mock:
        args.backend = "mock"
    return args


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  CLPsych 2026 Task 3 Pipeline")
    print(f"  Backend : {args.backend}  |  Model : {args.model}")
    print(f"  Data    : {args.data}")
    print(f"  Task    : {args.task}")
    print(f"{'='*55}\n")

    print("[DATA] Loading sequences...")
    if args.timelines:
        sequences = load_test_sequences(args.data, args.timelines)
    else:
        sequences = load_sequences_from_summary_file(args.data)
    print(f"       Loaded {len(sequences)} sequences from {len(set(s['timeline_id'] for s in sequences))} timelines.\n")
    if args.posts:
        print("[DATA] Loading post-level annotations...")
        posts = load_posts(args.posts)
        sequences = attach_posts_to_sequences(sequences, posts)
        print(f"       Attached posts to sequences.\n")

    # ── Task 3.1 ──────────────────────────────────────────────
    if args.task in ("31", "both"):
        print("[TASK 3.1] Generating structured narrative summaries...")
        results_31 = run_task31(
            sequences=sequences,
            backend=args.backend,
            model=args.model,
            output_dir=args.output,
        )

        if args.evaluate:
            print("\n[EVAL 3.1] Running evaluation against gold summaries...")
            evaluate_summaries(results_31, output_dir=args.output)

    # ── Task 3.2 ──────────────────────────────────────────────
    if args.task in ("32", "both"):
        print("\n[TASK 3.2] Extracting recurrent dynamic signatures...")
        run_task32(
            sequences=sequences,
            backend=args.backend,
            model=args.model,
            output_dir=args.output,
        )

    print(f"\n{'='*55}")
    print(f"  Done! Outputs saved to: {args.output}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
