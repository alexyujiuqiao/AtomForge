#!/usr/bin/env python3
"""
Generate Unconditional Generation Tasks

Usage:
    python -m experiments.benchmark.make_tasks_uncond --out experiments/tasks/uncond.jsonl --n_samples 200
"""

import argparse
import random
from pathlib import Path
from experiments.benchmark.task_schema import Task, save_tasks


def main():
    parser = argparse.ArgumentParser(description="Generate unconditional generation tasks")
    parser.add_argument('--out', type=str, default='experiments/tasks/uncond.jsonl', help='Output JSONL file')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of samples per task')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--model', type=str, default='gpt-5.2-pro', help='Model name')
    parser.add_argument('--task_id', type=str, default='uncond_001', help='Task ID')
    
    args = parser.parse_args()
    
    # Create output directory
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create single unconditional task
    task = Task(
        task_id=args.task_id,
        task_type="uncond",
        n_samples=args.n_samples,
        seed=args.seed,
        temperature=args.temperature,
        model_name=args.model,
        output_format="atomforge",
        constraints=None,
        ablation={"symmetry_expand": True, "use_charge_check": True}
    )
    
    task.validate()
    save_tasks([task], args.out)
    print(f"Created unconditional task: {args.task_id} ({args.n_samples} samples)")
    print(f"Saved to: {args.out}")


if __name__ == '__main__':
    main()

