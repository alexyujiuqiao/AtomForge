#!/usr/bin/env python3
"""
Benchmark Suite Runner

Runs a task suite with a specified runner (DSL, baseline, etc.).
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

from experiments.benchmark.task_schema import load_tasks, Task

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_suite")


def run_dsl_uncond(task: Task, out_root: Path) -> None:
    """Run DSL unconditional generation (reuse existing if available)."""
    # Check if outputs already exist
    existing_dir = Path("outputs/uncond/programs")
    if existing_dir.exists() and list(existing_dir.glob("*.atomforge")):
        logger.info(f"Reusing existing DSL outputs from {existing_dir}")
        return
    
    # Otherwise would call unconditional_generate.py
    logger.warning("DSL generation not implemented in run_suite; use unconditional_generate.py directly")


def run_baseline(task: Task, runner: str, out_root: Path) -> None:
    """Run a baseline runner."""
    baseline_map = {
        "raw_llm": "experiments.baselines.raw_llm",
        "tool_use_agentic": "experiments.baselines.tool_use_agentic",
        "code_interpreter": "experiments.baselines.code_interpreter",
    }
    
    if runner not in baseline_map:
        raise ValueError(f"Unknown runner: {runner}")
    
    # Create temporary task file for this task
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        from experiments.benchmark.task_schema import save_tasks
        save_tasks([task], f.name)
        task_file = f.name
    
    try:
        # Run baseline
        module = baseline_map[runner]
        cmd = [
            sys.executable, "-m", module,
            "--task_jsonl", task_file,
            "--out_root", str(out_root),
            "--task_id", task.task_id
        ]
        
        logger.info(f"Running {runner} for task {task.task_id}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Baseline {runner} failed: {result.stderr}")
        else:
            logger.info(f"Baseline {runner} completed for task {task.task_id}")
    finally:
        Path(task_file).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument("--tasks_jsonl", type=str, required=True, help="Task JSONL file")
    parser.add_argument("--runner", type=str, required=True, 
                       choices=["dsl", "raw_llm", "tool_use_agentic", "code_interpreter"],
                       help="Runner to use")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory")
    parser.add_argument("--max_tasks", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--max_samples_per_task", type=int, default=None, help="Max samples per task")
    parser.add_argument("--task_id", type=str, default=None, help="Specific task ID to run")
    
    args = parser.parse_args()
    
    tasks = load_tasks(args.tasks_jsonl)
    
    if args.task_id:
        tasks = [t for t in tasks if t.task_id == args.task_id]
    
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    
    if args.max_samples_per_task:
        for task in tasks:
            task.n_samples = min(task.n_samples, args.max_samples_per_task)
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running {len(tasks)} tasks with runner: {args.runner}")
    
    for task in tasks:
        logger.info(f"Processing task: {task.task_id} ({task.task_type})")
        
        if args.runner == "dsl":
            if task.task_type == "uncond":
                run_dsl_uncond(task, out_root)
            else:
                logger.warning(f"DSL runner not implemented for task_type: {task.task_type}")
        else:
            run_baseline(task, args.runner, out_root)
    
    logger.info("Benchmark suite complete")


if __name__ == "__main__":
    main()

