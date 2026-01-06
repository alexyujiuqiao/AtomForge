#!/usr/bin/env python3
"""
Raw LLM Baseline: One-shot generation without tool feedback.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from experiments.baselines.common import LLMClient, GenerationAttempt, save_artifacts, estimate_tokens
from experiments.benchmark.task_schema import Task


def generate_raw_llm(task: Task, client: LLMClient) -> Tuple[List[GenerationAttempt], List[Tuple[str, str]]]:
    """Generate using raw LLM (one-shot)."""
    attempts: List[GenerationAttempt] = []
    programs: List[Tuple[str, str]] = []
    
    # Simple prompt
    prompt = f"""Generate a crystal structure in AtomForge DSL format.

Requirements:
- Valid AtomForge DSL v2.1 syntax
- Include: header, units, lattice, symmetry, basis
- No ai_integration block
- Unique site names (ElementSymbol+index, e.g., O1, O2, Ti1)
- Fractional coordinates in [0,1)
- Reasonable lattice parameters

Output ONLY the AtomForge code, no markdown fences, no explanation.
"""
    
    if task.task_type == "cond" and task.constraints:
        prompt += "\nConstraints:\n"
        for key, value in task.constraints.items():
            prompt += f"- {key}: {value}\n"
    
    # Generate
    response, metadata = client.generate(prompt)
    
    attempt = GenerationAttempt(
        attempt_num=1,
        prompt=prompt,
        response=response,
        tokens_in=metadata.get("tokens_in") or estimate_tokens(prompt),
        tokens_out=metadata.get("tokens_out") or estimate_tokens(response),
        success=True
    )
    attempts.append(attempt)
    
    # Extract program (simple: assume response is the program)
    program_text = response.strip()
    # Remove markdown fences if present
    if program_text.startswith("```"):
        lines = program_text.split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        program_text = '\n'.join(lines).strip()
    
    # Save as .atomforge
    filename = f"{task.task_id}_sample_001.atomforge"
    programs.append((filename, program_text))
    
    return attempts, programs


def main():
    parser = argparse.ArgumentParser(description="Raw LLM baseline runner")
    parser.add_argument('--task_jsonl', type=str, required=True, help='Task JSONL file')
    parser.add_argument('--out_root', type=str, default='outputs/baselines/raw_llm', help='Output root directory')
    parser.add_argument('--task_id', type=str, default=None, help='Specific task ID to run (default: all)')
    
    args = parser.parse_args()
    
    from experiments.benchmark.task_schema import load_tasks
    tasks = load_tasks(args.task_jsonl)
    
    if args.task_id:
        tasks = [t for t in tasks if t.task_id == args.task_id]
    
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    for task in tasks:
        print(f"Running task: {task.task_id}")
        client = LLMClient(model=task.model_name, temperature=task.temperature)
        
        attempts, programs = generate_raw_llm(task, client)
        
        metadata = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "model": task.model_name,
            "temperature": task.temperature,
            "n_samples": task.n_samples,
            "n_attempts": len(attempts),
            "n_programs": len(programs)
        }
        
        save_artifacts(out_root, task.task_id, attempts, programs, metadata)
        print(f"  Saved {len(programs)} programs to {out_root / task.task_id}")


if __name__ == '__main__':
    main()

