#!/usr/bin/env python3
"""
Tool-Use Agentic Baseline: Iterative generation with error feedback.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from experiments.baselines.common import LLMClient, GenerationAttempt, save_artifacts, estimate_tokens
from experiments.benchmark.task_schema import Task

# Import parser for validation
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
except ImportError:
    sys.path.insert(0, str(project_root / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser


def extract_program_from_response(response: str) -> str:
    """Extract AtomForge code from LLM response."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    # Find atom_spec start
    idx = text.find('atom_spec')
    if idx >= 0:
        text = text[idx:]
    return text


def validate_program(program_text: str, parser: AtomForgeParser) -> Tuple[bool, Optional[str]]:
    """Validate program and return (success, error_message)."""
    try:
        program = parser.parse_and_transform(program_text)
        program.validate()
        return True, None
    except Exception as e:
        return False, str(e)


def generate_agentic(task: Task, client: LLMClient, max_attempts: int = 3) -> Tuple[List[GenerationAttempt], List[Tuple[str, str]]]:
    """Generate using agentic approach with error feedback."""
    attempts: List[GenerationAttempt] = []
    programs: List[Tuple[str, str]] = []
    parser = AtomForgeParser()
    
    base_prompt = """Generate a crystal structure in AtomForge DSL format.

Requirements:
- Valid AtomForge DSL v2.1 syntax
- Include: header, units, lattice, symmetry, basis
- No ai_integration block
- Unique site names (ElementSymbol+index)
- Fractional coordinates in [0,1)

Output ONLY the AtomForge code, no markdown fences.
"""
    
    if task.task_type == "cond" and task.constraints:
        base_prompt += "\nConstraints:\n"
        for key, value in task.constraints.items():
            base_prompt += f"- {key}: {value}\n"
    
    current_prompt = base_prompt
    error_history = []
    
    for attempt_num in range(1, max_attempts + 1):
        response, metadata = client.generate(current_prompt)
        
        program_text = extract_program_from_response(response)
        is_valid, error_msg = validate_program(program_text, parser)
        
        attempt = GenerationAttempt(
            attempt_num=attempt_num,
            prompt=current_prompt,
            response=response,
            tokens_in=metadata.get("tokens_in") or estimate_tokens(current_prompt),
            tokens_out=metadata.get("tokens_out") or estimate_tokens(response),
            error=error_msg if not is_valid else None,
            success=is_valid
        )
        attempts.append(attempt)
        
        if is_valid:
            filename = f"{task.task_id}_sample_001.atomforge"
            programs.append((filename, program_text))
            break
        else:
            error_history.append(f"Attempt {attempt_num}: {error_msg}")
            current_prompt = base_prompt + "\n\nPrevious attempts failed:\n" + "\n".join(error_history) + "\n\nFix the errors and try again."
    
    return attempts, programs


def main():
    parser = argparse.ArgumentParser(description="Tool-use agentic baseline runner")
    parser.add_argument('--task_jsonl', type=str, required=True, help='Task JSONL file')
    parser.add_argument('--out_root', type=str, default='outputs/baselines/tool_use_agentic', help='Output root directory')
    parser.add_argument('--task_id', type=str, default=None, help='Specific task ID to run')
    parser.add_argument('--max_attempts', type=int, default=3, help='Maximum retry attempts')
    
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
        
        attempts, programs = generate_agentic(task, client, args.max_attempts)
        
        metadata = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "model": task.model_name,
            "temperature": task.temperature,
            "n_samples": task.n_samples,
            "n_attempts": len(attempts),
            "n_programs": len(programs),
            "max_attempts": args.max_attempts
        }
        
        save_artifacts(out_root, task.task_id, attempts, programs, metadata)
        print(f"  Saved {len(programs)} programs to {out_root / task.task_id}")


if __name__ == '__main__':
    main()

