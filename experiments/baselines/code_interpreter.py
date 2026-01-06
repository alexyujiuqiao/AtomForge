#!/usr/bin/env python3
"""
Code Interpreter Baseline: LLM generates Python code (pymatgen) to build structure.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from experiments.baselines.common import LLMClient, GenerationAttempt, save_artifacts, estimate_tokens
from experiments.benchmark.task_schema import Task


def extract_code_from_response(response: str) -> str:
    """Extract Python code from LLM response."""
    text = response.strip()
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text


def run_python_code(code: str) -> Tuple[bool, str, Optional[str]]:
    """Run Python code and capture output/errors."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        script_path = f.name
    
    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr if not success else None
        return success, output, error
    except subprocess.TimeoutExpired:
        return False, "", "Execution timeout"
    except Exception as e:
        return False, "", str(e)
    finally:
        Path(script_path).unlink(missing_ok=True)


def generate_code_interpreter(task: Task, client: LLMClient) -> Tuple[List[GenerationAttempt], List[Tuple[str, str]]]:
    """Generate using code interpreter approach."""
    attempts: List[GenerationAttempt] = []
    programs: List[Tuple[str, str]] = []
    
    prompt = """Write Python code using pymatgen to create a crystal structure and export it to AtomForge DSL format.

Requirements:
- Use pymatgen.core.Structure and pymatgen.core.Lattice
- Create a valid crystal structure
- Export to AtomForge DSL v2.1 format (write the AtomForge code as a string)
- Include: header, units, lattice, symmetry, basis
- Output the AtomForge DSL code

Your code should:
1. Create a Structure object
2. Convert it to AtomForge DSL format
3. Print the AtomForge code

Output ONLY the Python code, no markdown fences, no explanation.
"""
    
    if task.task_type == "cond" and task.constraints:
        prompt += "\nConstraints:\n"
        for key, value in task.constraints.items():
            prompt += f"- {key}: {value}\n"
    
    response, metadata = client.generate(prompt)
    
    code = extract_code_from_response(response)
    success, output, error = run_python_code(code)
    
    attempt = GenerationAttempt(
        attempt_num=1,
        prompt=prompt,
        response=response,
        tokens_in=metadata.get("tokens_in") or estimate_tokens(prompt),
        tokens_out=metadata.get("tokens_out") or estimate_tokens(response),
        error=error if not success else None,
        success=success
    )
    attempts.append(attempt)
    
    if success and output:
        # Try to extract AtomForge code from output
        program_text = output.strip()
        filename = f"{task.task_id}_sample_001.atomforge"
        programs.append((filename, program_text))
    
    return attempts, programs


def main():
    parser = argparse.ArgumentParser(description="Code interpreter baseline runner")
    parser.add_argument('--task_jsonl', type=str, required=True, help='Task JSONL file')
    parser.add_argument('--out_root', type=str, default='outputs/baselines/code_interpreter', help='Output root directory')
    parser.add_argument('--task_id', type=str, default=None, help='Specific task ID to run')
    
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
        
        attempts, programs = generate_code_interpreter(task, client)
        
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

