#!/usr/bin/env python3
"""
Generate Conditional Generation Tasks

Samples constraints from reference dataset to create conditional generation tasks.

Usage:
    python -m experiments.benchmark.make_tasks_cond --out experiments/tasks/cond.jsonl --n_tasks 100 --ref_dir data
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from experiments.benchmark.task_schema import Task, save_tasks

# Import AtomForge parser
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
except ImportError:
    sys.path.insert(0, str(project_root / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser


def extract_constraints_from_program(program, file_path: Path) -> Dict[str, Any]:
    """Extract constraints from a parsed AtomForge program."""
    constraints: Dict[str, Any] = {}
    
    # Composition
    if program.basis and program.basis.sites:
        elements = {}
        for site in program.basis.sites:
            for sp in site.species:
                elem = sp.element
                occ = float(sp.occupancy)
                elements[elem] = elements.get(elem, 0) + occ
        constraints["composition"] = elements
    
    # Space group
    if program.symmetry:
        constraints["space_group"] = program.symmetry.space_group
    
    # Lattice type
    if program.lattice and program.lattice.bravais:
        constraints["lattice_type"] = program.lattice.bravais.type
    
    # Number of unique elements
    if constraints.get("composition"):
        constraints["nel_min"] = len(constraints["composition"])
        constraints["nel_max"] = len(constraints["composition"])
    
    # Number of sites (asymmetric unit)
    if program.basis:
        constraints["natoms_min"] = len(program.basis.sites)
        constraints["natoms_max"] = len(program.basis.sites)
    
    return constraints


def sample_reference_programs(ref_dir: Path, n_tasks: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample reference programs and extract constraints."""
    random.seed(seed)
    
    # Find all .atomforge files
    atomforge_files = sorted(ref_dir.glob("batch_*/*.atomforge"))
    if not atomforge_files:
        raise ValueError(f"No .atomforge files found in {ref_dir}")
    
    # Sample files
    sampled_files = random.sample(atomforge_files, min(n_tasks, len(atomforge_files)))
    
    parser = AtomForgeParser()
    tasks_data = []
    
    for i, file_path in enumerate(sampled_files, 1):
        try:
            program_text = file_path.read_text(encoding='utf-8')
            program = parser.parse_and_transform(program_text)
            program.validate()
            
            constraints = extract_constraints_from_program(program, file_path)
            if constraints:
                tasks_data.append({
                    "file_path": str(file_path),
                    "constraints": constraints
                })
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue
    
    return tasks_data


def main():
    parser = argparse.ArgumentParser(description="Generate conditional generation tasks")
    parser.add_argument('--out', type=str, default='experiments/tasks/cond.jsonl', help='Output JSONL file')
    parser.add_argument('--n_tasks', type=int, default=100, help='Number of conditional tasks to generate')
    parser.add_argument('--ref_dir', type=str, default='data', help='Reference dataset directory')
    parser.add_argument('--n_samples', type=int, default=1, help='Number of samples per task')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--model', type=str, default='gpt-5.2-pro', help='Model name')
    
    args = parser.parse_args()
    
    ref_path = Path(args.ref_dir)
    if not ref_path.exists():
        raise ValueError(f"Reference directory not found: {ref_path}")
    
    # Sample constraints from reference
    print(f"Sampling {args.n_tasks} tasks from {ref_path}...")
    tasks_data = sample_reference_programs(ref_path, args.n_tasks, args.seed)
    
    # Create tasks
    tasks = []
    for i, task_data in enumerate(tasks_data, 1):
        task = Task(
            task_id=f"cond_{i:03d}",
            task_type="cond",
            n_samples=args.n_samples,
            seed=args.seed + i,
            temperature=args.temperature,
            model_name=args.model,
            output_format="atomforge",
            constraints=task_data["constraints"],
            ablation={"symmetry_expand": True, "use_charge_check": True}
        )
        task.validate()
        tasks.append(task)
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tasks(tasks, args.out)
    
    print(f"Created {len(tasks)} conditional tasks")
    print(f"Saved to: {args.out}")


if __name__ == '__main__':
    main()

