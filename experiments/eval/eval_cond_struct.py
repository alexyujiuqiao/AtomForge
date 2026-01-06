#!/usr/bin/env python3
"""
Conditional Structure Evaluation

Evaluates conditional generation outputs with constraint checking.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval.eval_uncond_struct import (
    evaluate_directory,
    structure_fingerprint,
    load_reference_fingerprints
)
from experiments.eval.metrics import check_condition_violation
from experiments.eval.failure_modes import categorize_failure, FailureMode
from experiments.eval.convert import parse_atomforge_file, atomforge_to_structure

try:
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_cond_struct")


def evaluate_conditional_directory(
    gen_dir: Path,
    ref_dir: Path,
    out_dir: Path,
    task_jsonl: Path,
    max_gen: int,
    max_ref: int,
    expand_symmetry: bool = True
) -> None:
    """Evaluate conditional generation with constraint checking."""
    from experiments.benchmark.task_schema import load_tasks
    
    # Load tasks to get constraints
    tasks = load_tasks(str(task_jsonl))
    task_constraints = {task.task_id: task.constraints for task in tasks if task.constraints}
    
    # Use base evaluation but add constraint checking
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_fps = load_reference_fingerprints(ref_dir, max_ref)
    
    files = sorted(gen_dir.glob("*.atomforge"))
    if max_gen:
        files = files[:max_gen]
    
    logger.info(f"Evaluating {len(files)} conditional structures...")
    
    per_sample = []
    structures = []
    fingerprints = []
    
    for f in files:
        row: Dict[str, Any] = {"id": f.stem, "file": str(f)}
        
        # Try to infer task_id from filename
        task_id = None
        for tid in task_constraints.keys():
            if tid in f.stem:
                task_id = tid
                break
        
        parse_ok, res = parse_atomforge_file(f)
        row["parse_ok"] = parse_ok
        if not parse_ok:
            row["error"] = res
            per_sample.append(row)
            continue
        
        try:
            struct = atomforge_to_structure(res, expand_symmetry=expand_symmetry)
            row["struct_ok"] = True
        except Exception as e:
            row["struct_ok"] = False
            row["error"] = str(e)
            per_sample.append(row)
            continue
        
        structures.append(struct)
        fp = structure_fingerprint(struct)
        fingerprints.append(fp)
        
        row.update({
            "spacegroup": getattr(res.symmetry, "space_group", None) if hasattr(res, "symmetry") else None,
            "natoms": len(struct),
            "density": struct.density,
            "nel": len(struct.composition.elements),
            "formula": struct.composition.reduced_formula,
            "fingerprint": fp,
            "novel": fp not in ref_fps,
        })
        
        # Check constraints
        if task_id and task_id in task_constraints:
            violated, violation_msg = check_condition_violation(struct, task_constraints[task_id])
            row["condition_violation"] = violated
            row["violation_message"] = violation_msg
        else:
            row["condition_violation"] = False
            row["violation_message"] = None
        
        # Failure mode
        failure_mode = categorize_failure(None, row)
        row["failure_mode"] = failure_mode.value
        
        per_sample.append(row)
    
    # Save (similar to uncond but with constraint info)
    import csv
    from datetime import datetime
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "task_type": "conditional",
        "n_tasks": len(task_constraints),
        "counts": {
            "total": len(files),
            "parse_ok": sum(1 for r in per_sample if r.get("parse_ok")),
            "struct_ok": len(structures),
            "condition_satisfied": sum(1 for r in per_sample if not r.get("condition_violation", True)),
        },
    }
    
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    
    with open(out_dir / "metrics.jsonl", "w") as f:
        for row in per_sample:
            f.write(json.dumps(row, default=str) + "\n")
    
    if per_sample:
        fieldnames = sorted({k for r in per_sample for k in r.keys()})
        with open(out_dir / "per_sample.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)
    
    logger.info(f"Conditional evaluation complete. Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate conditional generation")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated files")
    parser.add_argument("--ref_dir", type=str, default="data", help="Reference dataset directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--task_jsonl", type=str, required=True, help="Task JSONL file with constraints")
    parser.add_argument("--max_gen", type=int, default=None, help="Max generated samples")
    parser.add_argument("--max_ref", type=int, default=5000, help="Max reference samples")
    parser.add_argument("--expand_symmetry", type=int, default=1, help="Expand symmetry (1) or not (0)")
    
    args = parser.parse_args()
    
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required")
    
    gen_path = Path(args.gen_dir)
    task_path = Path(args.task_jsonl)
    
    if not gen_path.exists():
        raise ValueError(f"Generation directory not found: {gen_path}")
    if not task_path.exists():
        raise ValueError(f"Task JSONL not found: {task_path}")
    
    evaluate_conditional_directory(
        gen_path,
        Path(args.ref_dir),
        Path(args.out_dir),
        task_path,
        args.max_gen,
        args.max_ref,
        expand_symmetry=bool(args.expand_symmetry)
    )


if __name__ == "__main__":
    main()

