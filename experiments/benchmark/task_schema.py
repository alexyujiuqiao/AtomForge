#!/usr/bin/env python3
"""
Task Schema for Benchmark Suite

Defines the JSONL task format for unconditional and conditional generation tasks.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union


@dataclass
class Task:
    """A single benchmark task."""
    task_id: str
    task_type: str  # "uncond" or "cond"
    n_samples: int
    seed: int
    temperature: float
    model_name: str
    output_format: str = "atomforge"  # "atomforge", "cif", "poscar"
    constraints: Optional[Dict[str, Any]] = None
    ablation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Task':
        """Create Task from dictionary."""
        return cls(**d)
    
    def validate(self) -> None:
        """Validate task fields."""
        if self.task_type not in ["uncond", "cond"]:
            raise ValueError(f"Invalid task_type: {self.task_type}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.output_format not in ["atomforge", "cif", "poscar"]:
            raise ValueError(f"Invalid output_format: {self.output_format}")
        if self.task_type == "cond" and not self.constraints:
            raise ValueError("Conditional tasks must have constraints")


def load_tasks(jsonl_path: str) -> List[Task]:
    """Load tasks from JSONL file."""
    tasks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task_dict = json.loads(line)
            tasks.append(Task.from_dict(task_dict))
    return tasks


def save_tasks(tasks: List[Task], jsonl_path: str) -> None:
    """Save tasks to JSONL file."""
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict()) + '\n')

