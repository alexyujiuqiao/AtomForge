# AtomForge Evaluation Framework

This directory contains the evaluation framework for AtomForge DSL generation experiments.

## Directory Structure

```
experiments/
├── benchmark/          # Task definition and suite runner
│   ├── task_schema.py
│   ├── make_tasks_uncond.py
│   ├── make_tasks_cond.py
│   ├── run_suite.py
│   └── report.py
├── baselines/          # Baseline generation methods
│   ├── common.py
│   ├── raw_llm.py
│   ├── tool_use_agentic.py
│   └── code_interpreter.py
├── eval/               # Evaluation metrics and scripts
│   ├── metrics.py
│   ├── failure_modes.py
│   ├── convert.py
│   ├── eval_uncond_struct.py
│   └── eval_cond_struct.py
├── unconditional_generate.py  # DSL unconditional generation
└── README.md           # This file
```

## Quick Start

### 1. Evaluate Existing Unconditional Outputs

```bash
python -m experiments.eval.eval_uncond_struct \
  --gen_dir outputs/uncond/programs \
  --ref_dir data \
  --out_dir outputs/uncond/eval2 \
  --max_gen 200
```

### 2. Generate Conditional Tasks

```bash
python -m experiments.benchmark.make_tasks_cond \
  --out experiments/tasks/cond.jsonl \
  --n_tasks 100 \
  --ref_dir data
```

### 3. Run a Baseline on Conditional Tasks

```bash
python -m experiments.benchmark.run_suite \
  --tasks_jsonl experiments/tasks/cond.jsonl \
  --runner raw_llm \
  --out_root outputs/baselines/raw_llm \
  --max_tasks 5
```

### 4. Evaluate Conditional Results

```bash
python -m experiments.eval.eval_cond_struct \
  --gen_dir outputs/baselines/raw_llm/cond_001/programs \
  --ref_dir data \
  --out_dir outputs/baselines/raw_llm/cond_001/eval \
  --task_jsonl experiments/tasks/cond.jsonl
```

### 5. Generate Final Report

```bash
python -m experiments.benchmark.report \
  --output_root outputs \
  --out outputs/report.md
```

## Task Format

Tasks are stored as JSONL files with one task per line:

```json
{
  "task_id": "uncond_001",
  "task_type": "uncond",
  "n_samples": 200,
  "seed": 42,
  "temperature": 0.8,
  "model_name": "gpt-5.2",
  "output_format": "atomforge",
  "constraints": null,
  "ablation": {"symmetry_expand": true, "use_charge_check": true}
}
```

For conditional tasks, `constraints` contains:
- `composition`: Dict of element counts
- `space_group`: Space group number
- `nel_min`, `nel_max`: Element count range
- `natoms_min`, `natoms_max`: Atom count range
- `lattice_type`: Lattice type string
- `density_range`: Dict with `min` and `max`

## Metrics

The evaluation framework computes:

- **Correctness**: parse_ok, struct_ok, min_interatomic_distance, charge_neutrality
- **Uniqueness**: StructureMatcher-based deduplication
- **Novelty**: Fingerprint comparison vs reference set
- **Distribution**: Density and element count distributions with Wasserstein distance
- **Efficiency**: LLM calls, tokens, runtime, solution size
- **Robustness**: Failure mode categorization

## Output Structure

```
outputs/
├── uncond/
│   ├── programs/       # Generated .atomforge files
│   └── eval/           # Evaluation results
│       ├── summary.json
│       ├── metrics.jsonl
│       ├── per_sample.csv
│       └── plots/
├── cond/
│   ├── programs/
│   └── eval/
├── baselines/
│   ├── raw_llm/
│   ├── tool_use_agentic/
│   └── code_interpreter/
└── report.md           # Final comparison report
```

