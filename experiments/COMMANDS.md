# Evaluation Framework - Command Reference

## Exact Commands to Run

### 1. Evaluate Existing Unconditional Outputs

```bash
python -m experiments.eval.eval_uncond_struct \
  --gen_dir outputs/uncond/programs \
  --ref_dir data \
  --out_dir outputs/uncond/eval2 \
  --max_gen 200 \
  --expand_symmetry 1
```

This will:
- Read all `.atomforge` files from `outputs/uncond/programs`
- Evaluate up to 200 samples
- Compare against reference dataset in `data/`
- Write results to `outputs/uncond/eval2/`

### 2. Generate Conditional Tasks

```bash
python -m experiments.benchmark.make_tasks_cond \
  --out experiments/tasks/cond.jsonl \
  --n_tasks 100 \
  --ref_dir data \
  --n_samples 1 \
  --seed 42
```

This will:
- Sample 100 reference programs from `data/batch_*/*.atomforge`
- Extract constraints (composition, space group, etc.)
- Create conditional tasks in `experiments/tasks/cond.jsonl`

### 3. Run One Conditional Baseline

```bash
# First, create a small task file for testing
python -m experiments.benchmark.make_tasks_cond \
  --out experiments/tasks/cond_test.jsonl \
  --n_tasks 5 \
  --ref_dir data

# Run raw_llm baseline on first task
python -m experiments.benchmark.run_suite \
  --tasks_jsonl experiments/tasks/cond_test.jsonl \
  --runner raw_llm \
  --out_root outputs/baselines/raw_llm \
  --task_id cond_001
```

Or run all tasks:
```bash
python -m experiments.benchmark.run_suite \
  --tasks_jsonl experiments/tasks/cond_test.jsonl \
  --runner raw_llm \
  --out_root outputs/baselines/raw_llm \
  --max_tasks 5
```

### 4. Evaluate Conditional Results

After running a baseline, evaluate the results:

```bash
python -m experiments.eval.eval_cond_struct \
  --gen_dir outputs/baselines/raw_llm/cond_001/programs \
  --ref_dir data \
  --out_dir outputs/baselines/raw_llm/cond_001/eval \
  --task_jsonl experiments/tasks/cond_test.jsonl \
  --max_gen 10
```

### 5. Build Final Report

```bash
python -m experiments.benchmark.report \
  --output_root outputs \
  --out outputs/report.md
```

This scans all evaluation directories and generates a comparison report.

## Additional Useful Commands

### Generate Unconditional Tasks

```bash
python -m experiments.benchmark.make_tasks_uncond \
  --out experiments/tasks/uncond.jsonl \
  --n_samples 200 \
  --model gpt-5.2
```

### Run Tool-Use Agentic Baseline

```bash
python -m experiments.benchmark.run_suite \
  --tasks_jsonl experiments/tasks/cond_test.jsonl \
  --runner tool_use_agentic \
  --out_root outputs/baselines/tool_use_agentic \
  --max_tasks 5
```

### Run Code Interpreter Baseline

```bash
python -m experiments.benchmark.run_suite \
  --tasks_jsonl experiments/tasks/cond_test.jsonl \
  --runner code_interpreter \
  --out_root outputs/baselines/code_interpreter \
  --max_tasks 5
```

## Running Tests

```bash
# Test task schema
python tests/test_task_schema.py

# Test parsing
python tests/test_eval_parse.py

# Test fingerprint determinism
python tests/test_fingerprint.py

# Test condition checking
python tests/test_condition_check.py
```

Or with pytest (if available):
```bash
conda run -n crystal-env python -m pytest tests/test_task_schema.py tests/test_eval_parse.py tests/test_fingerprint.py tests/test_condition_check.py -v
```

## Output Structure

After running evaluations, you'll have:

```
outputs/
├── uncond/
│   ├── programs/              # Generated .atomforge files
│   └── eval2/                 # Evaluation results
│       ├── summary.json
│       ├── metrics.jsonl
│       ├── per_sample.csv
│       └── plots/
│           ├── density.png
│           └── nel.png
├── baselines/
│   ├── raw_llm/
│   │   └── cond_001/
│   │       ├── programs/
│   │       ├── logs/
│   │       └── metadata.json
│   ├── tool_use_agentic/
│   └── code_interpreter/
└── report.md                  # Final comparison report
```

## Notes

- All baseline runners work in "stub mode" if `OPENAI_API_KEY` is not set (they'll generate placeholder outputs)
- The evaluation framework supports both `.atomforge`, `.cif`, and `POSCAR` file formats
- Symmetry expansion can be toggled with `--expand_symmetry 0/1`
- Reference dataset is automatically scanned from `data/batch_*/*.atomforge`

