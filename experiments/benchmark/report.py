#!/usr/bin/env python3
"""
Generate Benchmark Report

Scans evaluation outputs and generates a comparison report.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Load summary.json file."""
    try:
        return json.loads(summary_path.read_text())
    except Exception:
        return {}


def scan_evaluations(output_root: Path) -> Dict[str, Dict[str, Any]]:
    """Scan output directories for evaluation results."""
    results = {}
    
    # Scan baselines
    baselines_dir = output_root / "baselines"
    if baselines_dir.exists():
        for baseline_name in ["raw_llm", "tool_use_agentic", "code_interpreter"]:
            baseline_dir = baselines_dir / baseline_name
            if baseline_dir.exists():
                # Look for eval directories
                for task_dir in baseline_dir.iterdir():
                    if task_dir.is_dir():
                        eval_dir = task_dir / "eval"
                        if eval_dir.exists():
                            summary_path = eval_dir / "summary.json"
                            if summary_path.exists():
                                key = f"{baseline_name}/{task_dir.name}"
                                results[key] = load_summary(summary_path)
    
    # Scan DSL outputs
    for dsl_type in ["uncond", "cond"]:
        dsl_dir = output_root / dsl_type / "eval"
        if dsl_dir.exists():
            summary_path = dsl_dir / "summary.json"
            if summary_path.exists():
                results[f"dsl/{dsl_type}"] = load_summary(summary_path)
    
    return results


def generate_report(results: Dict[str, Dict[str, Any]], out_path: Path) -> None:
    """Generate markdown report."""
    lines = []
    lines.append("# Benchmark Evaluation Report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("## Summary Table\n")
    lines.append("| Runner | Parse OK | Struct OK | Unique Rate | Novelty Rate | Charge Neutral |")
    lines.append("|--------|----------|-----------|-------------|--------------|----------------|")
    
    for runner, data in results.items():
        counts = data.get("counts", {})
        validity = data.get("validity", {})
        uniqueness = data.get("uniqueness", {})
        novelty = data.get("novelty", {})
        
        parse_ok = counts.get("parse_ok", 0)
        struct_ok = counts.get("struct_ok", 0)
        total = counts.get("total", 1)
        
        parse_rate = f"{parse_ok}/{total}" if total > 0 else "N/A"
        struct_rate = f"{struct_ok}/{total}" if total > 0 else "N/A"
        unique_rate = f"{uniqueness.get('unique_rate', 0):.2%}" if uniqueness else "N/A"
        novelty_rate = f"{novelty.get('novelty_rate', 0):.2%}" if novelty else "N/A"
        charge_rate = f"{validity.get('charge_neutral_rate', 0):.2%}" if validity else "N/A"
        
        lines.append(f"| {runner} | {parse_rate} | {struct_rate} | {unique_rate} | {novelty_rate} | {charge_rate} |")
    
    lines.append("\n## Failure Mode Breakdown\n")
    lines.append("(See individual per_sample.csv files for detailed failure modes)\n")
    
    lines.append("\n## Distribution Metrics\n")
    for runner, data in results.items():
        dist = data.get("distribution", {})
        if dist:
            lines.append(f"### {runner}\n")
            if "density" in dist:
                d = dist["density"]
                lines.append(f"- Density: mean={d.get('mean', 0):.2f}, std={d.get('std', 0):.2f}")
            if "nel" in dist:
                n = dist["nel"]
                lines.append(f"- N_elements: mean={n.get('mean', 0):.1f}, std={n.get('std', 0):.1f}")
            lines.append("")
    
    lines.append("\n## How to Reproduce\n")
    lines.append("See individual evaluation directories for detailed results and reproduction commands.\n")
    
    out_path.write_text("\n".join(lines))
    print(f"Report saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--output_root", type=str, default="outputs", help="Root output directory")
    parser.add_argument("--out", type=str, default="outputs/report.md", help="Output report file")
    
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    if not output_root.exists():
        print(f"Warning: Output root not found: {output_root}")
        return
    
    results = scan_evaluations(output_root)
    
    if not results:
        print("No evaluation results found")
        return
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_report(results, out_path)


if __name__ == "__main__":
    main()

