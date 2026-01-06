#!/usr/bin/env python3
"""
Unconditional Structure Evaluation

Evaluates unconditional generation outputs using the unified metrics framework.
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval.convert import parse_atomforge_file, atomforge_to_structure, cif_to_structure, poscar_to_structure
from experiments.eval.metrics import (
    compute_min_interatomic_distance,
    check_charge_neutrality,
    compute_uniqueness,
    compute_novelty,
    compute_distribution_stats,
    compute_wasserstein_distance
)
from experiments.eval.failure_modes import categorize_failure, FailureMode

try:
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_uncond_struct")


def structure_fingerprint(structure: Structure) -> str:
    """Generate structure fingerprint."""
    import hashlib
    lattice = structure.lattice
    sig = {
        "lattice": {
            "a": round(lattice.a, 4),
            "b": round(lattice.b, 4),
            "c": round(lattice.c, 4),
            "alpha": round(lattice.alpha, 4),
            "beta": round(lattice.beta, 4),
            "gamma": round(lattice.gamma, 4),
        },
        "sites": []
    }
    for site in structure:
        coords = tuple(round(x, 4) for x in site.frac_coords)
        species_entries = []
        if hasattr(site.specie, "items"):
            for elem, occ in site.specie.items():
                species_entries.append((str(elem), round(float(occ), 4)))
        else:
            species_entries.append((str(site.specie), 1.0))
        species_entries.sort()
        sig["sites"].append({"position": coords, "species": species_entries})
    sig["sites"].sort(key=lambda s: (tuple(s["species"]), s["position"]))
    canonical = json.dumps(sig, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_reference_fingerprints(ref_dir: Path, max_ref: int) -> Set[str]:
    """Load reference structure fingerprints."""
    logger.info(f"Loading reference fingerprints from {ref_dir}...")
    files = sorted(ref_dir.glob("batch_*/*.atomforge"))
    if not files:
        logger.warning(f"No reference files found in {ref_dir}")
        return set()
    
    if max_ref:
        files = files[:max_ref]
    
    fingerprints = set()
    for f in files:
        ok, res = parse_atomforge_file(f)
        if not ok:
            continue
        try:
            struct = atomforge_to_structure(res, expand_symmetry=True)
            fp = structure_fingerprint(struct)
            fingerprints.add(fp)
        except Exception:
            continue
    
    logger.info(f"Loaded {len(fingerprints)} reference fingerprints")
    return fingerprints


def evaluate_directory(
    gen_dir: Path,
    ref_dir: Path,
    out_dir: Path,
    max_gen: int,
    max_ref: int,
    expand_symmetry: bool = True
) -> None:
    """Evaluate all structures in generation directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference
    ref_fps = load_reference_fingerprints(ref_dir, max_ref)
    
    # Find generated files
    files = sorted(gen_dir.glob("*.atomforge"))
    if not files:
        # Try CIF or POSCAR
        files = sorted(gen_dir.glob("*.cif")) + sorted(gen_dir.glob("POSCAR*"))
    
    if max_gen:
        files = files[:max_gen]
    
    logger.info(f"Evaluating {len(files)} generated structures...")
    
    per_sample = []
    structures = []
    fingerprints = []
    
    for f in files:
        row: Dict[str, Any] = {"id": f.stem, "file": str(f)}
        res = None  # Will hold parsed program if .atomforge
        
        # Parse
        if f.suffix == ".atomforge":
            parse_ok, res = parse_atomforge_file(f)
            row["parse_ok"] = parse_ok
            if not parse_ok:
                row["error"] = res
                per_sample.append(row)
                continue
            
            # Convert to structure
            try:
                struct = atomforge_to_structure(res, expand_symmetry=expand_symmetry)
                row["struct_ok"] = True
            except Exception as e:
                row["struct_ok"] = False
                row["error"] = str(e)
                per_sample.append(row)
                continue
        elif f.suffix == ".cif":
            try:
                struct = cif_to_structure(f)
                row["parse_ok"] = True
                row["struct_ok"] = True
            except Exception as e:
                row["parse_ok"] = False
                row["struct_ok"] = False
                row["error"] = str(e)
                per_sample.append(row)
                continue
        elif "POSCAR" in f.name:
            try:
                struct = poscar_to_structure(f)
                row["parse_ok"] = True
                row["struct_ok"] = True
            except Exception as e:
                row["parse_ok"] = False
                row["struct_ok"] = False
                row["error"] = str(e)
                per_sample.append(row)
                continue
        else:
            row["parse_ok"] = False
            row["error"] = "Unknown file format"
            per_sample.append(row)
            continue
        
        # Compute metrics
        structures.append(struct)
        fp = structure_fingerprint(struct)
        fingerprints.append(fp)
        
        spacegroup = None
        if res is not None:
            try:
                spacegroup = getattr(res, "symmetry", None)
                if spacegroup:
                    spacegroup = getattr(spacegroup, "space_group", None)
            except:
                pass
        
        row.update({
            "spacegroup": spacegroup,
            "natoms": len(struct),
            "density": struct.density,
            "nel": len(struct.composition.elements),
            "formula": struct.composition.reduced_formula,
        })
        
        min_dist = compute_min_interatomic_distance(struct)
        row["min_dist"] = min_dist
        row["valid_min_distance"] = min_dist > 0.5
        
        charge_status, net_charge = check_charge_neutrality(struct)
        row["charge_status"] = charge_status
        row["net_charge"] = net_charge
        row["charge_neutral"] = charge_status == "neutral"
        
        row["fingerprint"] = fp
        row["novel"] = fp not in ref_fps
        
        # Failure mode
        failure_mode = categorize_failure(None, row)
        row["failure_mode"] = failure_mode.value
        
        per_sample.append(row)
    
    # Uniqueness
    unique_indices, unique_rate = compute_uniqueness(structures, symprec=0.2)
    for i in unique_indices:
        if i < len(per_sample):
            per_sample[i]["unique"] = True
    for i, row in enumerate(per_sample):
        if "unique" not in row:
            row["unique"] = False
    
    # Summary
    successful = [r for r in per_sample if r.get("struct_ok")]
    novel_count, novelty_rate = compute_novelty(fingerprints, ref_fps)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "gen_dir": str(gen_dir),
            "ref_dir": str(ref_dir),
            "max_gen": max_gen,
            "max_ref": max_ref,
            "expand_symmetry": expand_symmetry,
        },
        "counts": {
            "total": len(files),
            "parse_ok": sum(1 for r in per_sample if r.get("parse_ok")),
            "struct_ok": len(successful),
        },
        "validity": {
            "min_dist_pass": sum(1 for r in successful if r.get("valid_min_distance")),
            "min_dist_rate": sum(1 for r in successful if r.get("valid_min_distance")) / max(len(successful), 1),
            "charge_neutral": sum(1 for r in successful if r.get("charge_neutral")),
            "charge_neutral_rate": sum(1 for r in successful if r.get("charge_neutral")) / max(len(successful), 1),
        },
        "uniqueness": {
            "unique_structures": len(unique_indices),
            "unique_rate": unique_rate,
        },
        "novelty": {
            "novel_structures": novel_count,
            "novelty_rate": novelty_rate,
        },
        "distribution": {
            "density": compute_distribution_stats([r.get("density", 0) for r in successful if "density" in r]),
            "nel": compute_distribution_stats([r.get("nel", 0) for r in successful if "nel" in r]),
        },
    }
    
    # Save outputs
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    
    metrics_jsonl = out_dir / "metrics.jsonl"
    with open(metrics_jsonl, "w") as f:
        for row in per_sample:
            f.write(json.dumps(row, default=str) + "\n")
    
    if per_sample:
        fieldnames = sorted({k for r in per_sample for k in r.keys()})
        with open(out_dir / "per_sample.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)
    
    # Plots
    if MATPLOTLIB_AVAILABLE:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        densities = [r.get("density") for r in successful if "density" in r]
        if densities:
            plt.figure(figsize=(8, 5))
            plt.hist(densities, bins=30, alpha=0.7)
            plt.xlabel("Density (g/cc)")
            plt.ylabel("Count")
            plt.title("Generated Density Distribution")
            plt.tight_layout()
            plt.savefig(plots_dir / "density.png", dpi=150)
            plt.close()
        
        nels = [r.get("nel") for r in successful if "nel" in r]
        if nels:
            plt.figure(figsize=(8, 5))
            plt.hist(nels, bins=range(1, max(nels) + 2), alpha=0.7, align="left")
            plt.xlabel("Number of unique elements")
            plt.ylabel("Count")
            plt.title("Generated Unique Elements Distribution")
            plt.tight_layout()
            plt.savefig(plots_dir / "nel.png", dpi=150)
            plt.close()
    
    logger.info(f"Evaluation complete. Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unconditional generation")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated files")
    parser.add_argument("--ref_dir", type=str, default="data", help="Reference dataset directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_gen", type=int, default=None, help="Max generated samples")
    parser.add_argument("--max_ref", type=int, default=5000, help="Max reference samples")
    parser.add_argument("--expand_symmetry", type=int, default=1, help="Expand symmetry (1) or not (0)")
    
    args = parser.parse_args()
    
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required")
    
    gen_path = Path(args.gen_dir)
    if not gen_path.exists():
        raise ValueError(f"Generation directory not found: {gen_path}")
    
    ref_path = Path(args.ref_dir)
    if not ref_path.exists():
        logger.warning(f"Reference directory not found: {ref_path}")
    
    evaluate_directory(
        gen_path,
        ref_path,
        Path(args.out_dir),
        args.max_gen,
        args.max_ref,
        expand_symmetry=bool(args.expand_symmetry)
    )


if __name__ == "__main__":
    main()

