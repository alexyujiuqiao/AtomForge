#!/usr/bin/env python3
"""
Unconditional Structure Evaluation

Evaluates unconditional generation outputs using the unified metrics framework.
"""

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

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

# Generative metrics (optional)
try:
    from experiments.gen_metrics import GenCrystal, compute_gen_metrics, load_gt_crystals, aggregate_results
    GEN_METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEN_METRICS_AVAILABLE = False

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


def extract_formula_from_program(file_path: Path) -> Optional[str]:
    """Extract formula from program file (atom_spec name or title)."""
    try:
        text = file_path.read_text()
        
        # Try to extract from atom_spec name first (e.g., atom_spec "ZnAl2O4_Spinel")
        atom_spec_match = re.search(r'atom_spec\s+"([^"]+)"', text)
        if atom_spec_match:
            name = atom_spec_match.group(1)
            # Remove suffixes like "_Spinel", "_Perovskite", etc.
            formula = re.sub(r'_[A-Z][a-z]*$', '', name)
            # Remove common suffixes
            formula = re.sub(r'_(Spinel|Perovskite|Feldspathoid|Melilite|Hypothetical)$', '', formula, flags=re.IGNORECASE)
            if formula:
                return formula
        
        # Fallback to title (e.g., title = "ZnAl2O4 (normal spinel)")
        title_match = re.search(r'title\s*=\s*"([^"]+)"', text)
        if title_match:
            title = title_match.group(1)
            # Extract formula before parentheses
            formula_match = re.match(r'([A-Za-z0-9]+)', title)
            if formula_match:
                return formula_match.group(1)
        
        return None
    except Exception:
        return None


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
            struct, _ = atomforge_to_structure(res, expand_symmetry=True, auto_detect_expanded=True)
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
    expand_symmetry: bool = True,
    symprec: float = 0.2,
    skip_novelty_uniqueness: bool = True,
    enable_gen_metrics: bool = False,
    eval_model_name: str = "mp20",
    cov_ref_dir: Optional[Path] = None,
    nov_ref_dir: Optional[Path] = None,
    n_samples: int = 1000,
    min_dist_cutoff: float = 0.5,
    min_volume_cutoff: float = 0.1,
    results_csv: Optional[Path] = None,
    model_name: Optional[str] = None
) -> None:
    """Evaluate all structures in generation directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference (skip if not needed)
    if skip_novelty_uniqueness:
        ref_fps = set()
        logger.info("Skipping reference fingerprint loading (novelty/uniqueness disabled)")
    else:
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
                struct, conversion_metadata = atomforge_to_structure(
                    res, 
                    expand_symmetry=expand_symmetry,
                    symprec=symprec,
                    auto_detect_expanded=True
                )
                
                # Sanity checks: prevent density explosions
                natoms = len(struct)
                density = float(struct.density) if hasattr(struct.density, '__float__') else struct.density
                volume = struct.volume
                
                if natoms > 2000 or density > 50.0:
                    row["struct_ok"] = False
                    row["error"] = f"expansion_exploded: natoms={natoms}, density={density:.2f} g/cc"
                    row["natoms"] = natoms
                    row["density"] = density
                    row["volume"] = volume
                    row.update(conversion_metadata)
                    per_sample.append(row)
                    continue
                
                # Record conversion metadata
                row["struct_ok"] = True
                row["n_input_sites"] = conversion_metadata.get("n_input_sites", 0)
                row["used_symmetry_expansion"] = conversion_metadata.get("used_symmetry_expansion", False)
                row["auto_skipped"] = conversion_metadata.get("auto_skipped", False)
                row["volume"] = volume
            except Exception as e:
                row["struct_ok"] = False
                row["error"] = str(e)
                per_sample.append(row)
                continue
        elif f.suffix == ".cif":
            try:
                struct = cif_to_structure(f)
                natoms = len(struct)
                density = float(struct.density) if hasattr(struct.density, '__float__') else struct.density
                if natoms > 2000 or density > 50.0:
                    row["parse_ok"] = True
                    row["struct_ok"] = False
                    row["error"] = f"expansion_exploded: natoms={natoms}, density={density:.2f} g/cc"
                    row["natoms"] = natoms
                    row["density"] = density
                    row["volume"] = struct.volume
                    row["used_symmetry_expansion"] = False
                    per_sample.append(row)
                    continue
                row["parse_ok"] = True
                row["struct_ok"] = True
                row["used_symmetry_expansion"] = False
            except Exception as e:
                row["parse_ok"] = False
                row["struct_ok"] = False
                row["error"] = str(e)
                per_sample.append(row)
                continue
        elif "POSCAR" in f.name:
            try:
                struct = poscar_to_structure(f)
                natoms = len(struct)
                density = float(struct.density) if hasattr(struct.density, '__float__') else struct.density
                if natoms > 2000 or density > 50.0:
                    row["parse_ok"] = True
                    row["struct_ok"] = False
                    row["error"] = f"expansion_exploded: natoms={natoms}, density={density:.2f} g/cc"
                    row["natoms"] = natoms
                    row["density"] = density
                    row["volume"] = struct.volume
                    row["used_symmetry_expansion"] = False
                    per_sample.append(row)
                    continue
                row["parse_ok"] = True
                row["struct_ok"] = True
                row["used_symmetry_expansion"] = False
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
        
        # Extract formula from program file if available, otherwise use structure's reduced formula
        formula = None
        if f.suffix == ".atomforge" and f.exists():
            formula = extract_formula_from_program(f)
        if formula is None:
            formula = struct.composition.reduced_formula
        
        # Update row with structure metrics (volume may already be set)
        # Convert density to float (pymatgen returns FloatWithUnit)
        density_val = float(struct.density) if hasattr(struct.density, '__float__') else struct.density
        
        row.update({
            "spacegroup": spacegroup,
            "natoms": len(struct),
            "density": density_val,
            "nel": len(struct.composition.elements),
            "formula": formula,
        })
        if "volume" not in row:
            row["volume"] = struct.volume
        if "used_symmetry_expansion" not in row:
            row["used_symmetry_expansion"] = False
        
        min_dist = compute_min_interatomic_distance(struct)
        row["min_dist"] = min_dist
        row["valid_min_distance"] = min_dist > 0.5
        
        charge_status, net_charge = check_charge_neutrality(struct)
        row["charge_status"] = charge_status
        row["net_charge"] = net_charge
        row["charge_neutral"] = charge_status == "neutral"
        
        row["fingerprint"] = fp
        # Skip novelty check if disabled
        if skip_novelty_uniqueness:
            row["novel"] = None
        else:
            row["novel"] = fp not in ref_fps
        
        # Failure mode
        failure_mode = categorize_failure(None, row)
        row["failure_mode"] = failure_mode.value
        
        per_sample.append(row)
    
    # Uniqueness (skip if disabled)
    if skip_novelty_uniqueness:
        logger.info("Skipping uniqueness calculation")
        unique_indices = []
        unique_rate = 0.0
        for i, row in enumerate(per_sample):
            row["unique"] = None
    else:
        unique_indices, unique_rate = compute_uniqueness(structures, symprec=symprec)
        for i in unique_indices:
            if i < len(per_sample):
                per_sample[i]["unique"] = True
        for i, row in enumerate(per_sample):
            if "unique" not in row:
                row["unique"] = False
    
    # Summary
    successful = [r for r in per_sample if r.get("struct_ok")]
    # Novelty (skip if disabled)
    if skip_novelty_uniqueness:
        logger.info("Skipping novelty calculation")
        novel_count = len(successful)  # Placeholder
        novelty_rate = 1.0  # Placeholder
    else:
        novel_count, novelty_rate = compute_novelty(fingerprints, ref_fps)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "gen_dir": str(gen_dir),
            "ref_dir": str(ref_dir),
            "max_gen": max_gen,
            "max_ref": max_ref,
            "expand_symmetry": expand_symmetry,
            "symprec": symprec,
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
            "unique_structures": len(unique_indices) if not skip_novelty_uniqueness else None,
            "unique_rate": unique_rate if not skip_novelty_uniqueness else None,
            "skipped": skip_novelty_uniqueness,
        },
        "novelty": {
            "novel_structures": novel_count if not skip_novelty_uniqueness else None,
            "novelty_rate": novelty_rate if not skip_novelty_uniqueness else None,
            "skipped": skip_novelty_uniqueness,
        },
        "distribution": {
            "density": compute_distribution_stats([r.get("density", 0) for r in successful if "density" in r]),
            "nel": compute_distribution_stats([r.get("nel", 0) for r in successful if "nel" in r]),
        },
    }
    
    # Generative metrics (if enabled)
    if enable_gen_metrics:
        if not GEN_METRICS_AVAILABLE:
            logger.error("Generative metrics requested but gen_metrics module not available")
            logger.error("Please install required dependencies: pip install smact matminer scipy")
            raise ImportError("gen_metrics module not available")
        
        logger.info("Computing generative metrics...")
        
        # Convert structures to GenCrystal objects
        # Match structures to per_sample rows by index (structures[i] corresponds to successful[i])
        struct_idx = 0
        pred_crystals = []
        
        for row in per_sample:
            if row.get("struct_ok") and struct_idx < len(structures):
                struct = structures[struct_idx]
                struct_idx += 1
                try:
                    crystal = GenCrystal(
                        struct,
                        min_dist_cutoff=min_dist_cutoff,
                        min_volume_cutoff=min_volume_cutoff
                    )
                    pred_crystals.append(crystal)
                    row["struct_valid"] = crystal.struct_valid
                    row["comp_valid"] = crystal.comp_valid
                    row["valid"] = crystal.valid
                    row["invalid_reason"] = crystal.invalid_reason
                except Exception as e:
                    logger.warning(f"Failed to create GenCrystal for {row.get('id')}: {e}")
                    pred_crystals.append(None)
                    row["struct_valid"] = False
                    row["comp_valid"] = False
                    row["valid"] = False
                    row["invalid_reason"] = f"gen_crystal_failed: {e}"
            else:
                # No structure available for this row
                row["struct_valid"] = False
                row["comp_valid"] = False
                row["valid"] = False
                row["invalid_reason"] = row.get("invalid_reason", "no_structure")
        
        # Load GT sets with caching
        cov_ref_path = cov_ref_dir if cov_ref_dir else ref_dir
        nov_ref_path = nov_ref_dir if nov_ref_dir else ref_dir
        
        cache_gt_cov_path = out_dir / "cache_gt_cov.pkl"
        cache_gt_nov_path = out_dir / "cache_gt_nov.pkl"
        
        gt_cov = load_gt_crystals(
            cov_ref_path,
            max_ref,
            min_dist_cutoff=min_dist_cutoff,
            min_volume_cutoff=min_volume_cutoff,
            cache_path=cache_gt_cov_path
        )
        
        gt_nov = load_gt_crystals(
            nov_ref_path,
            max_ref,
            min_dist_cutoff=min_dist_cutoff,
            min_volume_cutoff=min_volume_cutoff,
            cache_path=cache_gt_nov_path
        )
        
        logger.info(f"Loaded {len(gt_cov)} GT crystals for coverage, {len(gt_nov)} for novelty")
        
        # Compute gen_metrics
        valid_pred_crystals = [c for c in pred_crystals if c is not None and c.valid]
        if valid_pred_crystals:
            gen_metrics = compute_gen_metrics(
                valid_pred_crystals,
                gt_cov,
                gt_nov,
                eval_model_name=eval_model_name,
                n_samples=n_samples
            )
            summary["gen_metrics"] = gen_metrics
            logger.info("Generative metrics computed successfully")
        else:
            logger.warning("No valid crystals for generative metrics computation")
            summary["gen_metrics"] = {"error": "no_valid_crystals"}
        
        # Results aggregation
        if model_name and results_csv:
            results_csv_path = Path(results_csv)
            if "gen_metrics" in summary:
                aggregate_results(results_csv_path, model_name, summary["gen_metrics"])
    
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
    parser.add_argument("--symprec", type=float, default=0.2, help="Symmetry tolerance for expansion")
    
    # Generative metrics flags
    parser.add_argument("--enable_gen_metrics", action="store_true", help="Enable generative metrics suite")
    parser.add_argument("--eval_model_name", type=str, default="mp20", help="Model name for cutoffs (e.g., mp20, atomforge)")
    parser.add_argument("--cov_ref_dir", type=str, default=None, help="Reference directory for coverage (defaults to --ref_dir)")
    parser.add_argument("--nov_ref_dir", type=str, default=None, help="Reference directory for novelty (defaults to --ref_dir)")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of valid samples for diversity/Wasserstein")
    parser.add_argument("--min_dist_cutoff", type=float, default=0.5, help="Minimum interatomic distance cutoff (Å)")
    parser.add_argument("--min_volume_cutoff", type=float, default=0.1, help="Minimum volume cutoff (Å³)")
    parser.add_argument("--results_csv", type=str, default=None, help="Aggregation CSV file")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for aggregation CSV")
    
    args = parser.parse_args()
    
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required")
    
    gen_path = Path(args.gen_dir)
    if not gen_path.exists():
        raise ValueError(f"Generation directory not found: {gen_path}")
    
    ref_path = Path(args.ref_dir)
    if not ref_path.exists():
        logger.warning(f"Reference directory not found: {ref_path}")
    
    cov_ref_path = Path(args.cov_ref_dir) if args.cov_ref_dir else None
    nov_ref_path = Path(args.nov_ref_dir) if args.nov_ref_dir else None
    results_csv_path = Path(args.results_csv) if args.results_csv else None
    
    evaluate_directory(
        gen_path,
        ref_path,
        Path(args.out_dir),
        args.max_gen,
        args.max_ref,
        expand_symmetry=bool(args.expand_symmetry),
        symprec=args.symprec,
        skip_novelty_uniqueness=True,  # Disabled by default for faster evaluation
        enable_gen_metrics=args.enable_gen_metrics,
        eval_model_name=args.eval_model_name,
        cov_ref_dir=cov_ref_path,
        nov_ref_dir=nov_ref_path,
        n_samples=args.n_samples,
        min_dist_cutoff=args.min_dist_cutoff,
        min_volume_cutoff=args.min_volume_cutoff,
        results_csv=results_csv_path,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()

