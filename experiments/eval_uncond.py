#!/usr/bin/env python3
"""
Unconditional Generation Evaluation (structure-level)

Evaluates generated AtomForge programs directly from .atomforge files:
- Parse - IR - pymatgen Structure (with symmetry expansion)
- Validity: min interatomic distance, charge neutrality
- Uniqueness: StructureMatcher-based dedupe
- Novelty: fingerprints vs reference set
- Distribution matching: density & unique-element histograms + Wasserstein

Usage:
    python -m experiments.eval_uncond \
        --gen_dir outputs/uncond/programs \
        --ref_dir data \
        --out_dir outputs/uncond/eval \
        --max_gen 200 \
        --max_ref 5000 \
        --symprec 0.2
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Pymatgen
try:
    from pymatgen.core import Lattice as PmgLattice, Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    PYMATGEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYMATGEN_AVAILABLE = False

# SciPy for Wasserstein
try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False

# Matplotlib for simple plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

# AtomForge parser/IR
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
    from atomforge.src.atomforge_ir import Length, Angle
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser
    from atomforge_ir import Length, Angle

# Generative metrics (optional)
try:
    from experiments.gen_metrics import GenCrystal, compute_gen_metrics, load_gt_crystals, aggregate_results
    GEN_METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    GEN_METRICS_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval_uncond")


# -----------------------
# Helpers
# -----------------------

def get_val(x: Any, length_unit: Optional[str] = None, angle_unit: Optional[str] = None) -> float:
    """
    Extract value from Length/Angle objects with unit conversion.
    
    Args:
        x: Value to extract (Length, Angle, or numeric)
        length_unit: Unit for length conversion (from program.units.length or Length.unit)
        angle_unit: Unit for angle conversion (from program.units.angle or Angle.unit)
    
    Returns:
        Converted value in Angstroms (for Length) or degrees (for Angle)
    """
    # Unit conversion factors to Angstroms
    LENGTH_CONVERSIONS = {
        "angstrom": 1.0,
        "nm": 10.0,
        "pm": 0.01,
        "bohr": 0.529177,
        "a0": 0.529177,  # Bohr radius
    }
    
    # Unit conversion factors to degrees
    ANGLE_CONVERSIONS = {
        "degree": 1.0,
        "deg": 1.0,
        "radian": 57.29577951308232,  # 180/pi
        "rad": 57.29577951308232,
    }
    
    if isinstance(x, Length):
        value = float(x.value)
        # Check if Length object has its own unit
        unit = getattr(x, "unit", None) or length_unit
        if unit:
            unit_lower = unit.lower()
            if unit_lower in LENGTH_CONVERSIONS:
                return value * LENGTH_CONVERSIONS[unit_lower]
            else:
                logger.warning(f"Unknown length unit '{unit}', assuming Angstroms")
                return value
        else:
            # No unit specified, assume Angstroms (pymatgen default)
            return value
    elif isinstance(x, Angle):
        value = float(x.value)
        # Check if Angle object has its own unit
        unit = getattr(x, "unit", None) or angle_unit
        if unit:
            unit_lower = unit.lower()
            if unit_lower in ANGLE_CONVERSIONS:
                return value * ANGLE_CONVERSIONS[unit_lower]
            else:
                logger.warning(f"Unknown angle unit '{unit}', assuming degrees")
                return value
        else:
            # No unit specified, assume degrees (pymatgen default)
            return value
    return float(x)


def parse_program(path: Path, parser: AtomForgeParser) -> Tuple[bool, Any]:
    try:
        text = path.read_text()
        program = parser.parse_and_transform(text)
        program.validate()
        return True, program
    except Exception as e:  # pragma: no cover - logging-only
        return False, str(e)


def site_to_species_and_coords(site, lattice: PmgLattice) -> Tuple[List[Any], List[Tuple[float, float, float]]]:
    species: List[Any] = []
    coords: List[Tuple[float, float, float]] = []
    pos = site.position
    try:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    except Exception:
        raise ValueError(f"Invalid position for site {site.name}: {pos}")

    if site.frame == "cartesian":
        fx, fy, fz = lattice.get_fractional_coords([x, y, z])
    else:
        fx, fy, fz = x, y, z

    if not site.species:
        raise ValueError(f"Site {site.name} has no species")

    if len(site.species) == 1 and abs(site.species[0].occupancy - 1.0) < 1e-6:
        species.append(site.species[0].element)
        coords.append((fx, fy, fz))
    else:
        for sp in site.species:
            occ = float(sp.occupancy)
            if occ <= 0:
                continue
            species.append({sp.element: occ})
            coords.append((fx, fy, fz))
    return species, coords


def program_to_structure(program, symprec: float) -> Structure:
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required for structure conversion")

    lat = program.lattice.bravais if program.lattice else None
    if lat is None:
        raise ValueError("Missing lattice bravais parameters")
    
    # Get units from program if available
    length_unit = None
    angle_unit = None
    if program.units:
        length_unit = getattr(program.units, "length", None)
        angle_unit = getattr(program.units, "angle", None)

    lattice = PmgLattice.from_parameters(
        get_val(lat.a, length_unit=length_unit),
        get_val(lat.b, length_unit=length_unit),
        get_val(lat.c, length_unit=length_unit),
        get_val(lat.alpha, angle_unit=angle_unit),
        get_val(lat.beta, angle_unit=angle_unit),
        get_val(lat.gamma, angle_unit=angle_unit),
    )

    species_all: List[Any] = []
    coords_all: List[Tuple[float, float, float]] = []
    if not program.basis or not program.basis.sites:
        raise ValueError("Missing basis sites")
    for site in program.basis.sites:
        sps, crds = site_to_species_and_coords(site, lattice)
        species_all.extend(sps)
        coords_all.extend(crds)

    sg = program.symmetry.space_group if program.symmetry else None
    if sg is None:
        raise ValueError("Missing symmetry space_group")

    structure = Structure.from_spacegroup(
        sg,
        lattice,
        species_all,
        coords_all,
        symprec = symprec,
    )
    return structure


def structure_fingerprint(structure: Structure) -> str:
    lattice = structure.lattice
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles
    sig: Dict[str, Any] = {
        "bravais": {
            "a": round(a, 4),
            "b": round(b, 4),
            "c": round(c, 4),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "gamma": round(gamma, 4),
        },
        "sites": [],
    }
    for site in structure:
        coords = tuple(round(x, 4) for x in site.frac_coords)
        species_entries: List[Tuple[str, float]] = []
        if hasattr(site.specie, "items"):
            for elem, occ in site.specie.items():
                species_entries.append((str(elem), round(float(occ), 4)))
        else:
            species_entries.append((str(site.specie), 1.0))
        species_entries.sort(key=lambda t: (t[0], t[1]))
        sig["sites"].append({"position": coords, "species": species_entries})
    sig["sites"].sort(key=lambda s: (tuple(s["species"]), s["position"]))
    import hashlib

    canonical = json.dumps(sig, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def min_interatomic_distance(structure: Structure) -> float:
    dmat = structure.distance_matrix
    mask = np.eye(len(dmat), dtype=bool)
    masked = np.ma.masked_array(dmat, mask=mask)
    return float(np.min(masked))


def check_charge_neutrality(structure: Structure) -> Tuple[str, Optional[float]]:
    try:
        structure.add_oxidation_state_by_guess()
        total = sum(site.specie.oxi_state for site in structure)
        if abs(total) < 1e-2:
            return "neutral", 0.0
        return "charged", total
    except Exception:
        return "unknown_charge", None


def dedupe_structures(structures: List[Structure], symprec: float) -> List[int]:
    matcher = StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=False, stol=symprec)
    unique_indices: List[int] = []
    for i, s in enumerate(structures):
        matched = False
        for ui in unique_indices:
            if matcher.fit(structures[ui], s):
                matched = True
                break
        if not matched:
            unique_indices.append(i)
    return unique_indices


def load_reference_structures(ref_dir: Path, max_ref: int, symprec: float) -> Tuple[List[Structure], Set[str]]:
    parser = AtomForgeParser()
    files = sorted(ref_dir.glob("batch_*/*.atomforge"))
    if not files:
        raise ValueError(f"No reference .atomforge files found under {ref_dir}")
    if max_ref:
        files = files[:max_ref]

    ref_structs: List[Structure] = []
    ref_fps: Set[str] = set()
    for f in files:
        ok, res = parse_program(f, parser)
        if not ok:
            continue
        try:
            struct = program_to_structure(res, symprec)
            ref_structs.append(struct)
            ref_fps.add(structure_fingerprint(struct))
        except Exception:
            continue
    return ref_structs, ref_fps


def compute_distribution_metrics(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def write_report(out_dir: Path, summary: Dict[str, Any]) -> None:
    report_path = out_dir / "report.md"
    lines = []
    lines.append("# Unconditional Generation Evaluation\n")
    lines.append(f"Date: {datetime.now().isoformat()}\n")
    lines.append("## Summary\n")
    for k, v in summary.items():
        if isinstance(v, dict):
            lines.append(f"- {k}:")
            for kk, vv in v.items():
                lines.append(f"  - {kk}: {vv}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("\n## How to reproduce\n")
    lines.append("```bash")
    lines.append("python -m experiments.eval_uncond \\")
    lines.append("  --gen_dir outputs/uncond/programs \\")
    lines.append("  --ref_dir data \\")
    lines.append("  --out_dir outputs/uncond/eval \\")
    lines.append("  --max_gen 200 \\")
    lines.append("  --max_ref 5000 \\")
    lines.append("  --symprec 0.2")
    lines.append("```")
    lines.append("\n## What to do next\n- Run same evaluation on MatExpert generations (if available) or MP-20 baseline.\n- Add stability metrics (e.g., M3GNet hull) as optional module.\n")
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unconditional AtomForge generations")
    parser.add_argument("--gen_dir", type=str, default="outputs/uncond/programs", help="Directory with generated .atomforge files")
    parser.add_argument("--ref_dir", type=str, default="data", help="Directory containing reference .atomforge files")
    parser.add_argument("--out_dir", type=str, default="outputs/uncond/eval", help="Output directory for eval results")
    parser.add_argument("--max_gen", type=int, default=200, help="Max generated samples to evaluate (None=all)")
    parser.add_argument("--max_ref", type=int, default=5000, help="Max reference samples")
    parser.add_argument("--symprec", type=float, default=0.2, help="Symmetry tolerance for StructureMatcher/spacegroup")
    
    # Generative metrics flags
    parser.add_argument("--eval_suite", type=str, default="atomforge_basic", help="Evaluation suite (atomforge_basic or generative)")
    parser.add_argument("--enable_gen_metrics", action="store_true", help="Enable generative metrics suite")
    parser.add_argument("--eval_model_name", type=str, default="mp20", help="Model name for cutoffs (e.g., mp20)")
    parser.add_argument("--cov_ref_dir", type=str, default=None, help="Reference directory for coverage (defaults to --ref_dir)")
    parser.add_argument("--nov_ref_dir", type=str, default=None, help="Reference directory for novelty (defaults to --ref_dir)")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of valid samples for diversity/Wasserstein")
    parser.add_argument("--min_dist_cutoff", type=float, default=0.5, help="Minimum interatomic distance cutoff (Å)")
    parser.add_argument("--min_volume_cutoff", type=float, default=0.1, help="Minimum volume cutoff (Å³)")
    parser.add_argument("--results_csv", type=str, default="generative_model_results.csv", help="Aggregation CSV file")
    parser.add_argument("--model_name", type=str, default=None, help="Model name for aggregation CSV")
    
    args = parser.parse_args()

    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required; please install it")

    gen_path = Path(args.gen_dir)
    if not gen_path.exists():
        raise ValueError(f"Generation directory not found: {gen_path}")
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ref_path = Path(args.ref_dir)
    if not ref_path.exists():
        raise ValueError(f"Reference directory not found: {ref_path}")

    logger.info("Loading reference structures...")
    ref_structs, ref_fps = load_reference_structures(ref_path, args.max_ref, args.symprec)
    logger.info(f"Reference structures loaded: {len(ref_structs)}")

    files = sorted(gen_path.glob("*.atomforge"))
    if args.max_gen:
        files = files[: args.max_gen]
    logger.info(f"Found {len(files)} generated programs to evaluate")

    parser_af = AtomForgeParser()
    per_sample: List[Dict[str, Any]] = []
    structures: List[Structure] = []
    fingerprints: List[str] = []
    parse_ok = struct_ok = comp_ok = 0

    for f in files:
        row: Dict[str, Any] = {"id": f.stem, "file": str(f)}
        ok, res = parse_program(f, parser_af)
        if not ok:
            row.update({"parse_ok": False, "error": res})
            per_sample.append(row)
            continue
        parse_ok += 1
        row["parse_ok"] = True
        try:
            struct = program_to_structure(res, args.symprec)
            structures.append(struct)
            fp = structure_fingerprint(struct)
            fingerprints.append(fp)
            row.update({
                "struct_ok": True,
                "spacegroup": str(res.symmetry.space_group) if res.symmetry else None,
                "natoms": len(struct),
                "density": struct.density,
                "nel": len(struct.composition.elements),
                "formula": struct.composition.reduced_formula,
            })
            mind = min_interatomic_distance(struct)
            row["min_dist"] = mind
            row["valid_min_dist"] = mind > 0.5
            charge_status, net_charge = check_charge_neutrality(struct)
            row["charge_status"] = charge_status
            row["net_charge"] = net_charge
            row["comp_ok"] = charge_status == "neutral"
            struct_ok += 1
            if row["comp_ok"]:
                comp_ok += 1
        except Exception as e:
            row.update({"struct_ok": False, "error": str(e)})
        per_sample.append(row)

    unique_indices = dedupe_structures(structures, args.symprec) if structures else []
    unique_fps = {fingerprints[i] for i in unique_indices} if fingerprints else set()
    novel_flags = [fp not in ref_fps for fp in fingerprints]

    densities = [r["density"] for r in per_sample if r.get("struct_ok")]
    nels = [r["nel"] for r in per_sample if r.get("struct_ok")]

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "gen_dir": str(gen_path),
            "ref_dir": str(ref_path),
            "max_gen": args.max_gen,
            "max_ref": args.max_ref,
            "symprec": args.symprec,
        },
        "counts": {
            "total": len(files),
            "parse_ok": parse_ok,
            "struct_ok": struct_ok,
            "comp_ok": comp_ok,
        },
        "uniqueness": {
            "unique_structures": len(unique_indices),
            "unique_rate": len(unique_indices) / max(1, struct_ok),
        },
        "novelty": {
            "novel_structures": sum(novel_flags),
            "novel_rate": sum(novel_flags) / max(1, len(novel_flags)),
        },
        "validity": {
            "min_dist_pass": sum(1 for r in per_sample if r.get("valid_min_dist")),
            "min_dist_rate": sum(1 for r in per_sample if r.get("valid_min_dist")) / max(1, len(per_sample)),
            "charge_neutral": sum(1 for r in per_sample if r.get("charge_status") == "neutral"),
            "charge_neutral_rate": sum(1 for r in per_sample if r.get("charge_status") == "neutral") / max(1, len(per_sample)),
            "unknown_charge": sum(1 for r in per_sample if r.get("charge_status") == "unknown_charge"),
        },
        "distribution": {
            "density": compute_distribution_metrics(densities),
            "nel": compute_distribution_metrics(nels),
        },
    }

    if SCIPY_AVAILABLE and densities and ref_structs:
        ref_dens = [s.density for s in ref_structs]
        summary["distribution"]["wasserstein_density"] = float(wasserstein_distance(densities, ref_dens))
        ref_nels = [len(s.composition.elements) for s in ref_structs]
        if nels and ref_nels:
            summary["distribution"]["wasserstein_nel"] = float(wasserstein_distance(nels, ref_nels))

    # Generative metrics (if enabled)
    if args.enable_gen_metrics:
        if not GEN_METRICS_AVAILABLE:
            logger.error("Generative metrics requested but gen_metrics module not available")
            logger.error("Please install required dependencies: pip install smact matminer scipy")
            raise ImportError("gen_metrics module not available")
        
        logger.info("Computing generative metrics...")
        
        # Convert structures to GenCrystal objects
        # Note: structures list only contains successful conversions, so we need to match by index
        # structures[i] corresponds to per_sample[j] where j is the i-th struct_ok=True row
        struct_idx = 0
        pred_crystals = []
        
        for row in per_sample:
            if row.get("struct_ok") and struct_idx < len(structures):
                struct = structures[struct_idx]
                struct_idx += 1
                try:
                    crystal = GenCrystal(
                        struct,
                        min_dist_cutoff=args.min_dist_cutoff,
                        min_volume_cutoff=args.min_volume_cutoff
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
        cov_ref_dir = Path(args.cov_ref_dir) if args.cov_ref_dir else ref_path
        nov_ref_dir = Path(args.nov_ref_dir) if args.nov_ref_dir else ref_path
        
        cache_gt_cov_path = out_path / "cache_gt_cov.pkl"
        cache_gt_nov_path = out_path / "cache_gt_nov.pkl"
        
        gt_cov = load_gt_crystals(
            cov_ref_dir,
            args.max_ref,
            min_dist_cutoff=args.min_dist_cutoff,
            min_volume_cutoff=args.min_volume_cutoff,
            cache_path=cache_gt_cov_path
        )
        
        gt_nov = load_gt_crystals(
            nov_ref_dir,
            args.max_ref,
            min_dist_cutoff=args.min_dist_cutoff,
            min_volume_cutoff=args.min_volume_cutoff,
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
                eval_model_name=args.eval_model_name,
                n_samples=args.n_samples
            )
            summary["gen_metrics"] = gen_metrics
            logger.info("Generative metrics computed successfully")
        else:
            logger.warning("No valid crystals for generative metrics computation")
            summary["gen_metrics"] = {"error": "no_valid_crystals"}
        
        # Results aggregation
        if args.model_name:
            results_csv_path = Path(args.results_csv)
            if "gen_metrics" in summary:
                aggregate_results(results_csv_path, args.model_name, summary["gen_metrics"])
    
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))

    metrics_jsonl = out_path / "metrics.jsonl"
    with open(metrics_jsonl, "w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, default=str) + "\n")

    csv_path = out_path / "per_sample.csv"
    if per_sample:
        fieldnames = sorted({k for r in per_sample for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)

    if MATPLOTLIB_AVAILABLE:
        plots_dir = out_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        if densities:
            plt.figure(figsize=(8, 5))
            plt.hist(densities, bins=30, alpha=0.7)
            plt.xlabel("Density (g/cc)")
            plt.ylabel("Count")
            plt.title("Generated Density Distribution")
            plt.tight_layout()
            plt.savefig(plots_dir / "density.png", dpi=150)
            plt.close()
        if nels:
            plt.figure(figsize=(8, 5))
            plt.hist(nels, bins=range(1, max(nels) + 2), alpha=0.7, align="left")
            plt.xlabel("Number of unique elements")
            plt.ylabel("Count")
            plt.title("Generated Unique Elements Distribution")
            plt.tight_layout()
            plt.savefig(plots_dir / "nel.png", dpi=150)
            plt.close()

    write_report(out_path, summary)

    logger.info("\n=== EVAL SUMMARY ===")
    logger.info(f"Total: {summary['counts']['total']}")
    logger.info(f"Parse OK: {summary['counts']['parse_ok']}")
    logger.info(f"Struct OK: {summary['counts']['struct_ok']}")
    logger.info(f"Charge neutral: {summary['validity']['charge_neutral']}")
    logger.info(f"Unique structures: {summary['uniqueness']['unique_structures']}")
    logger.info(f"Novel structures: {summary['novelty']['novel_structures']}")
    logger.info(f"Summary saved to: {out_path / 'summary.json'}")


if __name__ == "__main__":
    main()

