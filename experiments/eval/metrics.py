#!/usr/bin/env python3
"""
Core Metrics Definitions

Computes evaluation metrics for both unconditional and conditional generation.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_min_interatomic_distance(structure: Structure) -> float:
    """Compute minimum interatomic distance in Angstrom."""
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen required")
    dmat = structure.distance_matrix
    mask = np.eye(len(dmat), dtype=bool)
    masked = np.ma.masked_array(dmat, mask=mask)
    return float(np.min(masked))


def check_charge_neutrality(structure: Structure) -> Tuple[str, Optional[float]]:
    """
    Check charge neutrality.
    
    Returns:
        (status, net_charge) where status is "neutral", "charged", or "unknown_charge"
    """
    if not PYMATGEN_AVAILABLE:
        return "unknown_charge", None
    
    try:
        structure.add_oxidation_state_by_guess()
        total = sum(site.specie.oxi_state for site in structure)
        if abs(total) < 0.01:
            return "neutral", 0.0
        return "charged", total
    except Exception:
        return "unknown_charge", None


def compute_uniqueness(structures: List[Structure], symprec: float = 0.2) -> Tuple[List[int], float]:
    """
    Compute uniqueness using StructureMatcher.
    
    Returns:
        (unique_indices, unique_rate)
    """
    if not PYMATGEN_AVAILABLE or not structures:
        return [], 0.0
    
    matcher = StructureMatcher(
        primitive_cell=False,
        scale=True,
        attempt_supercell=False,
        stol=symprec
    )
    
    unique_indices: List[int] = []
    for i, s in enumerate(structures):
        matched = False
        for ui in unique_indices:
            if matcher.fit(structures[ui], s):
                matched = True
                break
        if not matched:
            unique_indices.append(i)
    
    unique_rate = len(unique_indices) / len(structures) if structures else 0.0
    return unique_indices, unique_rate


def compute_novelty(fingerprints: List[str], reference_fingerprints: set) -> Tuple[int, float]:
    """
    Compute novelty (fraction of fingerprints not in reference set).
    
    Returns:
        (novel_count, novelty_rate)
    """
    if not fingerprints:
        return 0, 0.0
    
    novel_count = sum(1 for fp in fingerprints if fp not in reference_fingerprints)
    novelty_rate = novel_count / len(fingerprints)
    return novel_count, novelty_rate


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute distribution statistics (mean, std, min, max)."""
    if not values:
        return {}
    
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_wasserstein_distance(values1: List[float], values2: List[float]) -> Optional[float]:
    """Compute Wasserstein distance between two distributions."""
    if not SCIPY_AVAILABLE or not values1 or not values2:
        return None
    
    try:
        return float(wasserstein_distance(values1, values2))
    except Exception:
        return None


def check_condition_violation(structure: Structure, constraints: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if structure violates given constraints.
    
    Returns:
        (violated, violation_message)
    """
    if not PYMATGEN_AVAILABLE:
        return False, None
    
    violations = []
    
    # Composition check
    if "composition" in constraints:
        target_comp = constraints["composition"]
        if isinstance(target_comp, dict):
            struct_comp = structure.composition.as_dict()
            for elem, count in target_comp.items():
                if elem not in struct_comp or abs(struct_comp[elem] - count) > 0.01:
                    violations.append(f"Composition mismatch: expected {elem}={count}, got {struct_comp.get(elem, 0)}")
    
    # Space group check
    if "space_group" in constraints:
        target_sg = constraints["space_group"]
        # Note: Getting space group from structure requires analysis
        # For now, skip this check (would need spglib)
        pass
    
    # Element count check
    if "nel_min" in constraints or "nel_max" in constraints:
        nel = len(structure.composition.elements)
        if "nel_min" in constraints and nel < constraints["nel_min"]:
            violations.append(f"Too few elements: {nel} < {constraints['nel_min']}")
        if "nel_max" in constraints and nel > constraints["nel_max"]:
            violations.append(f"Too many elements: {nel} > {constraints['nel_max']}")
    
    # Atom count check
    if "natoms_min" in constraints or "natoms_max" in constraints:
        natoms = len(structure)
        if "natoms_min" in constraints and natoms < constraints["natoms_min"]:
            violations.append(f"Too few atoms: {natoms} < {constraints['natoms_min']}")
        if "natoms_max" in constraints and natoms > constraints["natoms_max"]:
            violations.append(f"Too many atoms: {natoms} > {constraints['natoms_max']}")
    
    # Lattice type check
    if "lattice_type" in constraints:
        # Would need to determine lattice type from structure
        pass
    
    # Density range check
    if "density_range" in constraints:
        density = structure.density
        dr = constraints["density_range"]
        if "min" in dr and density < dr["min"]:
            violations.append(f"Density too low: {density:.2f} < {dr['min']}")
        if "max" in dr and density > dr["max"]:
            violations.append(f"Density too high: {density:.2f} > {dr['max']}")
    
    if violations:
        return True, "; ".join(violations)
    return False, None

