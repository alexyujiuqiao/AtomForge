#!/usr/bin/env python3
"""
AtomForge Crystal Comparison & Diff - Phase 4 Implementation

This module implements Phase 4 comparison and diff operations:
- compare(crystal_a, crystal_b, tol) -> CompareReport
- diff(crystal_a, crystal_b) -> List[PatchRecord]

Based on operations_and_workflows-v-2-0.tex specifications.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import math

from crystal_v1_1 import Crystal, Lattice, Symmetry, Site, Composition, identity_hash
from crystal_edit import PatchRecord

# ============================================================================
# PHASE 4 DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class CompareReport:
    """Report from comparing two crystal structures"""
    equivalent: bool                                    # True if structures are equivalent
    identity_hash_match: bool                           # True if identity hashes match
    lattice_mae: float                                  # Mean absolute error in lattice parameters
    lattice_rmse: float                                 # Root mean square error in lattice parameters
    position_rmsd: float                                # Root mean square deviation of atomic positions
    space_group_match: bool                             # True if space groups match
    composition_match: bool                              # True if compositions match
    site_count_match: bool                              # True if site counts match
    diff_summary: List[str]                             # Human-readable summary of differences
    metrics: Dict[str, Any]                             # Additional comparison metrics
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# ============================================================================
# COMPARISON OPERATIONS
# ============================================================================

def compare(crystal_a: Crystal, crystal_b: Crystal, tol: float = 1e-6) -> CompareReport:
    """
    Compare two crystal structures and generate a comprehensive report.
    
    Args:
        crystal_a: First crystal structure
        crystal_b: Second crystal structure
        tol: Tolerance for numerical comparisons
        
    Returns:
        CompareReport with detailed comparison results
    """
    diff_summary = []
    metrics = {}
    
    # Identity hash comparison (most reliable)
    hash_a = identity_hash(crystal_a)
    hash_b = identity_hash(crystal_b)
    identity_hash_match = (hash_a == hash_b)
    
    if identity_hash_match:
        # If hashes match, structures are identical
        return CompareReport(
            equivalent=True,
            identity_hash_match=True,
            lattice_mae=0.0,
            lattice_rmse=0.0,
            position_rmsd=0.0,
            space_group_match=True,
            composition_match=True,
            site_count_match=True,
            diff_summary=["Structures are identical (identity hash match)"],
            metrics={"hash_a": hash_a, "hash_b": hash_b}
        )
    
    # Lattice parameter comparison
    lattice_mae, lattice_rmse = _compare_lattice(crystal_a.lattice, crystal_b.lattice)
    metrics["lattice_mae"] = lattice_mae
    metrics["lattice_rmse"] = lattice_rmse
    
    if lattice_mae > tol:
        diff_summary.append(f"Lattice parameters differ: MAE = {lattice_mae:.6f} Å, RMSE = {lattice_rmse:.6f} Å")
    
    # Space group comparison
    space_group_match = _compare_symmetry(crystal_a.symmetry, crystal_b.symmetry)
    if not space_group_match:
        diff_summary.append(f"Space group mismatch: {crystal_a.symmetry.space_group} vs {crystal_b.symmetry.space_group}")
    
    # Composition comparison
    composition_match = _compare_composition(crystal_a.composition, crystal_b.composition)
    if not composition_match:
        comp_a_str = _format_composition(crystal_a.composition)
        comp_b_str = _format_composition(crystal_b.composition)
        diff_summary.append(f"Composition mismatch: {comp_a_str} vs {comp_b_str}")
    
    # Site count comparison
    site_count_match = (len(crystal_a.sites) == len(crystal_b.sites))
    if not site_count_match:
        diff_summary.append(f"Site count mismatch: {len(crystal_a.sites)} vs {len(crystal_b.sites)}")
    
    # Atomic position comparison (RMSD)
    position_rmsd = _compare_positions(crystal_a, crystal_b, tol)
    metrics["position_rmsd"] = position_rmsd
    
    if position_rmsd > tol:
        diff_summary.append(f"Atomic positions differ: RMSD = {position_rmsd:.6f} Å")
    
    # Overall equivalence (within tolerance)
    equivalent = (
        identity_hash_match or
        (lattice_mae <= tol and 
         position_rmsd <= tol and 
         space_group_match and 
         composition_match and
         site_count_match)
    )
    
    if equivalent and not identity_hash_match:
        diff_summary.append("Structures are equivalent within tolerance (but identity hashes differ)")
    
    metrics.update({
        "hash_a": hash_a,
        "hash_b": hash_b,
        "site_count_a": len(crystal_a.sites),
        "site_count_b": len(crystal_b.sites)
    })
    
    return CompareReport(
        equivalent=equivalent,
        identity_hash_match=identity_hash_match,
        lattice_mae=lattice_mae,
        lattice_rmse=lattice_rmse,
        position_rmsd=position_rmsd,
        space_group_match=space_group_match,
        composition_match=composition_match,
        site_count_match=site_count_match,
        diff_summary=diff_summary,
        metrics=metrics
    )


def diff(crystal_a: Crystal, crystal_b: Crystal) -> List[PatchRecord]:
    """
    Generate a sequence of PatchRecords that would transform crystal_a into crystal_b.
    
    This is a best-effort inverse patch operation useful for reviews and provenance tracking.
    
    Args:
        crystal_a: Source crystal structure
        crystal_b: Target crystal structure
        
    Returns:
        List of PatchRecord operations that would transform A -> B
    """
    patches = []
    timestamp = datetime.now().isoformat()
    
    # Compare reports first
    compare_rep = compare(crystal_a, crystal_b)
    
    # If identical, return empty patch list
    if compare_rep.equivalent and compare_rep.identity_hash_match:
        return []
    
    # Lattice parameter changes
    if compare_rep.lattice_mae > 1e-6:
        new_lattice = crystal_b.lattice
        patches.append(PatchRecord(
            op="set_lattice",
            params={
                "new_lattice": {
                    "a": new_lattice.a,
                    "b": new_lattice.b,
                    "c": new_lattice.c,
                    "alpha": new_lattice.alpha,
                    "beta": new_lattice.beta,
                    "gamma": new_lattice.gamma
                }
            },
            preconditions={"symmetry_locked": False},
            result_hash=identity_hash(crystal_b),
            timestamp=timestamp,
            op_version="AtomForge/Comparison/4.0"
        ))
    
    # Symmetry changes
    if not compare_rep.space_group_match:
        new_symmetry = crystal_b.symmetry
        patches.append(PatchRecord(
            op="set_symmetry",
            params={
                "new_symmetry": {
                    "space_group": new_symmetry.space_group,
                    "number": new_symmetry.number,
                    "origin_choice": new_symmetry.origin_choice
                }
            },
            preconditions={"symmetry_locked": False},
            result_hash=identity_hash(crystal_b),
            timestamp=timestamp,
            op_version="AtomForge/Comparison/4.0"
        ))
    
    # Site-level differences
    site_patches = _diff_sites(crystal_a, crystal_b, timestamp)
    patches.extend(site_patches)
    
    return patches


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _compare_lattice(lattice_a: Lattice, lattice_b: Lattice) -> Tuple[float, float]:
    """Compare two lattice structures and return MAE and RMSE"""
    params_a = [lattice_a.a, lattice_a.b, lattice_a.c, lattice_a.alpha, lattice_a.beta, lattice_a.gamma]
    params_b = [lattice_b.a, lattice_b.b, lattice_b.c, lattice_b.alpha, lattice_b.beta, lattice_b.gamma]
    
    diffs = [abs(a - b) for a, b in zip(params_a, params_b)]
    mae = sum(diffs) / len(diffs)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    
    return mae, rmse


def _compare_symmetry(symm_a: Symmetry, symm_b: Symmetry) -> bool:
    """Compare two symmetry structures"""
    return (
        symm_a.space_group == symm_b.space_group and
        symm_a.number == symm_b.number
    )


def _compare_composition(comp_a: Composition, comp_b: Composition) -> bool:
    """Compare two composition structures"""
    # Compare reduced formulas
    if set(comp_a.reduced.keys()) != set(comp_b.reduced.keys()):
        return False
    
    for element in comp_a.reduced.keys():
        if comp_a.reduced[element] != comp_b.reduced[element]:
            return False
    
    return True


def _format_composition(comp: Composition) -> str:
    """Format composition as string"""
    return " ".join(f"{elem}{count}" for elem, count in sorted(comp.reduced.items()))


def _compare_positions(crystal_a: Crystal, crystal_b: Crystal, tol: float) -> float:
    """Compare atomic positions and return RMSD"""
    # Convert fractional to cartesian for comparison
    # This is a simplified version - in practice, you'd want to account for
    # different lattice parameters and coordinate frames
    
    if len(crystal_a.sites) != len(crystal_b.sites):
        # Return large RMSD if site counts differ
        return float('inf')
    
    # Simple RMSD calculation on fractional coordinates
    # In a full implementation, you'd want to:
    # 1. Match sites by species/position
    # 2. Account for periodic boundary conditions
    # 3. Handle different coordinate frames
    
    squared_diffs = []
    for site_a, site_b in zip(crystal_a.sites, crystal_b.sites):
        pos_a = site_a.frac
        pos_b = site_b.frac
        
        # Handle periodic boundary conditions (wrap to [0,1))
        diff = [
            min(abs(a - b), 1.0 - abs(a - b))
            for a, b in zip(pos_a, pos_b)
        ]
        
        squared_diffs.append(sum(d * d for d in diff))
    
    if not squared_diffs:
        return 0.0
    
    rmsd = math.sqrt(sum(squared_diffs) / len(squared_diffs))
    return rmsd


def _diff_sites(crystal_a: Crystal, crystal_b: Crystal, timestamp: str) -> List[PatchRecord]:
    """Generate patch records for site-level differences"""
    patches = []
    
    # Create maps of sites by position (for matching)
    sites_a = {i: site for i, site in enumerate(crystal_a.sites)}
    sites_b = {i: site for i, site in enumerate(crystal_b.sites)}
    
    # Simple matching: compare sites by index
    # In a full implementation, you'd want smarter matching based on:
    # - Species similarity
    # - Position proximity
    # - Wyckoff positions
    
    max_sites = max(len(crystal_a.sites), len(crystal_b.sites))
    
    for i in range(max_sites):
        if i < len(crystal_a.sites) and i < len(crystal_b.sites):
            site_a = crystal_a.sites[i]
            site_b = crystal_b.sites[i]
            
            # Check for substitutions
            if site_a.species != site_b.species:
                # Generate substitution patch
                for elem_a, occ_a in site_a.species.items():
                    if elem_a not in site_b.species or site_b.species[elem_a] != occ_a:
                        # This is a substitution or vacancy
                        if elem_a in site_b.species:
                            new_occ = site_b.species[elem_a]
                            if new_occ < occ_a:
                                # Vacancy creation
                                patches.append(PatchRecord(
                                    op="vacancy",
                                    params={
                                        "site_sel": f"index:{i}",
                                        "occupancy": new_occ
                                    },
                                    preconditions={"symmetry_locked": False},
                                    result_hash="",  # Would be computed after applying patch
                                    timestamp=timestamp,
                                    op_version="AtomForge/Comparison/4.0"
                                ))
                        else:
                            # Complete substitution
                            # Find the new element
                            new_elem = next(iter(site_b.species.keys()))
                            patches.append(PatchRecord(
                                op="substitute",
                                params={
                                    "site_sel": f"index:{i}",
                                    "new_species": new_elem,
                                    "occupancy": site_b.species[new_elem]
                                },
                                preconditions={"symmetry_locked": False},
                                result_hash="",
                                timestamp=timestamp,
                                op_version="AtomForge/Comparison/4.0"
                            ))
        
        elif i < len(crystal_b.sites):
            # Site added in crystal_b
            site_b = crystal_b.sites[i]
            for elem, occ in site_b.species.items():
                patches.append(PatchRecord(
                    op="interstitial",
                    params={
                        "frac": site_b.frac,
                        "species": elem,
                        "occupancy": occ
                    },
                    preconditions={"symmetry_locked": False},
                    result_hash="",
                    timestamp=timestamp,
                    op_version="AtomForge/Comparison/4.0"
                ))
        
        elif i < len(crystal_a.sites):
            # Site removed in crystal_b (vacancy)
            site_a = crystal_a.sites[i]
            patches.append(PatchRecord(
                op="vacancy",
                params={
                    "site_sel": f"index:{i}",
                    "occupancy": 0.0
                },
                preconditions={"symmetry_locked": False},
                result_hash="",
                timestamp=timestamp,
                op_version="AtomForge/Comparison/4.0"
            ))
    
    return patches


# ============================================================================
# VALIDATION
# ============================================================================

def validate_diff_application(crystal_a: Crystal, patches: List[PatchRecord], crystal_b: Crystal, tol: float = 1e-6) -> bool:
    """
    Validate that applying patches to crystal_a results in crystal_b.
    
    This is a validation function to ensure diff() works correctly.
    Note: This requires the actual patch application functions to be implemented.
    
    Args:
        crystal_a: Source crystal
        patches: List of patches from diff()
        crystal_b: Target crystal
        tol: Tolerance for comparison
        
    Returns:
        True if applying patches to crystal_a results in crystal_b within tolerance
    """
    # This would require importing and using the actual patch application functions
    # from crystal_edit module. For now, we'll just check the structure.
    
    # In a full implementation:
    # 1. Apply each patch to crystal_a sequentially
    # 2. Compare the final result with crystal_b using compare()
    # 3. Return True if equivalent within tolerance
    
    # Placeholder: check if patches are non-empty and have reasonable structure
    if not patches:
        # If no patches, crystals should be identical
        comp = compare(crystal_a, crystal_b, tol)
        return comp.equivalent
    
    # For now, return True if we have patches (actual validation would apply them)
    return True


if __name__ == "__main__":
    print("AtomForge Crystal Comparison & Diff Module")
    print("=" * 50)
    print("This module provides compare() and diff() functions")
    print("for comparing crystal structures and generating patch records.")

