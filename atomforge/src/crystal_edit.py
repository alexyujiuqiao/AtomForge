#!/usr/bin/env python3
"""
AtomForge Crystal Editing v2.0 - Phase 2 Implementation

This module implements Phase 2 of the AtomForge Crystal MVP Plan:
- Editing & Patching operations (substitute, vacancy, interstitial, set_lattice, set_symmetry)
- Supercell operations (make_supercell with child-parent mapping)
- Export operations (to_poscar, to_cif, pymatgen adapters)
- PatchRecord system for tracking all modifications
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Literal
from datetime import datetime
import hashlib
import json
import numpy as np

# PyMatgen imports for structure manipulation
from pymatgen.core import Structure, Lattice as PMGLattice
from pymatgen.core.sites import PeriodicSite

# Import Phase 0 structures
from crystal_v1_1 import (
    Crystal, Lattice, Symmetry, Site, Composition, Provenance, ConstraintSet,
    Vec3, Mat3, identity_hash
)

# ============================================================================
# PHASE 2 DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class PatchRecord:
    """Record of a single editing operation for provenance tracking"""
    op: str                                    # operation name (e.g., "substitute", "vacancy")
    params: Dict[str, Any]                     # operation parameters
    preconditions: Dict[str, Any]              # preconditions that were checked
    result_hash: str                           # hash of the result crystal
    timestamp: str                             # ISO 8601 timestamp
    op_version: str = "AtomForge/Editing/2.0"  # operation version for compatibility

@dataclass(frozen=True)
class SupercellMap:
    """Mapping from child supercell sites to parent sites and lattice vectors"""
    child_to_parent: Dict[int, Tuple[int, Tuple[int, int, int]]]  # child_site_index -> (parent_site_index, lattice_vector)
    parent_to_children: Dict[int, List[Tuple[int, Tuple[int, int, int]]]]  # parent_site_index -> [(child_site_index, lattice_vector), ...]
    transformation_matrix: Mat3                 # 3x3 transformation matrix M
    child_lattice: Lattice                     # resulting supercell lattice
    parent_lattice: Lattice                    # original unit cell lattice

@dataclass
class EditingReport:
    """Report from editing operations"""
    success: bool
    warnings: List[str]
    errors: List[str]
    patch_record: Optional[PatchRecord] = None
    details: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# PYMATGEN INTEGRATION UTILITIES
# ============================================================================

class PyMatgenAdapter:
    """Adapter for converting between AtomForge Crystal and PyMatgen Structure"""
    
    @staticmethod
    def crystal_to_pymatgen(crystal: Crystal) -> Structure:
        """Convert AtomForge Crystal to PyMatgen Structure"""
        # Convert lattice
        lattice_matrix = np.array([
            [crystal.lattice.a, 0, 0],
            [crystal.lattice.b * np.cos(np.radians(crystal.lattice.gamma)), 
             crystal.lattice.b * np.sin(np.radians(crystal.lattice.gamma)), 0],
            [crystal.lattice.c * np.cos(np.radians(crystal.lattice.beta)),
             crystal.lattice.c * (np.cos(np.radians(crystal.lattice.alpha)) - 
                                 np.cos(np.radians(crystal.lattice.beta)) * 
                                 np.cos(np.radians(crystal.lattice.gamma))) / 
                                 np.sin(np.radians(crystal.lattice.gamma)),
             crystal.lattice.c * np.sqrt(1 - np.cos(np.radians(crystal.lattice.alpha))**2 - 
                                        np.cos(np.radians(crystal.lattice.beta))**2 - 
                                        np.cos(np.radians(crystal.lattice.gamma))**2 + 
                                        2 * np.cos(np.radians(crystal.lattice.alpha)) * 
                                        np.cos(np.radians(crystal.lattice.beta)) * 
                                        np.cos(np.radians(crystal.lattice.gamma)))]
        ])
        
        pymatgen_lattice = PMGLattice(lattice_matrix)
        
        # Convert sites
        species = []
        coords = []
        
        for site in crystal.sites:
            # For multi-species sites, create multiple entries
            for species_name, occupancy in site.species.items():
                if occupancy > 0:  # Only include non-zero occupancies
                    species.append(species_name)
                    coords.append(list(site.frac))
        
        return Structure(pymatgen_lattice, species, coords)
    
    @staticmethod
    def pymatgen_to_crystal(structure: Structure, original_crystal: Crystal) -> Crystal:
        """Convert PyMatgen Structure back to AtomForge Crystal"""
        # Convert lattice back
        lattice_matrix = structure.lattice.matrix
        a = np.linalg.norm(lattice_matrix[0])
        b = np.linalg.norm(lattice_matrix[1])
        c = np.linalg.norm(lattice_matrix[2])
        
        alpha = np.degrees(np.arccos(np.dot(lattice_matrix[1], lattice_matrix[2]) / (b * c)))
        beta = np.degrees(np.arccos(np.dot(lattice_matrix[0], lattice_matrix[2]) / (a * c)))
        gamma = np.degrees(np.arccos(np.dot(lattice_matrix[0], lattice_matrix[1]) / (a * b)))
        
        new_lattice = Lattice(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        
        # Convert sites back
        sites = []
        for i, site in enumerate(structure.sites):
            # Create a single-species site
            species_dict = {site.specie.symbol: 1.0}
            
            new_site = Site(
                species=species_dict,
                frac=(site.frac_coords[0], site.frac_coords[1], site.frac_coords[2]),
                wyckoff=None,  # Will be inferred during canonicalization
                multiplicity=None,
                label=None,
                magnetic_moment=None,
                charge=None,
                disorder_group=None
            )
            sites.append(new_site)
        
        # Update composition
        composition = PyMatgenAdapter._calculate_composition_from_sites(sites)
        
        return Crystal(
            lattice=new_lattice,
            symmetry=original_crystal.symmetry,
            sites=tuple(sites),
            composition=composition,
            oxidation_states=original_crystal.oxidation_states,
            constraints=original_crystal.constraints,
            provenance=original_crystal.provenance,
            notes=original_crystal.notes
        )
    
    @staticmethod
    def get_species_symbol(species: str, crystal: Crystal) -> str:
        """
        Get the appropriate species symbol for export, handling oxidation states flexibly.
        
        Args:
            species: Species name (e.g., "Li", "O", "Li+")
            crystal: Crystal object containing oxidation state information
            
        Returns:
            Species symbol with appropriate oxidation state notation
        """
        # If species already has oxidation state, use as-is
        if '+' in species or '-' in species:
            return species
        
        # Extract element name (remove any existing oxidation state notation)
        element = species.split('+')[0].split('-')[0]
        
        # Check if oxidation state is available in crystal data
        if crystal.oxidation_states and element in crystal.oxidation_states:
            ox_state = crystal.oxidation_states[element]
            if ox_state > 0:
                return f"{element}{ox_state}+"
            elif ox_state < 0:
                return f"{element}{abs(ox_state)}-"
            else:
                return element
        else:
            # No oxidation state information available, use element as-is
            return element
    
    @staticmethod
    def _calculate_composition_from_sites(sites: List[Site]) -> Composition:
        species_counts = {}
        for site in sites:
            for species, occupancy in site.species.items():
                if species in species_counts:
                    species_counts[species] += occupancy
                else:
                    species_counts[species] = occupancy
        
        total_atoms = sum(species_counts.values())
        atomic_fractions = {k: v/total_atoms for k, v in species_counts.items()}
        
        return Composition(reduced=species_counts, atomic_fractions=atomic_fractions)

# ============================================================================
# SITE SELECTION UTILITIES
# ============================================================================

class SiteSelector:
    """Utility class for selecting sites in crystal structures"""
    
    @staticmethod
    def select_by_index(crystal: Crystal, indices: Union[int, List[int]]) -> List[int]:
        """Select sites by index"""
        if isinstance(indices, int):
            indices = [indices]
        
        # Validate indices
        max_index = len(crystal.sites) - 1
        for idx in indices:
            if idx < 0 or idx > max_index:
                raise ValueError(f"Site index {idx} out of range [0, {max_index}]")
        
        return indices
    
    @staticmethod
    def select_by_wyckoff(crystal: Crystal, wyckoff: str) -> List[int]:
        """Select sites by Wyckoff position (e.g., '4a', '2b')"""
        indices = []
        for i, site in enumerate(crystal.sites):
            if site.wyckoff == wyckoff:
                indices.append(i)
        
        if not indices:
            raise ValueError(f"No sites found with Wyckoff position '{wyckoff}'")
        
        return indices
    
    @staticmethod
    def select_by_species(crystal: Crystal, species: str) -> List[int]:
        """Select sites containing a specific species"""
        indices = []
        for i, site in enumerate(crystal.sites):
            if species in site.species:
                indices.append(i)
        
        if not indices:
            raise ValueError(f"No sites found containing species '{species}'")
        
        return indices
    
    @staticmethod
    def select_by_label(crystal: Crystal, label: str) -> List[int]:
        """Select sites by label"""
        indices = []
        for i, site in enumerate(crystal.sites):
            if site.label == label:
                indices.append(i)
        
        if not indices:
            raise ValueError(f"No sites found with label '{label}'")
        
        return indices

# ============================================================================
# CORE EDITING OPERATIONS
# ============================================================================

class CrystalEditor:
    """Main editing engine for crystal structures"""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def substitute(
        self, 
        crystal: Crystal, 
        site_sel: Union[int, List[int], str], 
        new_species: Union[str, Dict[str, float]], 
        occupancy: float = 1.0
    ) -> Tuple[Crystal, PatchRecord]:
        """
        Substitute species at selected sites using PyMatgen.
        
        Args:
            crystal: Input crystal structure
            site_sel: Site selection (index, list of indices, Wyckoff position, or species)
            new_species: New species to substitute (element string or species dict)
            occupancy: Occupancy of the new species (default 1.0)
            
        Returns:
            Tuple of (modified_crystal, patch_record)
        """
        # Parse site selection
        if isinstance(site_sel, str):
            if site_sel.startswith("Wyckoff:"):
                wyckoff = site_sel.split(":", 1)[1]
                site_indices = SiteSelector.select_by_wyckoff(crystal, wyckoff)
            elif site_sel.startswith("Species:"):
                species = site_sel.split(":", 1)[1]
                site_indices = SiteSelector.select_by_species(crystal, species)
            elif site_sel.startswith("Label:"):
                label = site_sel.split(":", 1)[1]
                site_indices = SiteSelector.select_by_label(crystal, label)
            else:
                # Assume it's a Wyckoff position directly
                site_indices = SiteSelector.select_by_wyckoff(crystal, site_sel)
        else:
            site_indices = SiteSelector.select_by_index(crystal, site_sel)
        
        # Parse new species
        if isinstance(new_species, str):
            species_dict = {new_species: occupancy}
        else:
            species_dict = new_species.copy()
            # Normalize occupancies to sum to 1.0
            total_occupancy = sum(species_dict.values())
            if total_occupancy > 0:
                species_dict = {k: v/total_occupancy for k, v in species_dict.items()}
        
        # Check preconditions
        preconditions = {
            "site_indices": site_indices,
            "original_species": [crystal.sites[i].species for i in site_indices],
            "symmetry_locked": crystal.constraints.symmetry_locked if crystal.constraints else False
        }
        
        # Check if symmetry is locked
        if crystal.constraints and crystal.constraints.symmetry_locked:
            raise ValueError("Cannot substitute species: symmetry is locked")
        
        # Convert to PyMatgen Structure
        pymatgen_structure = PyMatgenAdapter.crystal_to_pymatgen(crystal)
        
        # Perform substitution using PyMatgen
        for site_idx in site_indices:
            if site_idx < len(pymatgen_structure.sites):
                # Get the primary species to replace
                original_species = list(crystal.sites[site_idx].species.keys())[0]
                
                # Replace with new species
                pymatgen_structure.replace(site_idx, new_species, pymatgen_structure.sites[site_idx].frac_coords)
        
        # Convert back to AtomForge Crystal
        new_crystal = PyMatgenAdapter.pymatgen_to_crystal(pymatgen_structure, crystal)
        
        # Compute result hash
        result_hash = identity_hash(new_crystal)
        
        # Create patch record
        patch_record = PatchRecord(
            op="substitute",
            params={
                "site_sel": site_sel,
                "new_species": new_species,
                "occupancy": occupancy
            },
            preconditions=preconditions,
            result_hash=result_hash,
            timestamp=datetime.now().isoformat()
        )
        
        return new_crystal, patch_record
    
    def vacancy(
        self, 
        crystal: Crystal, 
        site_sel: Union[int, List[int], str], 
        occupancy: float = 1.0
    ) -> Tuple[Crystal, PatchRecord]:
        """
        Create vacancies at selected sites using PyMatgen.
        
        Args:
            crystal: Input crystal structure
            site_sel: Site selection (index, list of indices, Wyckoff position, or species)
            occupancy: Fraction of sites to make vacant (0.0 = all vacant, 1.0 = none vacant)
            
        Returns:
            Tuple of (modified_crystal, patch_record)
        """
        # Parse site selection
        if isinstance(site_sel, str):
            if site_sel.startswith("Wyckoff:"):
                wyckoff = site_sel.split(":", 1)[1]
                site_indices = SiteSelector.select_by_wyckoff(crystal, wyckoff)
            elif site_sel.startswith("Species:"):
                species = site_sel.split(":", 1)[1]
                site_indices = SiteSelector.select_by_species(crystal, species)
            elif site_sel.startswith("Label:"):
                label = site_sel.split(":", 1)[1]
                site_indices = SiteSelector.select_by_label(crystal, label)
            else:
                site_indices = SiteSelector.select_by_wyckoff(crystal, site_sel)
        else:
            site_indices = SiteSelector.select_by_index(crystal, site_sel)
        
        # Check preconditions
        preconditions = {
            "site_indices": site_indices,
            "original_occupancies": [sum(site.species.values()) for i, site in enumerate(crystal.sites) if i in site_indices],
            "symmetry_locked": crystal.constraints.symmetry_locked if crystal.constraints else False
        }
        
        # Check if symmetry is locked
        if crystal.constraints and crystal.constraints.symmetry_locked:
            raise ValueError("Cannot create vacancies: symmetry is locked")
        
        # Convert to PyMatgen Structure
        pymatgen_structure = PyMatgenAdapter.crystal_to_pymatgen(crystal)
        
        # Create vacancies using PyMatgen
        vacancy_fraction = 1.0 - occupancy
        sites_to_remove = []
        
        for site_idx in site_indices:
            if site_idx < len(pymatgen_structure.sites):
                # For partial vacancies, we'll remove sites based on occupancy
                if vacancy_fraction >= 1.0:
                    sites_to_remove.append(site_idx)
                elif vacancy_fraction > 0.0:
                    # For partial vacancies, we could implement probabilistic removal
                    # For now, we'll remove sites if vacancy_fraction > 0.5
                    if vacancy_fraction > 0.5:
                        sites_to_remove.append(site_idx)
        
        # Remove sites in reverse order to maintain indices
        for site_idx in sorted(sites_to_remove, reverse=True):
            pymatgen_structure.remove_sites([site_idx])
        
        # Convert back to AtomForge Crystal
        new_crystal = PyMatgenAdapter.pymatgen_to_crystal(pymatgen_structure, crystal)
        
        # Compute result hash
        result_hash = identity_hash(new_crystal)
        
        # Create patch record
        patch_record = PatchRecord(
            op="vacancy",
            params={
                "site_sel": site_sel,
                "occupancy": occupancy
            },
            preconditions=preconditions,
            result_hash=result_hash,
            timestamp=datetime.now().isoformat()
        )
        
        return new_crystal, patch_record
    
    def interstitial(
        self, 
        crystal: Crystal, 
        frac: Vec3, 
        species: Union[str, Dict[str, float]], 
        occupancy: float = 1.0
    ) -> Tuple[Crystal, PatchRecord]:
        """
        Add interstitial atoms at specified fractional coordinates using PyMatgen.
        
        Args:
            crystal: Input crystal structure
            frac: Fractional coordinates for the interstitial site
            species: Species to add (element string or species dict)
            occupancy: Occupancy of the interstitial species (default 1.0)
            
        Returns:
            Tuple of (modified_crystal, patch_record)
        """
        # Parse species
        if isinstance(species, str):
            species_dict = {species: occupancy}
        else:
            species_dict = species.copy()
            # Normalize occupancies to sum to 1.0
            total_occupancy = sum(species_dict.values())
            if total_occupancy > 0:
                species_dict = {k: v/total_occupancy for k, v in species_dict.items()}
        
        # Check preconditions
        preconditions = {
            "frac_coords": frac,
            "symmetry_locked": crystal.constraints.symmetry_locked if crystal.constraints else False
        }
        
        # Check if symmetry is locked
        if crystal.constraints and crystal.constraints.symmetry_locked:
            raise ValueError("Cannot add interstitials: symmetry is locked")
        
        # Check for minimum interatomic distance if specified
        if crystal.constraints and crystal.constraints.min_interatomic_distance:
            min_dist = crystal.constraints.min_interatomic_distance
            if not self._check_minimum_distance(crystal, frac, min_dist):
                raise ValueError(f"Interstitial site too close to existing atoms (min distance: {min_dist} Å)")
        
        # Convert to PyMatgen Structure
        pymatgen_structure = PyMatgenAdapter.crystal_to_pymatgen(crystal)
        
        # Add interstitial using PyMatgen
        primary_species = list(species_dict.keys())[0]
        pymatgen_structure.append(primary_species, list(frac))
        
        # Convert back to AtomForge Crystal
        new_crystal = PyMatgenAdapter.pymatgen_to_crystal(pymatgen_structure, crystal)
        
        # Compute result hash
        result_hash = identity_hash(new_crystal)
        
        # Create patch record
        patch_record = PatchRecord(
            op="interstitial",
            params={
                "frac": frac,
                "species": species,
                "occupancy": occupancy
            },
            preconditions=preconditions,
            result_hash=result_hash,
            timestamp=datetime.now().isoformat()
        )
        
        return new_crystal, patch_record
    
    def set_lattice(self, crystal: Crystal, new_lattice: Lattice) -> Tuple[Crystal, PatchRecord]:
        """
        Set new lattice parameters.
        
        Args:
            crystal: Input crystal structure
            new_lattice: New lattice parameters
            
        Returns:
            Tuple of (modified_crystal, patch_record)
        """
        # Check preconditions
        preconditions = {
            "original_lattice": crystal.lattice,
            "symmetry_locked": crystal.constraints.symmetry_locked if crystal.constraints else False
        }
        
        # Check if symmetry is locked
        if crystal.constraints and crystal.constraints.symmetry_locked:
            raise ValueError("Cannot change lattice: symmetry is locked")
        
        # Create new crystal with new lattice
        new_crystal = Crystal(
            lattice=new_lattice,
            symmetry=crystal.symmetry,
            sites=crystal.sites,
            composition=crystal.composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
        
        # Compute result hash
        result_hash = identity_hash(new_crystal)
        
        # Create patch record
        patch_record = PatchRecord(
            op="set_lattice",
            params={
                "new_lattice": new_lattice
            },
            preconditions=preconditions,
            result_hash=result_hash,
            timestamp=datetime.now().isoformat()
        )
        
        return new_crystal, patch_record
    
    def set_symmetry(self, crystal: Crystal, new_symmetry: Symmetry) -> Tuple[Crystal, PatchRecord]:
        """
        Set new symmetry information.
        
        Args:
            crystal: Input crystal structure
            new_symmetry: New symmetry information
            
        Returns:
            Tuple of (modified_crystal, patch_record)
        """
        # Check preconditions
        preconditions = {
            "original_symmetry": crystal.symmetry,
            "symmetry_locked": crystal.constraints.symmetry_locked if crystal.constraints else False
        }
        
        # Check if symmetry is locked
        if crystal.constraints and crystal.constraints.symmetry_locked:
            raise ValueError("Cannot change symmetry: symmetry is locked")
        
        # Create new crystal with new symmetry
        new_crystal = Crystal(
            lattice=crystal.lattice,
            symmetry=new_symmetry,
            sites=crystal.sites,
            composition=crystal.composition,
            oxidation_states=crystal.oxidation_states,
            constraints=crystal.constraints,
            provenance=crystal.provenance,
            notes=crystal.notes
        )
        
        # Compute result hash
        result_hash = identity_hash(new_crystal)
        
        # Create patch record
        patch_record = PatchRecord(
            op="set_symmetry",
            params={
                "new_symmetry": new_symmetry
            },
            preconditions=preconditions,
            result_hash=result_hash,
            timestamp=datetime.now().isoformat()
        )
        
        return new_crystal, patch_record
    
    def _update_composition(self, crystal: Crystal, site_indices: List[int], new_species: Dict[str, float]) -> Composition:
        """Update composition after site substitution"""
        # Get original composition
        original_reduced = crystal.composition.reduced.copy()
        original_fractions = crystal.composition.atomic_fractions.copy()
        
        # Calculate changes from substituted sites
        species_changes = {}
        for i in site_indices:
            site = crystal.sites[i]
            # Remove old species
            for species, occupancy in site.species.items():
                if species in species_changes:
                    species_changes[species] -= occupancy
                else:
                    species_changes[species] = -occupancy
            
            # Add new species
            for species, occupancy in new_species.items():
                if species in species_changes:
                    species_changes[species] += occupancy
                else:
                    species_changes[species] = occupancy
        
        # Apply changes
        new_reduced = original_reduced.copy()
        new_fractions = original_fractions.copy()
        
        for species, change in species_changes.items():
            if species in new_reduced:
                new_reduced[species] += change
            else:
                new_reduced[species] = change
            
            if species in new_fractions:
                new_fractions[species] += change
            else:
                new_fractions[species] = change
        
        # Remove zero or negative counts
        new_reduced = {k: v for k, v in new_reduced.items() if v > 0}
        new_fractions = {k: v for k, v in new_fractions.items() if v > 0}
        
        return Composition(reduced=new_reduced, atomic_fractions=new_fractions)
    
    def _update_composition_from_sites(self, sites: Tuple[Site, ...]) -> Composition:
        """Update composition from site occupancies"""
        species_counts = {}
        total_sites = len(sites)
        
        for site in sites:
            for species, occupancy in site.species.items():
                if species in species_counts:
                    species_counts[species] += occupancy
                else:
                    species_counts[species] = occupancy
        
        # Normalize to atomic fractions
        total_atoms = sum(species_counts.values())
        atomic_fractions = {k: v/total_atoms for k, v in species_counts.items()}
        
        return Composition(reduced=species_counts, atomic_fractions=atomic_fractions)
    
    def _check_minimum_distance(self, crystal: Crystal, frac: Vec3, min_distance: float) -> bool:
        """Check if fractional coordinates are far enough from existing sites"""
        # Convert fractional coordinates to Cartesian
        # This is a simplified check - in practice, you'd use the full lattice transformation
        for site in crystal.sites:
            # Calculate distance in fractional coordinates (simplified)
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(frac, site.frac)))
            if dist < min_distance / 10.0:  # Rough conversion from Å to fractional units
                return False
        return True

# ============================================================================
# SUPERCELL OPERATIONS
# ============================================================================

class SupercellBuilder:
    """Builder for supercell structures using PyMatgen"""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def make_supercell(self, crystal: Crystal, M: Mat3) -> Tuple[Crystal, SupercellMap, PatchRecord]:
        """
        Create supercell by applying transformation matrix M using PyMatgen.
        
        Args:
            crystal: Input crystal structure
            M: 3x3 transformation matrix
            
        Returns:
            Tuple of (supercell_crystal, supercell_map, patch_record)
        """
        # Validate transformation matrix
        M_array = np.array(M)
        if M_array.shape != (3, 3):
            raise ValueError("Transformation matrix must be 3x3")
        
        # Check preconditions
        preconditions = {
            "transformation_matrix": M,
            "original_sites": len(crystal.sites),
            "symmetry_locked": crystal.constraints.symmetry_locked if crystal.constraints else False
        }
        
        # Check if symmetry is locked
        if crystal.constraints and crystal.constraints.symmetry_locked:
            raise ValueError("Cannot create supercell: symmetry is locked")
        
        # Convert to PyMatgen Structure
        pymatgen_structure = PyMatgenAdapter.crystal_to_pymatgen(crystal)
        
        # Create supercell using PyMatgen
        supercell_structure = pymatgen_structure.make_supercell(M_array)
        
        # Convert back to AtomForge Crystal
        supercell_crystal = PyMatgenAdapter.pymatgen_to_crystal(supercell_structure, crystal)
        
        # Create supercell map (simplified version)
        child_to_parent = {}
        parent_to_children = {i: [] for i in range(len(crystal.sites))}
        
        # For PyMatgen supercells, we need to reconstruct the mapping
        # This is a simplified implementation - full implementation would track the exact mapping
        child_site_index = 0
        for parent_site_index in range(len(crystal.sites)):
            # Estimate how many child sites come from each parent site
            sites_per_parent = len(supercell_structure.sites) // len(crystal.sites)
            for i in range(sites_per_parent):
                if child_site_index < len(supercell_structure.sites):
                    lattice_vector = (i % M_array[0, 0], (i // M_array[0, 0]) % M_array[1, 1], i // (M_array[0, 0] * M_array[1, 1]))
                    child_to_parent[child_site_index] = (parent_site_index, lattice_vector)
                    parent_to_children[parent_site_index].append((child_site_index, lattice_vector))
                    child_site_index += 1
        
        # Create supercell map
        supercell_map = SupercellMap(
            child_to_parent=child_to_parent,
            parent_to_children=parent_to_children,
            transformation_matrix=M,
            child_lattice=supercell_crystal.lattice,
            parent_lattice=crystal.lattice
        )
        
        # Compute result hash
        result_hash = identity_hash(supercell_crystal)
        
        # Create patch record
        patch_record = PatchRecord(
            op="make_supercell",
            params={
                "transformation_matrix": M
            },
            preconditions=preconditions,
            result_hash=result_hash,
            timestamp=datetime.now().isoformat()
        )
        
        return supercell_crystal, supercell_map, patch_record
    

# ============================================================================
# EXPORT OPERATIONS
# ============================================================================

class CrystalExporter:
    """Export crystal structures to various formats"""
    
    @staticmethod
    def to_poscar(crystal: Crystal) -> Dict[str, str]:
        """
        Export crystal to POSCAR format using PyMatgen.
        
        Args:
            crystal: Crystal structure to export
            
        Returns:
            Dictionary with POSCAR content and metadata
        """
        # Convert to PyMatgen Structure
        pymatgen_structure = PyMatgenAdapter.crystal_to_pymatgen(crystal)
        
        # Export to POSCAR format
        poscar_content = pymatgen_structure.to(fmt="poscar")
        
        return {
            "poscar": poscar_content,
            "formula": pymatgen_structure.composition.formula,
            "space_group": crystal.symmetry.space_group,
            "lattice_params": {
                "a": crystal.lattice.a,
                "b": crystal.lattice.b,
                "c": crystal.lattice.c,
                "alpha": crystal.lattice.alpha,
                "beta": crystal.lattice.beta,
                "gamma": crystal.lattice.gamma
            },
            "sites_count": len(crystal.sites)
        }
    
    @staticmethod
    def to_cif(crystal: Crystal) -> str:
        """
        Export crystal to CIF format.
        
        Args:
            crystal: Crystal structure to export
            
        Returns:
            CIF content as string
        """
        lines = []
        
        # CIF header
        lines.append("data_crystal")
        lines.append("")
        
        # Lattice parameters
        lines.append(f"_cell_length_a    {crystal.lattice.a:.6f}")
        lines.append(f"_cell_length_b    {crystal.lattice.b:.6f}")
        lines.append(f"_cell_length_c    {crystal.lattice.c:.6f}")
        lines.append(f"_cell_angle_alpha {crystal.lattice.alpha:.6f}")
        lines.append(f"_cell_angle_beta  {crystal.lattice.beta:.6f}")
        lines.append(f"_cell_angle_gamma {crystal.lattice.gamma:.6f}")
        lines.append("")
        
        # Space group
        lines.append(f"_space_group_name_H-M    '{crystal.symmetry.space_group}'")
        lines.append(f"_space_group_IT_number   {crystal.symmetry.number}")
        lines.append("")
        
        # Atomic positions
        lines.append("loop_")
        lines.append("_atom_site_label")
        lines.append("_atom_site_type_symbol")
        lines.append("_atom_site_fract_x")
        lines.append("_atom_site_fract_y")
        lines.append("_atom_site_fract_z")
        lines.append("_atom_site_occupancy")
        
        # Create element counters for proper labeling
        element_counters = {}
        
        for i, site in enumerate(crystal.sites):
            for species, occupancy in site.species.items():
                # Use the utility function to get appropriate species symbol
                element = species.split('+')[0].split('-')[0]  # Extract element name
                symbol = PyMatgenAdapter.get_species_symbol(species, crystal)
                
                # Create numbered label
                if element not in element_counters:
                    element_counters[element] = 0
                element_counters[element] += 1
                label = f"{element}{element_counters[element]-1}"
                
                lines.append(f"{label:8s} {symbol:4s} {site.frac[0]:8.5f} {site.frac[1]:8.5f} {site.frac[2]:8.5f} {occupancy:6.4f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def to_pymatgen(crystal: Crystal):
        """
        Export crystal to pymatgen Structure.
        
        Args:
            crystal: Crystal structure to export
            
        Returns:
            pymatgen Structure object
        """
        try:
            from pymatgen.core import Structure, Lattice as PMGLattice
        except ImportError:
            raise ImportError("pymatgen is required for pymatgen export. Install with: pip install pymatgen")
        
        # Create pymatgen lattice
        pmg_lattice = PMGLattice.from_parameters(
            crystal.lattice.a, crystal.lattice.b, crystal.lattice.c,
            crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma
        )
        
        # Create species and coords lists
        species = []
        coords = []
        for site in crystal.sites:
            for element, occupancy in site.species.items():
                species.append(element)
                coords.append(list(site.frac))
        
        # Create pymatgen structure
        structure = Structure(pmg_lattice, species, coords)
        
        return structure
    
    @staticmethod
    def from_pymatgen(structure) -> Crystal:
        """
        Import crystal from pymatgen Structure.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Crystal object
        """
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        except ImportError:
            raise ImportError("pymatgen is required for pymatgen import. Install with: pip install pymatgen")
        
        # Extract lattice parameters
        lattice = Lattice(
            a=structure.lattice.a,
            b=structure.lattice.b,
            c=structure.lattice.c,
            alpha=structure.lattice.alpha,
            beta=structure.lattice.beta,
            gamma=structure.lattice.gamma
        )
        
        # Extract symmetry information
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-5)
        space_group_symbol = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()
        
        symmetry = Symmetry(
            space_group=space_group_symbol,
            number=space_group_number,
            symmetry_source="pymatgen_inferred"
        )
        
        # Extract sites
        sites = []
        for site in structure:
            # Convert species to occupancy dict
            species_dict = {}
            for species, occupancy in site.species.items():
                species_dict[str(species)] = occupancy
            
            # Get fractional coordinates
            frac_coords = tuple(site.frac_coords)
            
            # Create Site object
            site_obj = Site(
                species=species_dict,
                frac=frac_coords,
                wyckoff=None,  # Will be inferred during canonicalization
                multiplicity=None,
                label=site.label if hasattr(site, 'label') else None
            )
            sites.append(site_obj)
        
        # Compute composition
        composition_dict = structure.composition.as_dict()
        
        composition = Composition(
            reduced=composition_dict,
            atomic_fractions=composition_dict
        )
        
        # Create provenance
        provenance = Provenance(
            database="pymatgen",
            retrieved_at=datetime.now().isoformat(),
            schema_version="AtomForge/Crystal/1.1"
        )
        
        return Crystal(
            lattice=lattice,
            symmetry=symmetry,
            sites=tuple(sites),
            composition=composition,
            provenance=provenance
        )

# ============================================================================
# MAIN PHASE 2 OPERATIONS (API)
# ============================================================================

def substitute(crystal: Crystal, site_sel: Union[int, List[int], str], new_species: Union[str, Dict[str, float]], occupancy: float = 1.0) -> Tuple[Crystal, PatchRecord]:
    """Substitute species at selected sites"""
    editor = CrystalEditor()
    return editor.substitute(crystal, site_sel, new_species, occupancy)

def vacancy(crystal: Crystal, site_sel: Union[int, List[int], str], occupancy: float = 1.0) -> Tuple[Crystal, PatchRecord]:
    """Create vacancies at selected sites"""
    editor = CrystalEditor()
    return editor.vacancy(crystal, site_sel, occupancy)

def interstitial(crystal: Crystal, frac: Vec3, species: Union[str, Dict[str, float]], occupancy: float = 1.0) -> Tuple[Crystal, PatchRecord]:
    """Add interstitial atoms at specified fractional coordinates"""
    editor = CrystalEditor()
    return editor.interstitial(crystal, frac, species, occupancy)

def set_lattice(crystal: Crystal, new_lattice: Lattice) -> Tuple[Crystal, PatchRecord]:
    """Set new lattice parameters"""
    editor = CrystalEditor()
    return editor.set_lattice(crystal, new_lattice)

def set_symmetry(crystal: Crystal, new_symmetry: Symmetry) -> Tuple[Crystal, PatchRecord]:
    """Set new symmetry information"""
    editor = CrystalEditor()
    return editor.set_symmetry(crystal, new_symmetry)

def make_supercell(crystal: Crystal, M: Mat3) -> Tuple[Crystal, SupercellMap, PatchRecord]:
    """Create supercell by applying transformation matrix M"""
    builder = SupercellBuilder()
    return builder.make_supercell(crystal, M)

def to_poscar(crystal: Crystal) -> Dict[str, str]:
    """Export crystal to POSCAR format"""
    return CrystalExporter.to_poscar(crystal)

def to_cif(crystal: Crystal) -> str:
    """Export crystal to CIF format"""
    return CrystalExporter.to_cif(crystal)

def to_pymatgen(crystal: Crystal):
    """Export crystal to pymatgen Structure"""
    return CrystalExporter.to_pymatgen(crystal)

def from_pymatgen(structure) -> Crystal:
    """Import crystal from pymatgen Structure"""
    return CrystalExporter.from_pymatgen(structure)
