#!/usr/bin/env python3
"""
AtomForge Patch System

This module implements a patching system for AtomForge DSL files using the Lark parser.
It supports the patch operations defined in the grammar:
- update_occupancy: Update species occupancy in sites
- add_site: Add new sites to the basis
- modify_property: Modify computational properties
"""

import json
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from lark import Lark, Transformer, Tree, Token
from lark.visitors import Visitor, Interpreter

from .atomforge_parser import parse_atomforge_string, AtomForgeTransformer
from .atomforge_ir import AtomForgeFile, AtomSpec, Site, Species, PropertyValidation

class PatchOperation(Enum):
    """Supported patch operations"""
    UPDATE_OCCUPANCY = "update_occupancy"
    ADD_SITE = "add_site"
    MODIFY_PROPERTY = "modify_property"

@dataclass
class Patch:
    """Represents a single patch operation"""
    op: PatchOperation
    target: str  # Target identifier (e.g., site name, property name)
    value: Any   # New value
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.op, str):
            self.op = PatchOperation(self.op)

@dataclass
class PatchSet:
    """Represents a set of patches to be applied"""
    patches: List[Patch] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_patch(self, patch: Patch):
        """Add a patch to the set"""
        self.patches.append(patch)
    
    def add_patches(self, patches: List[Patch]):
        """Add multiple patches to the set"""
        self.patches.extend(patches)

class AtomForgePatchTransformer(Transformer):
    """
    Transformer for parsing patch operations from AtomForge DSL
    """
    
    def patch(self, operations):
        """Parse a patch block"""
        return list(operations)
    
    def update_occupancy(self, site_name, species_index, occupancy):
        """Parse update_occupancy operation"""
        return {
            'op': PatchOperation.UPDATE_OCCUPANCY,
            'site_name': str(site_name),
            'species_index': int(species_index),
            'occupancy': float(occupancy)
        }
    
    def add_site(self, site_name, wyckoff, position, frame, species, label=None):
        """Parse add_site operation"""
        return {
            'op': PatchOperation.ADD_SITE,
            'site_name': str(site_name),
            'wyckoff': str(wyckoff).strip('"'),
            'position': position,
            'frame': str(frame),
            'species': species,
            'label': str(label).strip('"') if label else None
        }
    
    def modify_property(self, property_name, value):
        """Parse modify_property operation"""
        return {
            'op': PatchOperation.MODIFY_PROPERTY,
            'property_name': str(property_name),
            'value': float(value)
        }

class AtomForgePatcher:
    """
    Main patching system for AtomForge DSL files.
    
    This class applies patches to AtomForge DSL content using the Lark parser
    and the defined IR structures.
    """
    
    def __init__(self):
        # Load the grammar for parsing patches
        from .atomforge_parser import ATOMFORGE_GRAMMAR
        self.patch_parser = Lark(ATOMFORGE_GRAMMAR, start="patch", parser="lalr")
        self.patch_transformer = AtomForgePatchTransformer()
    
    def apply_patch_set(self, dsl_content: str, patch_set: PatchSet) -> str:
        """
        Apply a set of patches to AtomForge DSL content.
        
        Args:
            dsl_content: The original AtomForge DSL content
            patch_set: The set of patches to apply
            
        Returns:
            Modified DSL content
        """
        # Parse the original DSL
        atomforge_file = parse_atomforge_string(dsl_content)
        
        # Apply each patch
        for patch in patch_set.patches:
            try:
                atomforge_file = self._apply_single_patch(atomforge_file, patch)
            except Exception as e:
                print(f"Warning: Failed to apply patch {patch}: {e}")
                continue
        
        # Convert back to DSL format
        return self._atomforge_file_to_dsl(atomforge_file)
    
    def _apply_single_patch(self, atomforge_file: AtomForgeFile, patch: Patch) -> AtomForgeFile:
        """Apply a single patch to the AtomForge file"""
        if patch.op == PatchOperation.UPDATE_OCCUPANCY:
            return self._apply_update_occupancy(atomforge_file, patch)
        elif patch.op == PatchOperation.ADD_SITE:
            return self._apply_add_site(atomforge_file, patch)
        elif patch.op == PatchOperation.MODIFY_PROPERTY:
            return self._apply_modify_property(atomforge_file, patch)
        else:
            raise ValueError(f"Unsupported patch operation: {patch.op}")
    
    def _apply_update_occupancy(self, atomforge_file: AtomForgeFile, patch: Patch) -> AtomForgeFile:
        """Apply update_occupancy patch"""
        site_name = patch.metadata.get('site_name')
        species_index = patch.metadata.get('species_index')
        new_occupancy = patch.value
        
        # Find the site and update occupancy
        for site in atomforge_file.spec.basis:
            if site.name == site_name:
                if 0 <= species_index < len(site.species):
                    site.species[species_index].occupancy = new_occupancy
                    return atomforge_file
                else:
                    raise ValueError(f"Species index {species_index} out of range for site {site_name}")
        
        raise ValueError(f"Site {site_name} not found")
    
    def _apply_add_site(self, atomforge_file: AtomForgeFile, patch: Patch) -> AtomForgeFile:
        """Apply add_site patch"""
        site_name = patch.metadata.get('site_name')
        wyckoff = patch.metadata.get('wyckoff')
        position = patch.metadata.get('position')
        frame = patch.metadata.get('frame')
        species = patch.metadata.get('species')
        label = patch.metadata.get('label')
        
        # Create new site
        new_site = Site(
            name=site_name,
            wyckoff=wyckoff,
            position=position,
            frame=frame,
            species=species,
            label=label
        )
        
        # Add to basis
        atomforge_file.spec.basis.append(new_site)
        return atomforge_file
    
    def _apply_modify_property(self, atomforge_file: AtomForgeFile, patch: Patch) -> AtomForgeFile:
        """Apply modify_property patch"""
        property_name = patch.metadata.get('property_name')
        new_value = patch.value
        
        # Update property validation
        if atomforge_file.spec.property_validation:
            if property_name == "energy_cutoff":
                atomforge_file.spec.property_validation.computational_backend["energy_cutoff"] = new_value
            elif property_name == "k_point_density":
                atomforge_file.spec.property_validation.computational_backend["k_point_density"] = new_value
            elif property_name == "energy_tolerance":
                if atomforge_file.spec.property_validation.convergence_criteria:
                    atomforge_file.spec.property_validation.convergence_criteria.energy_tolerance = new_value
            elif property_name == "force_tolerance":
                if atomforge_file.spec.property_validation.convergence_criteria:
                    atomforge_file.spec.property_validation.convergence_criteria.force_tolerance = new_value
            else:
                raise ValueError(f"Unknown property: {property_name}")
        
        return atomforge_file
    
    def _atomforge_file_to_dsl(self, atomforge_file: AtomForgeFile) -> str:
        """Convert AtomForgeFile back to DSL format"""
        lines = []
        
        # Add version declaration
        if atomforge_file.language_version:
            lines.append(f'#atomforge_version "{atomforge_file.language_version.version}";')
        
        # Start atom_spec
        spec = atomforge_file.spec
        lines.append(f'atom_spec {spec.name} {{')
        
        # Add header
        if spec.header:
            lines.append("  header {")
            lines.append(f'    dsl_version = "{spec.header.dsl_version}",')
            lines.append(f'    title = "{spec.header.title}",')
            lines.append(f'    created = {spec.header.created.strftime("%Y-%m-%d")},')
            if spec.header.uuid:
                lines.append(f'    uuid = "{spec.header.uuid}",')
            lines.append("  }")
        
        # Add description
        if spec.description:
            lines.append(f'  description = "{spec.description}",')
        
        # Add units
        if spec.units:
            lines.append("  units {")
            lines.append(f'    system = "{spec.units.system}",')
            lines.append(f'    length = {spec.units.length},')
            lines.append(f'    angle = {spec.units.angle},')
            lines.append("  }")
        
        # Add lattice
        if spec.lattice:
            lines.append("  lattice {")
            lines.append(f'    type = {spec.lattice.type},')
            lines.append(f'    a = {spec.lattice.a},')
            lines.append(f'    b = {spec.lattice.b},')
            lines.append(f'    c = {spec.lattice.c},')
            lines.append(f'    alpha = {spec.lattice.alpha},')
            lines.append(f'    beta = {spec.lattice.beta},')
            lines.append(f'    gamma = {spec.lattice.gamma},')
            lines.append("  }")
        
        # Add symmetry
        if spec.symmetry:
            lines.append("  symmetry {")
            lines.append(f'    space_group = {spec.symmetry.space_group},')
            if spec.symmetry.origin_choice is not None:
                lines.append(f'    origin_choice = {spec.symmetry.origin_choice},')
            lines.append("  }")
        
        # Add basis
        if spec.basis:
            lines.append("  basis {")
            for site in spec.basis:
                lines.append(f'    site {site.name} {{')
                lines.append(f'      wyckoff = "{site.wyckoff}",')
                lines.append(f'      position = ({site.position[0]}, {site.position[1]}, {site.position[2]}),')
                lines.append(f'      frame = {site.frame},')
                lines.append("      species = (")
                for species in site.species:
                    lines.append("        {")
                    lines.append(f'          element = "{species.element}",')
                    lines.append(f'          occupancy = {species.occupancy},')
                    if species.charge is not None:
                        lines.append(f'          charge = {species.charge},')
                    lines.append("        },")
                lines.append("      ),")
                if site.adp_iso is not None:
                    lines.append(f'      adp_iso = {site.adp_iso},')
                if site.label:
                    lines.append(f'      label = "{site.label}",')
                lines.append("    }")
            lines.append("  }")
        
        # Add property_validation
        if spec.property_validation:
            lines.append("  property_validation {")
            lines.append("    computational_backend: VASP {")
            lines.append(f'      functional: "{spec.property_validation.computational_backend["functional"]}",')
            lines.append(f'      energy_cutoff: {spec.property_validation.computational_backend["energy_cutoff"]},')
            lines.append(f'      k_point_density: {spec.property_validation.computational_backend["k_point_density"]},')
            lines.append("    },")
            
            if spec.property_validation.convergence_criteria:
                lines.append("    convergence_criteria: {")
                if spec.property_validation.convergence_criteria.energy_tolerance is not None:
                    lines.append(f'      energy_tolerance: {spec.property_validation.convergence_criteria.energy_tolerance},')
                if spec.property_validation.convergence_criteria.force_tolerance is not None:
                    lines.append(f'      force_tolerance: {spec.property_validation.convergence_criteria.force_tolerance},')
                if spec.property_validation.convergence_criteria.stress_tolerance is not None:
                    lines.append(f'      stress_tolerance: {spec.property_validation.convergence_criteria.stress_tolerance},')
                lines.append("    },")
            
            if spec.property_validation.target_properties:
                lines.append("    target_properties: {")
                if spec.property_validation.target_properties.formation_energy is not None:
                    lines.append(f'      formation_energy: {str(spec.property_validation.target_properties.formation_energy).lower()},')
                if spec.property_validation.target_properties.band_gap is not None:
                    lines.append(f'      band_gap: {str(spec.property_validation.target_properties.band_gap).lower()},')
                if spec.property_validation.target_properties.elastic_constants is not None:
                    lines.append(f'      elastic_constants: {str(spec.property_validation.target_properties.elastic_constants).lower()},')
                lines.append("    },")
            
            lines.append("  }")
        
        # Add provenance
        if spec.provenance:
            lines.append("  provenance {")
            lines.append(f'    source = "{spec.provenance.source}",')
            if spec.provenance.method:
                lines.append(f'    method = "{spec.provenance.method}",')
            if spec.provenance.doi:
                lines.append(f'    doi = "{spec.provenance.doi}",')
            lines.append("  }")
        
        # Close atom_spec
        lines.append("}")
        
        return "\n".join(lines)

class PatchLoader:
    """Load and manage patch files"""
    
    @staticmethod
    def load_patch_file(file_path: Union[str, Path]) -> PatchSet:
        """Load patches from a JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        patch_set = PatchSet()
        
        if isinstance(data, list):
            # List of patches
            for patch_data in data:
                patch = Patch(**patch_data)
                patch_set.add_patch(patch)
        elif isinstance(data, dict):
            # Single patch or patch set with metadata
            if "patches" in data:
                if "metadata" in data:
                    patch_set.metadata = data["metadata"]
                for patch_data in data["patches"]:
                    patch = Patch(**patch_data)
                    patch_set.add_patch(patch)
            elif "op" in data:
                # Single patch
                patch = Patch(**data)
                patch_set.add_patch(patch)
            else:
                # Just metadata
                patch_set.metadata = data
        
        return patch_set
    
    @staticmethod
    def save_patch_file(patch_set: PatchSet, file_path: Union[str, Path]):
        """Save patches to a JSON file"""
        def serialize_value(value):
            """Serialize values that might not be JSON serializable"""
            if hasattr(value, '__dict__'):
                # Convert dataclass objects to dict
                return {k: serialize_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                return value
        
        data = {
            "metadata": patch_set.metadata,
            "patches": [
                {
                    "op": patch.op.value,
                    "target": patch.target,
                    "value": serialize_value(patch.value),
                    "metadata": serialize_value(patch.metadata)
                }
                for patch in patch_set.patches
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def create_patch_examples():
    """Create example patches for common use cases"""
    
    # Example 1: Update occupancy
    occupancy_patch = Patch(
        op=PatchOperation.UPDATE_OCCUPANCY,
        target="Si1",
        value=0.8,
        metadata={
            'site_name': 'Si1',
            'species_index': 0
        }
    )
    
    # Example 2: Add a new site
    add_site_patch = Patch(
        op=PatchOperation.ADD_SITE,
        target="Si2",
        value=None,
        metadata={
            'site_name': 'Si2',
            'wyckoff': '4b',
            'position': (0.25, 0.25, 0.25),
            'frame': 'fractional',
            'species': [Species(element='Si', occupancy=1.0)],
            'label': 'Additional Si site'
        }
    )
    
    # Example 3: Modify computational property
    property_patch = Patch(
        op=PatchOperation.MODIFY_PROPERTY,
        target="energy_cutoff",
        value=800.0,
        metadata={
            'property_name': 'energy_cutoff'
        }
    )
    
    return [occupancy_patch, add_site_patch, property_patch]

# Example usage
if __name__ == "__main__":
    # Create a patch set
    patch_set = PatchSet()
    patch_set.add_patches(create_patch_examples())
    
    # Example DSL content
    example_dsl = '''
            #atomforge_version "1.0";
            atom_spec Si {
            header {
                dsl_version = "1.0",
                title = "Si",
                created = 2023-10-01,
                uuid = "550e8400-e29b-41d4-a716-446655440000"
            }
            description = "Silicon crystal structure",
            units {
                system = "crystallographic_default",
                length = angstrom,
                angle = degree
            }
            lattice {
                type = cubic,
                a = 5.43,
                b = 5.43,
                c = 5.43,
                alpha = 90.0,
                beta = 90.0,
                gamma = 90.0
            }
            symmetry {
                space_group = "Fd-3m",
                origin_choice = 1
            }
            basis {
                site Si1 {
                wyckoff = "8a",
                position = (0.0, 0.0, 0.0),
                frame = fractional,
                species = (
                    {
                    element = "Si",
                    occupancy = 1.0
                    }
                )
                }
            }
            property_validation {
                computational_backend: VASP {
                functional: "PBE",
                energy_cutoff: 520,
                k_point_density: 1000.0
                }
            }
            provenance {
                source = "Materials Project"
            }
            }
            '''
    
    # Apply patches
    patcher = AtomForgePatcher()
    modified_dsl = patcher.apply_patch_set(example_dsl, patch_set)
    
    print("Original DSL:")
    print(example_dsl)
    print("\nModified DSL:")
    print(modified_dsl) 