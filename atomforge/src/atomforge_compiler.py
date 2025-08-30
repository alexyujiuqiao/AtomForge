#!/usr/bin/env python3
"""
AtomForge DSL v2.1 Compiler Implementation

A comprehensive compiler for AtomForge DSL v2.1 that inherits from the core DSLCompiler,
supporting compilation to various output formats including JSON, CIF, and VASP formats.
Based on the MVP compiler implementation using pymatgen for materials science output.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from dataclasses import asdict

# Add core to path for inheritance
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from core.compiler import DSLCompiler

# Import pymatgen for materials science output formats
try:
    from pymatgen.core import Lattice as PmgLattice, Structure
    from pymatgen.io.vasp import Poscar
    from pymatgen.io.cif import CifWriter
    PYMAGEN_AVAILABLE = True
except ImportError:
    PYMAGEN_AVAILABLE = False
    print("Warning: pymatgen not available. VASP and CIF output will be limited.")

from .atomforge_ir import AtomForgeProgram
from .atomforge_parser import AtomForgeParser, AtomForgeTransformer


class AtomForgeCompiler(DSLCompiler):
    """
    AtomForge DSL v2.1 compiler that inherits from the core DSLCompiler.
    Compiles AtomForge programs into various output formats.
    """
    
    def __init__(self, output_format: str = "json"):
        """
        Initialize the AtomForge v2.1 compiler.
        
        Args:
            output_format: Target output format ("json", "cif", "vasp", "dict")
        """
        parser = AtomForgeParser()
        transformer = AtomForgeTransformer()
        super().__init__(parser, transformer)
        self.output_format = output_format
    
    def compile(self, code: str) -> Union[str, Dict[str, Any]]:
        """
        Compile AtomForge DSL v2.1 code to the specified output format.
        
        Args:
            code: AtomForge DSL v2.1 source code as a string
            
        Returns:
            Compiled output in the specified format
        """
        # Parse and transform the code
        program = self.parser.parse_and_transform(code)
        
        # Validate the program
        program.validate()
        
        # Compile based on output format
        if self.output_format == "json":
            return self._compile_to_json(program)
        elif self.output_format == "cif":
            return self._compile_to_cif(program)
        elif self.output_format == "vasp":
            return self._compile_to_vasp(program)
        elif self.output_format == "dict":
            return self._compile_to_dict(program)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
    
    def _compile_to_json(self, program: AtomForgeProgram) -> str:
        """Compile program to JSON format using pymatgen-compatible approach"""
        # Convert dataclasses to native Python structures
        raw = asdict(program)
        
        # Clean the data (remove None or empty entries)
        def clean(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if v is not None and v != []}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            return obj
        
        cleaned = clean(raw)
        return json.dumps(cleaned, indent=2)
    
    def _compile_to_cif(self, program: AtomForgeProgram) -> str:
        """Compile program to CIF format using pymatgen"""
        if not PYMAGEN_AVAILABLE:
            return self._compile_to_cif_fallback(program)
        
        try:
            # Build the lattice
            if not program.lattice or not program.lattice.bravais:
                raise ValueError("Lattice information is required for CIF generation")
            
            lat = program.lattice.bravais
            lattice = PmgLattice.from_parameters(
                lat.a, lat.b, lat.c,
                lat.alpha, lat.beta, lat.gamma
            )
            
            # Build species and coords lists
            species_symbols = []
            frac_coords = []
            
            if program.basis:
                for site in program.basis:
                    if site.species:
                        for sp in site.species:
                            species_symbols.append(sp.element)
                            frac_coords.append(site.position)
            
            # Create a Structure
            struct = Structure(
                lattice,
                species_symbols,
                frac_coords,
                coords_are_cartesian=False
            )
            
            writer = CifWriter(struct)
            return writer.__str__()
            
        except Exception as e:
            print(f"Error generating CIF with pymatgen: {e}")
            return self._compile_to_cif_fallback(program)
    
    def _compile_to_cif_fallback(self, program: AtomForgeProgram) -> str:
        """Fallback CIF generation without pymatgen"""
        cif_lines = []
        
        # Header
        cif_lines.append(f"data_{program.identifier}")
        cif_lines.append(f"_atomforge_dsl_version {program.header.dsl_version}")
        cif_lines.append(f"_atomforge_title {program.header.title}")
        cif_lines.append(f"_atomforge_created {program.header.created.isoformat()}")
        
        # Lattice parameters
        if program.lattice and program.lattice.bravais:
            bravais = program.lattice.bravais
            cif_lines.append(f"_cell_length_a {bravais.a}")
            cif_lines.append(f"_cell_length_b {bravais.b}")
            cif_lines.append(f"_cell_length_c {bravais.c}")
            cif_lines.append(f"_cell_angle_alpha {bravais.alpha}")
            cif_lines.append(f"_cell_angle_beta {bravais.beta}")
            cif_lines.append(f"_cell_angle_gamma {bravais.gamma}")
        
        # Space group
        if program.symmetry:
            cif_lines.append(f"_symmetry_space_group_name_H-M {program.symmetry.space_group}")
        
        # Atomic positions
        if program.basis:
            cif_lines.append("loop_")
            cif_lines.append("_atom_site_label")
            cif_lines.append("_atom_site_type_symbol")
            cif_lines.append("_atom_site_fract_x")
            cif_lines.append("_atom_site_fract_y")
            cif_lines.append("_atom_site_fract_z")
            cif_lines.append("_atom_site_occupancy")
            
            for site in program.basis:
                if site.species:
                    for species in site.species:
                        cif_lines.append(f"{site.name} {species.element} {site.position[0]:.6f} {site.position[1]:.6f} {site.position[2]:.6f} {species.occupancy:.6f}")
        
        return "\n".join(cif_lines)
    
    def _compile_to_vasp(self, program: AtomForgeProgram) -> str:
        """Compile program to VASP POSCAR format using pymatgen"""
        if not PYMAGEN_AVAILABLE:
            return self._compile_to_vasp_fallback(program)
        
        try:
            # Build the lattice
            if not program.lattice or not program.lattice.bravais:
                raise ValueError("Lattice information is required for VASP generation")
            
            lat = program.lattice.bravais
            lattice = PmgLattice.from_parameters(
                lat.a, lat.b, lat.c,
                lat.alpha, lat.beta, lat.gamma
            )
            
            # Build species and coords lists
            species_symbols = []
            frac_coords = []
            
            if program.basis:
                for site in program.basis:
                    if site.species:
                        for sp in site.species:
                            species_symbols.append(sp.element)
                            frac_coords.append(site.position)
            
            # Create a Structure
            struct = Structure(
                lattice,
                species_symbols,
                frac_coords,
                coords_are_cartesian=False
            )
            
            poscar = Poscar(struct)
            return poscar.get_string()
            
        except Exception as e:
            print(f"Error generating VASP with pymatgen: {e}")
            return self._compile_to_vasp_fallback(program)
    
    def _compile_to_vasp_fallback(self, program: AtomForgeProgram) -> str:
        """Fallback VASP generation without pymatgen"""
        vasp_lines = []
        
        # Title
        vasp_lines.append(program.header.title)
        
        # Scaling factor
        vasp_lines.append("1.0")
        
        # Lattice vectors
        if program.lattice and program.lattice.bravais:
            bravais = program.lattice.bravais
            # Convert to cartesian coordinates (simplified)
            vasp_lines.append(f"{bravais.a:.6f} 0.000000 0.000000")
            vasp_lines.append(f"0.000000 {bravais.b:.6f} 0.000000")
            vasp_lines.append(f"0.000000 0.000000 {bravais.c:.6f}")
        
        # Element symbols
        if program.basis:
            elements = set()
            for site in program.basis:
                if site.species:
                    for species in site.species:
                        elements.add(species.element)
            vasp_lines.append(" ".join(sorted(elements)))
        
        # Element counts
        if program.basis:
            element_counts = {}
            for site in program.basis:
                if site.species:
                    for species in site.species:
                        element_counts[species.element] = element_counts.get(species.element, 0) + 1
            vasp_lines.append(" ".join(str(element_counts.get(elem, 0)) for elem in sorted(elements)))
        
        # Direct coordinates
        vasp_lines.append("Direct")
        
        # Atomic positions
        if program.basis:
            for site in program.basis:
                if site.species:
                    for species in site.species:
                        vasp_lines.append(f"{site.position[0]:.6f} {site.position[1]:.6f} {site.position[2]:.6f}")
        
        return "\n".join(vasp_lines)
    
    def _compile_to_dict(self, program: AtomForgeProgram) -> Dict[str, Any]:
        """Compile program to dictionary format"""
        def serialize_dataclass(obj):
            if hasattr(obj, '__dict__'):
                return {k: serialize_dataclass(v) for k, v in obj.__dict__.items() 
                       if not k.startswith('_')}
            elif isinstance(obj, (list, tuple)):
                return [serialize_dataclass(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize_dataclass(v) for k, v in obj.items()}
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        return serialize_dataclass(program)
    
    def _compile(self, program: AtomForgeProgram) -> str:
        """
        Implementation of the abstract _compile method from DSLCompiler.
        Compiles the program to the specified output format.
        
        Args:
            program: The AtomForge program to compile
            
        Returns:
            Compiled output as a string
        """
        # Validate the program
        program.validate()
        
        # Compile based on output format
        if self.output_format == "json":
            return self._compile_to_json(program)
        elif self.output_format == "cif":
            return self._compile_to_cif(program)
        elif self.output_format == "vasp":
            return self._compile_to_vasp(program)
        elif self.output_format == "dict":
            return str(self._compile_to_dict(program))
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
    
    # VASP Input Generators (following MVP approach)
    def generate_incar(self, program: AtomForgeProgram) -> str:
        """Generate VASP INCAR file string based on property validation"""
        lines = []
        
        # System name
        lines.append(f"SYSTEM = {program.identifier}")
        
        # Functional and computational settings
        if program.properties and program.properties.validation:
            pv = program.properties.validation
            if hasattr(pv, 'computational_backend'):
                cb = pv.computational_backend
                functional = getattr(cb, 'functional', None)
                if functional:
                    lines.append(f"GGA = {functional}")
                encut = getattr(cb, 'energy_cutoff', None)
                if encut:
                    lines.append(f"ENCUT = {encut}")
        
        # Precision and convergence
        lines.append("PREC = Accurate")
        if program.properties and program.properties.validation:
            pv = program.properties.validation
            if hasattr(pv, 'convergence_criteria'):
                cc = pv.convergence_criteria
                etol = getattr(cc, 'energy_tolerance', None)
                if etol:
                    lines.append(f"EDIFF = {etol}")
        
        # Add defaults
        lines.append("ISMEAR = 0")
        lines.append("SIGMA = 0.05")
        return "\n".join(lines)
    
    def generate_kpoints(self, program: AtomForgeProgram) -> str:
        """Generate VASP KPOINTS file string using a simple Monkhorst-Pack grid"""
        lines = [
            "KPOINTS generated by AtomForge",
            "0",
            "Monkhorst-Pack"
        ]
        
        # Default grid size from computational_backend
        grid = None
        if program.properties and program.properties.validation:
            pv = program.properties.validation
            if hasattr(pv, 'computational_backend'):
                grid = getattr(pv.computational_backend, 'k_point_density', None)
        
        # Fallback to 1x1x1 if not provided
        try:
            k = int(grid) if grid else 1
        except Exception:
            k = 1
        
        lines.append(f"{k} {k} {k}")
        lines.append("0 0 0")
        return "\n".join(lines)
    
    def generate_potcar(self, program: AtomForgeProgram) -> str:
        """Generate simple POTCAR reference list for each unique element"""
        elements = []
        if program.basis and program.basis.sites:
            for site in program.basis.sites:
                if site.species:
                    for sp in site.species:
                        if sp.element not in elements:
                            elements.append(sp.element)
        
        # Construct POTCAR entries
        lines = [f"POTCAR for {el}: PAW_PBE {el}" for el in elements]
        return "\n".join(lines)
    
    def to_dict(self, code: str) -> Dict[str, Any]:
        """
        Returns a dictionary representing all intermediate stages of compilation.
        
        Args:
            code: AtomForge DSL v2.1 source code as a string
            
        Returns:
            Dictionary with source code, parse tree, AST, and compiler output
        """
        parse_tree = self.parser.parse(code)
        program = self.transformer.transform(parse_tree)
        output = self._compile_to_json(program)
        
        return {
            "source_code": code,
            "parse_tree": str(parse_tree),
            "ast": program,
            "compiler_output": output
        }


# Convenience functions
def compile_atomforge(content: str, output_format: str = "json") -> Union[str, Dict[str, Any]]:
    """
    Convenience function to compile AtomForge DSL v2.1 content.
    
    Args:
        content: The AtomForge DSL v2.1 content to compile
        output_format: Target output format ("json", "cif", "vasp", "dict")
        
    Returns:
        Compiled output in the specified format
    """
    compiler = AtomForgeCompiler(output_format)
    return compiler.compile(content)


def compile_atomforge_file(file_path: str, output_format: str = "json") -> Union[str, Dict[str, Any]]:
    """
    Convenience function to compile an AtomForge DSL v2.1 file.
    
    Args:
        file_path: Path to the AtomForge DSL v2.1 file
        output_format: Target output format ("json", "cif", "vasp", "dict")
        
    Returns:
        Compiled output in the specified format
    """
    compiler = AtomForgeCompiler(output_format)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return compiler.compile(content)


if __name__ == "__main__":
    # Example usage
    example_content = '''
            atom_spec "silicon_diamond" {
            header { 
                dsl_version = "2.1",
                title = "Silicon Diamond Structure",
                created = 2025-05-08
            }
            lattice { 
                type = cubic, 
                a = 5.431 angstrom 
            }
            symmetry { 
                space_group = 227 
            }
            basis {
                site "Si1" { 
                wyckoff = "8a", 
                position = (0.0, 0.0, 0.0), 
                frame = fractional, 
                species = ({ element = "Si", occupancy = 1.0 }) 
                }
            }
            }
            '''
    
    try:
        # Test JSON compilation
        json_result = compile_atomforge(example_content, "json")
        print("JSON compilation successful!")
        print(json_result[:200] + "...")
        
        # Test CIF compilation
        cif_result = compile_atomforge(example_content, "cif")
        print("\nCIF compilation successful!")
        print(cif_result)
        
    except Exception as e:
        print(f"Compilation failed: {e}") 