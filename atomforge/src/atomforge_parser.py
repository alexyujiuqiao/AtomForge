#!/usr/bin/env python3
"""
AtomForge DSL v2.1 Parser Implementation

A comprehensive parser for AtomForge DSL v2.1 that inherits from the core DSL framework,
supporting all advanced features including AI integration, procedural generation,
benchmarking, and the revolutionary patching system.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

# Add core to path for inheritance
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from core.parser import DSLParser, DSLTransformer

try:
    from lark import Lark, Transformer, v_args, Token
    from lark.exceptions import LarkError, UnexpectedInput
except ImportError:
    raise ImportError("Lark parser library is required. Install with: pip install lark")

from .atomforge_ir import (
    AtomForgeProgram, Header, Units, TypeSystem, StructuralTypes, ChemicalTypes,
    ComputationalTypes, AutoInference, Lattice, Bravais, Symmetry, Superspace,
    Basis, Site, Species, EmergingMaterials, LayerStack, LayerSpec, Topology,
    Metamaterial, QuantumState, AIIntegration, GraphRepresentation, GenerationModel,
    ActiveLearning, MultiFidelity, ProceduralGeneration, TemplateGeneration,
    ParameterSweep, MLGuided, Hybridization, Benchmarking, BenchmarkTask,
    Patch, PatchOperation, Defects, DefectEntry, Tile, Bonds, Bond, Elastic,
    Phonon, Density, Environment, Properties, Validation, Simplification,
    Provenance, Meta, Length, Angle, Temperature, Pressure, Displacement
)

# Grammar file path
GRAMMAR_FILE = Path(__file__).parent / "atomforge.lark"


class AtomForgeParser(DSLParser):
    """
    AtomForge DSL v2.1 parser that inherits from the core DSLParser.
    Loads the AtomForge v2.1 grammar and parses DSL code.
    """
    
    def __init__(self, grammar_file: str = None, start_symbol: str = "start"):
        """
        Initialize the AtomForge v2.1 parser.
        
        Args:
            grammar_file: Path to the AtomForge v2.1 grammar file (optional)
            start_symbol: The start symbol for grammar parsing
        """
        if grammar_file is None:
            grammar_file = str(GRAMMAR_FILE)
        
        super().__init__(grammar_file, start_symbol)
    
    def parse_and_transform(self, code: str) -> AtomForgeProgram:
        """
        Parse the DSL code and transform it into AtomForge v2.1 data structures.
        
        Args:
            code: AtomForge DSL v2.1 code as a string
            
        Returns:
            AtomForgeProgram object
        """
        tree = self.parse(code)
        transformer = AtomForgeTransformer()
        return transformer.transform(tree)


class AtomForgeTransformer(DSLTransformer):
    """
    AtomForge DSL v2.1 transformer that inherits from the core DSLTransformer.
    Converts Lark parse trees into AtomForge v2.1 data structures.
    """
    
    def __init__(self):
        super().__init__()
        self.errors = []
        self.warnings = []
    
    def start(self, args):
        """Transform the start rule"""
        return args[0]
    
    def program(self, args):
        """Transform the main program structure"""
        # Extract identifier and components
        identifier = args[1]
        header = args[3]
        
        # Build program structure
        program_data = {
            "identifier": identifier,
            "header": header,
            "description": None,
            "units": None,
            "type_system": None,
            "lattice": None,
            "symmetry": None,
            "basis": None,
            "emerging_materials": None,
            "defects": None,
            "tile": None,
            "bonds": None,
            "elastic": None,
            "phonon": None,
            "density": None,
            "environment": None,
            "ai_integration": None,
            "procedural_generation": None,
            "benchmarking": None,
            "properties": None,
            "validation": None,
            "simplification": None,
            "provenance": None,
            "patch": None,
            "meta": None
        }
        
        # Process optional components
        for component in args[4:]:
            if isinstance(component, dict):
                for key, value in component.items():
                    if key in program_data:
                        program_data[key] = value
        
        return AtomForgeProgram(**program_data)
    
    def header(self, args):
        """Transform header block"""
        header_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "dsl_version":
                    header_data["dsl_version"] = value
                elif key == "content_schema_version":
                    header_data["content_schema_version"] = value
                elif key == "uuid":
                    header_data["uuid"] = value
                elif key == "title":
                    header_data["title"] = value
                elif key == "created":
                    header_data["created"] = value
                elif key == "modified":
                    header_data["modified"] = value
        
        return {"header": Header(**header_data)}
    
    def description(self, args):
        """Transform description"""
        return {"description": args[0]}
    
    def units(self, args):
        """Transform units block"""
        units_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "system":
                    units_data["system"] = value
                elif key == "length":
                    units_data["length"] = value
                elif key == "angle":
                    units_data["angle"] = value
                elif key == "disp":
                    units_data["disp"] = value
                elif key == "temp":
                    units_data["temp"] = value
                elif key == "pressure":
                    units_data["pressure"] = value
        
        return {"units": Units(**units_data)}
    
    def type_system(self, args):
        """Transform type system block"""
        type_system_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "structural_types" in arg:
                    type_system_data["structural_types"] = arg["structural_types"]
                elif "chemical_types" in arg:
                    type_system_data["chemical_types"] = arg["chemical_types"]
                elif "computational_types" in arg:
                    type_system_data["computational_types"] = arg["computational_types"]
                elif "auto_inference" in arg:
                    type_system_data["auto_inference"] = arg["auto_inference"]
        
        return {"type_system": TypeSystem(**type_system_data)}
    
    def structural_types(self, args):
        """Transform structural types"""
        structural_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "coordination_environment":
                    structural_data["coordination_environment"] = value
                elif key == "crystal_system":
                    structural_data["crystal_system"] = value
                elif key == "space_group_family":
                    structural_data["space_group_family"] = value
                elif key == "connectivity":
                    structural_data["connectivity"] = value
                elif key == "compatibility_rules":
                    structural_data["compatibility_rules"] = value
        
        return {"structural_types": StructuralTypes(**structural_data)}
    
    def lattice(self, args):
        """Transform lattice block"""
        lattice_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "description" in arg:
                    lattice_data["description"] = arg["description"]
                elif "bravais" in arg:
                    lattice_data["bravais"] = arg["bravais"]
                elif "vectors" in arg:
                    lattice_data["vectors"] = arg["vectors"]
        
        return {"lattice": Lattice(**lattice_data)}
    
    def bravais(self, args):
        """Transform Bravais lattice parameters"""
        bravais_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "type":
                    bravais_data["type"] = value
                elif key == "a":
                    bravais_data["a"] = value
                elif key == "b":
                    bravais_data["b"] = value
                elif key == "c":
                    bravais_data["c"] = value
                elif key == "alpha":
                    bravais_data["alpha"] = value
                elif key == "beta":
                    bravais_data["beta"] = value
                elif key == "gamma":
                    bravais_data["gamma"] = value
        
        return {"bravais": Bravais(**bravais_data)}
    
    def vectors(self, args):
        """Transform lattice vectors"""
        vectors = []
        for arg in args:
            if isinstance(arg, list) and len(arg) == 3:
                vectors.append(tuple(arg))
        
        return {"vectors": vectors}
    
    def symmetry(self, args):
        """Transform symmetry block"""
        symmetry_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "description" in arg:
                    symmetry_data["description"] = arg["description"]
                elif "space_group" in arg:
                    symmetry_data["space_group"] = arg["space_group"]
                elif "origin_choice" in arg:
                    symmetry_data["origin_choice"] = arg["origin_choice"]
                elif "magnetic_group" in arg:
                    symmetry_data["magnetic_group"] = arg["magnetic_group"]
                elif "superspace" in arg:
                    symmetry_data["superspace"] = arg["superspace"]
        
        return {"symmetry": Symmetry(**symmetry_data)}
    
    def basis(self, args):
        """Transform basis block"""
        basis_data = {"sites": []}
        for arg in args:
            if isinstance(arg, dict) and "site" in arg:
                basis_data["sites"].append(arg["site"])
            elif isinstance(arg, dict) and "description" in arg:
                basis_data["description"] = arg["description"]
        
        return {"basis": Basis(**basis_data)}
    
    def site(self, args):
        """Transform atomic site"""
        site_data = {}
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    if key in ["name", "description", "wyckoff", "position", "frame", 
                              "species", "moment", "constraint", "adp_iso", "adp_aniso", "label"]:
                        site_data[key] = value
        
        return {"site": Site(**site_data)}
    
    def species(self, args):
        """Transform species specification"""
        species_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "element":
                    species_data["element"] = value
                elif key == "occupancy":
                    species_data["occupancy"] = value
                elif key == "isotope":
                    species_data["isotope"] = value
                elif key == "charge":
                    species_data["charge"] = value
                elif key == "valence":
                    species_data["valence"] = value
        
        return {"species": Species(**species_data)}
    
    def emerging_materials(self, args):
        """Transform emerging materials block"""
        emerging_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "type" in arg:
                    emerging_data["type"] = arg["type"]
                elif "layer_stack" in arg:
                    emerging_data["layer_stack"] = arg["layer_stack"]
                elif "topology" in arg:
                    emerging_data["topology"] = arg["topology"]
                elif "metamaterial" in arg:
                    emerging_data["metamaterial"] = arg["metamaterial"]
                elif "quantum_state" in arg:
                    emerging_data["quantum_state"] = arg["quantum_state"]
        
        return {"emerging_materials": EmergingMaterials(**emerging_data)}
    
    def ai_integration(self, args):
        """Transform AI integration block"""
        ai_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "graph_representation" in arg:
                    ai_data["graph_representation"] = arg["graph_representation"]
                elif "generation_model" in arg:
                    ai_data["generation_model"] = arg["generation_model"]
                elif "active_learning" in arg:
                    ai_data["active_learning"] = arg["active_learning"]
                elif "multi_fidelity" in arg:
                    ai_data["multi_fidelity"] = arg["multi_fidelity"]
        
        return {"ai_integration": AIIntegration(**ai_data)}
    
    def procedural_generation(self, args):
        """Transform procedural generation block"""
        procedural_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "generator_type" in arg:
                    procedural_data["generator_type"] = arg["generator_type"]
                elif "template_generation" in arg:
                    procedural_data["template_generation"] = arg["template_generation"]
                elif "parameter_sweep" in arg:
                    procedural_data["parameter_sweep"] = arg["parameter_sweep"]
                elif "ml_guided" in arg:
                    procedural_data["ml_guided"] = arg["ml_guided"]
                elif "hybridization" in arg:
                    procedural_data["hybridization"] = arg["hybridization"]
        
        return {"procedural_generation": ProceduralGeneration(**procedural_data)}
    
    def benchmarking(self, args):
        """Transform benchmarking block"""
        benchmarking_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "benchmark_type" in arg:
                    benchmarking_data["benchmark_type"] = arg["benchmark_type"]
                elif "tasks" in arg:
                    benchmarking_data["tasks"] = arg["tasks"]
        
        return {"benchmarking": Benchmarking(**benchmarking_data)}
    
    def properties(self, args):
        """Transform properties block"""
        properties_data = {}
        for arg in args:
            if isinstance(arg, dict) and "property_entry" in arg:
                entry = arg["property_entry"]
                if "name" in entry and "value" in entry:
                    properties_data[entry["name"]] = entry["value"]
        
        return {"properties": Properties(properties=properties_data)}
    
    def validation(self, args):
        """Transform validation block"""
        validation_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "tolerance":
                    validation_data["tolerance"] = value
                elif key == "occupancy_clamp":
                    validation_data["occupancy_clamp"] = value
                elif key == "vector_unit_consistent":
                    validation_data["vector_unit_consistent"] = value
                elif key == "max_transform_depth":
                    validation_data["max_transform_depth"] = value
                elif key == "enforce_units":
                    validation_data["enforce_units"] = value
        
        return {"validation": Validation(**validation_data)}
    
    def simplification(self, args):
        """Transform simplification block"""
        simplification_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "complexity_level":
                    simplification_data["complexity_level"] = value
                elif key == "auto_complete":
                    simplification_data["auto_complete"] = value
                elif key == "suggest_defaults":
                    simplification_data["suggest_defaults"] = value
        
        return {"simplification": Simplification(**simplification_data)}
    
    def provenance(self, args):
        """Transform provenance block"""
        provenance_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "source":
                    provenance_data["source"] = value
                elif key == "method":
                    provenance_data["method"] = value
                elif key == "doi":
                    provenance_data["doi"] = value
                elif key == "url":
                    provenance_data["url"] = value
        
        return {"provenance": Provenance(**provenance_data)}
    
    def patch(self, args):
        """Transform patch block"""
        patch_data = {"operations": []}
        for arg in args:
            if isinstance(arg, dict) and "patch_op" in arg:
                patch_data["operations"].append(arg["patch_op"])
        
        return {"patch": Patch(**patch_data)}
    
    def patch_op(self, args):
        """Transform patch operation"""
        if len(args) >= 2:
            op_type = args[0]
            if op_type == "add":
                return {"patch_op": PatchOperation(type="add", target=args[1])}
            elif op_type == "remove":
                return {"patch_op": PatchOperation(type="remove", path=args[1])}
            elif op_type == "update":
                return {"patch_op": PatchOperation(type="update", path=args[1], value=args[2])}
        
        return {"patch_op": PatchOperation(type="unknown")}
    
    def meta(self, args):
        """Transform meta block"""
        meta_data = {}
        for arg in args:
            if isinstance(arg, dict) and "meta_entry" in arg:
                entry = arg["meta_entry"]
                if "name" in entry and "value" in entry:
                    meta_data[entry["name"]] = entry["value"]
        
        return {"meta": Meta(metadata=meta_data)}
    
    def meta_entry(self, args):
        """Transform meta entry"""
        if len(args) >= 2:
            return {"meta_entry": {"name": args[0], "value": args[1]}}
        return {"meta_entry": {}}
    
    # Terminal transformations
    def identifier(self, args):
        return str(args[0])
    
    def string(self, args):
        return str(args[0]).strip('"')
    
    def number(self, args):
        try:
            return float(args[0])
        except ValueError:
            return int(args[0])
    
    def int(self, args):
        return int(args[0])
    
    def float(self, args):
        return float(args[0])
    
    def bool(self, args):
        return str(args[0]).lower() == "true"
    
    def date(self, args):
        return datetime.strptime(str(args[0]), "%Y-%m-%d")
    
    def vec3(self, args):
        return [float(args[0]), float(args[1]), float(args[2])]
    
    def vec3f(self, args):
        return [float(args[0]), float(args[1]), float(args[2])]
    
    def vec3i(self, args):
        return [int(args[0]), int(args[1]), int(args[2])]
    
    def length(self, args):
        if len(args) == 1:
            return Length(value=float(args[0]), unit=None)
        else:
            return Length(value=float(args[0]), unit=str(args[1]))
    
    def angle(self, args):
        if len(args) == 1:
            return Angle(value=float(args[0]), unit=None)
        else:
            return Angle(value=float(args[0]), unit=str(args[1]))
    
    def temp(self, args):
        if len(args) == 1:
            return Temperature(value=float(args[0]), unit=None)
        else:
            return Temperature(value=float(args[0]), unit=str(args[1]))
    
    def pres(self, args):
        if len(args) == 1:
            return Pressure(value=float(args[0]), unit=None)
        else:
            return Pressure(value=float(args[0]), unit=str(args[1]))
    
    def disp(self, args):
        if len(args) == 1:
            return Displacement(value=float(args[0]), unit=None)
        else:
            return Displacement(value=float(args[0]), unit=str(args[1]))
    
    def frame(self, args):
        return str(args[0])
    
    def axis(self, args):
        return str(args[0])
    
    def id(self, args):
        return str(args[0])
    
    def qname(self, args):
        if len(args) == 1:
            return str(args[0])
        else:
            return f"{args[0]}:{args[1]}"
    
    def ident(self, args):
        return str(args[0])
    
    def feature_list(self, args):
        return [str(arg) for arg in args]
    
    def base64(self, args):
        return "".join(args)
    
    def base64_char(self, args):
        return str(args[0])
    
    def lat_type(self, args):
        return str(args[0])
    
    def space_group(self, args):
        return args[0]
    
    def wyckoff(self, args):
        return str(args[0])
    
    def element(self, args):
        return str(args[0])
    
    def len_unit(self, args):
        return str(args[0])
    
    def ang_unit(self, args):
        return str(args[0])
    
    def temp_unit(self, args):
        return str(args[0])
    
    def pres_unit(self, args):
        return str(args[0])
    
    def disp_unit(self, args):
        return str(args[0])





# Convenience functions
def parse_atomforge(content: str) -> AtomForgeProgram:
    """
    Convenience function to parse AtomForge DSL v2.1 content.
    
    Args:
        content: The AtomForge DSL v2.1 content to parse
        
    Returns:
        Parsed AtomForgeProgram object
    """
    parser = AtomForgeParser()
    return parser.parse_and_transform(content)


def parse_atomforge_file(file_path: str) -> AtomForgeProgram:
    """
    Convenience function to parse an AtomForge DSL v2.1 file.
    
    Args:
        file_path: Path to the AtomForge DSL v2.1 file
        
    Returns:
        Parsed AtomForgeProgram object
    """
    parser = AtomForgeParser()
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parser.parse_and_transform(content)


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
        result = parse_atomforge(example_content)
        print("Parsing successful!")
        print(f"Program identifier: {result.identifier}")
        print(f"DSL version: {result.header.dsl_version}")
        print(f"Title: {result.header.title}")
    except Exception as e:
        print(f"Parsing failed: {e}") 