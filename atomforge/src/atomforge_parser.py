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
core_path = os.path.join(os.path.dirname(__file__), '..', '..', 'core')
if core_path not in sys.path:
    sys.path.append(core_path)

try:
    from parser import DSLParser, DSLTransformer
except ImportError:
    # Fallback: create simple base classes if core module not available
    class DSLParser:
        def __init__(self, grammar_file, start_symbol):
            self.grammar_file = grammar_file
            self.start_symbol = start_symbol
        def parse(self, code):
            pass
        def parse_and_transform(self, code):
            pass
    
    class DSLTransformer:
        def __init__(self):
            pass
        def transform(self, tree):
            pass

try:
    from lark import Lark, Transformer, v_args, Token
    from lark.exceptions import LarkError, UnexpectedInput
except ImportError:
    raise ImportError("Lark parser library is required. Install with: pip install lark")

from atomforge_ir import (
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

# Grammar file path - using full atomforge.lark grammar
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
        # New grammar structure: atom_spec identifier "{" header description? core_blocks optional_blocks "}"
        identifier = args[0]

        header_obj = None
        description_val = None
        core_blocks_data = {}
        optional_blocks_data = {}

        # Process args in order
        for arg in args[1:]:
            if isinstance(arg, dict):
                if "header" in arg:
                    header_obj = arg["header"]
                elif "description" in arg:
                    description_val = arg["description"]
                elif "core_blocks" in arg:
                    core_blocks_data = arg["core_blocks"]
                elif "optional_blocks" in arg:
                    optional_blocks_data = arg["optional_blocks"]
                else:
                    # Direct component
                    for key, value in arg.items():
                        if key in ["units", "type_system", "lattice", "symmetry", "basis"]:
                            core_blocks_data[key] = value
                        else:
                            optional_blocks_data[key] = value

        program_data = {
            "identifier": identifier,
            "header": header_obj,
            "description": description_val,
            "units": core_blocks_data.get("units"),
            "type_system": core_blocks_data.get("type_system"),
            "lattice": core_blocks_data.get("lattice"),
            "symmetry": core_blocks_data.get("symmetry"),
            "basis": core_blocks_data.get("basis"),
            "emerging_materials": optional_blocks_data.get("emerging_materials"),
            "defects": optional_blocks_data.get("defects"),
            "tile": optional_blocks_data.get("tile"),
            "bonds": optional_blocks_data.get("bonds"),
            "elastic": optional_blocks_data.get("elastic"),
            "phonon": optional_blocks_data.get("phonon"),
            "density": optional_blocks_data.get("density"),
            "environment": optional_blocks_data.get("environment"),
            "ai_integration": optional_blocks_data.get("ai_integration"),
            "procedural_generation": optional_blocks_data.get("procedural_generation"),
            "benchmarking": optional_blocks_data.get("benchmarking"),
            "properties": optional_blocks_data.get("properties"),
            "validation": optional_blocks_data.get("validation"),
            "simplification": optional_blocks_data.get("simplification"),
            "provenance": optional_blocks_data.get("provenance"),
            "patch": optional_blocks_data.get("patch"),
            "meta": optional_blocks_data.get("meta")
        }
        
        return AtomForgeProgram(**program_data)
    
    def core_blocks(self, args):
        """Transform core blocks"""
        core_data = {}
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    if key in ["units", "type_system", "lattice", "symmetry", "basis"]:
                        core_data[key] = value
        return {"core_blocks": core_data}
    
    def optional_blocks(self, args):
        """Transform optional blocks"""
        optional_data = {}
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    optional_data[key] = value
        return {"optional_blocks": optional_data}
    
    def header(self, args):
        """Transform header block"""
        # Grammar order:
        # dsl_version,
        # [content_schema_version],
        # [uuid],
        # title,
        # created,
        # [modified]
        string_values = [a for a in args if isinstance(a, str)]

        if not string_values:
            raise ValueError("header requires at least dsl_version, title, created")

        dsl_version = string_values[0]
        title = string_values[-2] if len(string_values) >= 3 else None
        created = string_values[-1] if len(string_values) >= 2 else None

        content_schema_version = None
        uuid_val = None
        # Remaining middle values (if any) map to content_schema_version and uuid in order
        middle = string_values[1:-2] if len(string_values) > 3 else []
        if middle:
            content_schema_version = middle[0]
        if len(middle) >= 2:
            uuid_val = middle[1]

        header_data = {
            "dsl_version": dsl_version,
            "title": title,
            "created": created,
            "content_schema_version": content_schema_version,
            "uuid": uuid_val,
        }
        if len(string_values) >= 6:
            header_data["modified"] = string_values[-1]

        return {"header": Header(**header_data)}
    
    def description(self, args):
        """Transform description"""
        return {"description": args[0]}
    
    def units(self, args):
        """Transform units block (minimal core: use fixed defaults)"""
        # Core grammar treats unit fields as simple strings; for minimal programs,
        # we map them directly into the Units dataclass, falling back to defaults
        # if anything is missing.
        units_data = {
            "system": "crystallographic_default",
            "length": "angstrom",
            "angle": "degree",
            "disp": "angstrom^2",
            "temp": "K",
            "pressure": "GPa",
        }
        # If the parse tree carried explicit values, they will be in args as strings.
        # We walk them positionally in the expected order.
        if args and len(args) >= 6:
            system, length, angle, disp, temp, pressure = args[:6]
            units_data.update(
                {
                    "system": system,
                    "length": length,
                    "angle": angle,
                    "disp": disp,
                    "temp": temp,
                    "pressure": pressure,
                }
            )
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
    
    def chemical_types(self, args):
        """Transform chemical types"""
        chemical_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "element_category":
                    chemical_data["element_category"] = value
                elif key == "oxidation_states":
                    chemical_data["oxidation_states"] = value
                elif key == "coordination_preference":
                    chemical_data["coordination_preference"] = value
                elif key == "validation_rules":
                    chemical_data["validation_rules"] = value
        
        return {"chemical_types": ChemicalTypes(**chemical_data)}
    
    def computational_types(self, args):
        """Transform computational types"""
        computational_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "calculation_method":
                    computational_data["calculation_method"] = value
                elif key == "basis_requirements":
                    computational_data["basis_requirements"] = value
                elif key == "k_point_density":
                    computational_data["k_point_density"] = value
                elif key == "resource_requirements":
                    computational_data["resource_requirements"] = value
        
        return {"computational_types": ComputationalTypes(**computational_data)}
    
    def auto_inference(self, args):
        """Transform auto inference"""
        inference_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "enabled":
                    inference_data["enabled"] = bool(value)
                elif key == "inference_methods":
                    inference_data["inference_methods"] = value
                elif key == "validation_strictness":
                    inference_data["validation_strictness"] = value
                elif key == "error_handling":
                    inference_data["error_handling"] = value
        
        return {"auto_inference": AutoInference(**inference_data)}
    
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
    
    def structural_type(self, args):
        """Transform individual structural type entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def chemical_type(self, args):
        """Transform individual chemical type entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def computational_type(self, args):
        """Transform individual computational type entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def resource_requirement(self, args):
        """Transform resource requirement entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def chemical_property(self, args):
        """Transform chemical property entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def type_system_property(self, args):
        """Transform type system property entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
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
        """Transform Bravais lattice parameters (minimal core)."""
        # The core grammar emits the sequence:
        #   type, a, b, c, alpha, beta, gamma
        # possibly interleaved with commas and field names.
        # We extract numeric / string values positionally and
        # fall back to safe defaults if anything is missing.
        scalar_values = [v for v in args if isinstance(v, (str, int, float))]

        bravais_data = {
            "type": "triclinic",
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }

        if len(scalar_values) >= 1:
            bravais_data["type"] = scalar_values[0]
        if len(scalar_values) >= 2:
            bravais_data["a"] = float(scalar_values[1])
        if len(scalar_values) >= 3:
            bravais_data["b"] = float(scalar_values[2])
        if len(scalar_values) >= 4:
            bravais_data["c"] = float(scalar_values[3])
        if len(scalar_values) >= 5:
            bravais_data["alpha"] = float(scalar_values[4])
        if len(scalar_values) >= 6:
            bravais_data["beta"] = float(scalar_values[5])
        if len(scalar_values) >= 7:
            bravais_data["gamma"] = float(scalar_values[6])

        return {"bravais": Bravais(**bravais_data)}
    
    def vectors(self, args):
        """Transform lattice vectors"""
        vectors = []
        for arg in args:
            if isinstance(arg, list) and len(arg) == 3:
                vectors.append(tuple(arg))
        
        return {"vectors": vectors}
    
    def symmetry(self, args):
        """Transform symmetry block (minimal core)."""
        # Core grammar emits:
        #   symmetry { [description] space_group = space_group, origin_choice = int ... }
        # Our minimal generator currently does:
        #   symmetry {
        #     space_group = <int or string>,
        #     origin_choice = 1
        #   }
        symmetry_data = {}

        # First capture any nested description dicts (from the full grammar)
        for arg in args:
            if isinstance(arg, dict) and "description" in arg:
                symmetry_data["description"] = arg["description"]

        # Then extract scalar values positionally for space_group and origin_choice
        scalar_values = [v for v in args if isinstance(v, (str, int, float))]
        if len(scalar_values) >= 1:
            symmetry_data["space_group"] = scalar_values[0]
        if len(scalar_values) >= 2:
            symmetry_data["origin_choice"] = int(scalar_values[1])

        # Optional fields are ignored in the minimal core symmetry
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
        """Transform atomic site (minimal core-compatible)."""
        # Grammar (core):
        #   site: "site" identifier "{" description?
        #                  "wyckoff" "=" wyckoff ","
        #                  "position" "=" vec3f ","
        #                  "frame" "=" frame ","
        #                  "species" "=" "(" species ("," species)* ")"
        #                  ... "}"
        name = None
        wyckoff = None
        position = None
        frame_val = None
        species_list = []
        description = None
        moment = None
        constraint = None
        adp_iso = None
        adp_aniso = None
        label = None

        for arg in args:
            if isinstance(arg, dict):
                # description/species/etc. come in as dicts from their own rules
                if "description" in arg:
                    description = arg["description"]
                if "species" in arg:
                    val = arg["species"]
                    if isinstance(val, list):
                        species_list.extend(val)
                    else:
                        species_list.append(val)
                if "moment" in arg:
                    moment = arg["moment"]
                if "constraint" in arg:
                    constraint = arg["constraint"]
                if "adp_iso" in arg:
                    adp_iso = arg["adp_iso"]
                if "adp_aniso" in arg:
                    adp_aniso = arg["adp_aniso"]
                if "label" in arg:
                    label = arg["label"]
            else:
                # Scalars: name, wyckoff, position, frame
                if isinstance(arg, str):
                    if name is None:
                        name = arg
                    elif arg in ("fractional", "cartesian") and frame_val is None:
                        frame_val = arg
                    elif wyckoff is None:
                        wyckoff = arg
                elif isinstance(arg, (list, tuple)) and len(arg) == 3 and position is None:
                    position = tuple(arg)

        site_data = {
            "name": name or "site1",
            "wyckoff": wyckoff or "1a",
            "position": position or (0.0, 0.0, 0.0),
            "frame": frame_val or "fractional",
            "species": species_list or [Species(element="X", occupancy=1.0)],
        }
        if description is not None:
            site_data["description"] = description
        if moment is not None:
            site_data["moment"] = moment
        if constraint is not None:
            site_data["constraint"] = constraint
        if adp_iso is not None:
            site_data["adp_iso"] = adp_iso
        if adp_aniso is not None:
            site_data["adp_aniso"] = adp_aniso
        if label is not None:
            site_data["label"] = label

        return {"site": Site(**site_data)}
    
    def species(self, args):
        """Transform species specification (minimal core)."""
        # Core grammar: species { element = string, occupancy = number, ... }
        # Lark passes the scalar values as args; we map them positionally.
        scalar_values = [v for v in args if isinstance(v, (str, int, float))]

        species_data = {}
        if len(scalar_values) >= 1:
            species_data["element"] = scalar_values[0]
        if len(scalar_values) >= 2:
            species_data["occupancy"] = float(scalar_values[1])
        if len(scalar_values) >= 3:
            species_data["isotope"] = int(scalar_values[2])
        if len(scalar_values) >= 4:
            species_data["charge"] = float(scalar_values[3])
        if len(scalar_values) >= 5:
            species_data["valence"] = float(scalar_values[4])

        return {"species": Species(**species_data)}
    
    def defects(self, args):
        """Transform defects block"""
        defects_list = []
        for arg in args:
            if isinstance(arg, dict) and "defect_entry" in arg:
                defects_list.append(arg["defect_entry"])
        
        return {"defects": Defects(defects=defects_list)}
    
    def defect_entry(self, args):
        """Transform defect entry"""
        defect_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "site_ref":
                    defect_data["site_ref"] = value
                elif key == "type":
                    defect_data["type"] = value
                elif key == "prob":
                    defect_data["prob"] = float(value)
                elif key == "species":
                    defect_data["species"] = value
        
        return {"defect_entry": DefectEntry(**defect_data)}
    
    def tile(self, args):
        """Transform tile block"""
        tile_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "repeat" in arg:
                    tile_data["repeat"] = arg["repeat"]
                elif "origin_shift" in arg:
                    tile_data["origin_shift"] = arg["origin_shift"]
                elif "transforms" in arg:
                    tile_data["transforms"] = arg["transforms"]
            elif isinstance(arg, (list, tuple)) and len(arg) == 3:
                if "repeat" not in tile_data:
                    tile_data["repeat"] = tuple(arg)
        
        return {"tile": Tile(**tile_data)}
    
    def transforms(self, args):
        """Transform transforms list"""
        transform_list = []
        for arg in args:
            if isinstance(arg, dict):
                if "transform_seq" in arg:
                    transform_list.append(arg["transform_seq"])
                elif "transform_op" in arg:
                    transform_list.append(arg["transform_op"])
            else:
                transform_list.append(str(arg))
        
        return {"transforms": transform_list}
    
    def transform_seq(self, args):
        """Transform transform sequence"""
        if len(args) == 1:
            return {"transform_seq": args[0]}
        return {"transform_seq": args}
    
    def transform_op(self, args):
        """Transform transform operation"""
        if len(args) >= 1:
            op_type = args[0]
            if isinstance(op_type, dict):
                return {"transform_op": op_type}
            return {"transform_op": {"type": str(op_type), "args": args[1:]}}
        return {"transform_op": {}}
    
    def mirror(self, args):
        """Transform mirror operation"""
        axis = args[0] if args else "x"
        return {"mirror": {"type": "mirror", "axis": axis}}
    
    def rotate(self, args):
        """Transform rotate operation"""
        if len(args) >= 2:
            axis = args[0]
            angle = args[1]
            return {"rotate": {"type": "rotate", "axis": axis, "angle": angle}}
        return {"rotate": {}}
    
    def translate(self, args):
        """Transform translate operation"""
        if len(args) >= 2:
            vec = args[0]
            frame = args[1] if len(args) > 1 else "cartesian"
            return {"translate": {"type": "translate", "vector": vec, "frame": frame}}
        return {"translate": {}}
    
    def mat4(self, args):
        """Transform matrix4 operation"""
        if len(args) >= 4:
            return {"mat4": {"type": "matrix4", "matrix": args[:4]}}
        return {"mat4": {}}
    
    def bonds(self, args):
        """Transform bonds block"""
        bonds_list = []
        for arg in args:
            if isinstance(arg, dict) and "bond" in arg:
                bonds_list.append(arg["bond"])
        
        return {"bonds": Bonds(bonds=bonds_list)}
    
    def bond(self, args):
        """Transform bond specification"""
        if len(args) >= 3:
            site1 = args[0]
            site2 = args[1]
            length = args[2]
            return {"bond": Bond(site1=site1, site2=site2, length=length)}
        return {"bond": Bond(site1="", site2="", length=Length(value=0.0))}
    
    def elastic(self, args):
        """Transform elastic block"""
        c_ijkl = []
        for arg in args:
            if isinstance(arg, (int, float)):
                c_ijkl.append(float(arg))
            elif isinstance(arg, list):
                c_ijkl.extend([float(x) for x in arg])
        
        return {"elastic": Elastic(c_ijkl=c_ijkl)}
    
    def phonon(self, args):
        """Transform phonon block"""
        phonon_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "q_grid" in arg:
                    phonon_data["q_grid"] = arg["q_grid"]
                elif "frequencies" in arg:
                    phonon_data["frequencies"] = arg["frequencies"]
            elif isinstance(arg, (list, tuple)) and len(arg) == 3:
                if "q_grid" not in phonon_data:
                    phonon_data["q_grid"] = tuple(arg)
            elif isinstance(arg, list):
                phonon_data["frequencies"] = [float(x) for x in arg]
        
        return {"phonon": Phonon(**phonon_data)}
    
    def density(self, args):
        """Transform density block"""
        density_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "grid" in arg:
                    density_data["grid"] = arg["grid"]
                elif "format" in arg:
                    density_data["format"] = arg["format"]
                elif "data" in arg:
                    density_data["data"] = arg["data"]
                elif "data_file" in arg:
                    density_data["data_file"] = arg["data_file"]
                elif "description" in arg:
                    density_data["description"] = arg["description"]
            elif isinstance(arg, (list, tuple)) and len(arg) == 3:
                if "grid" not in density_data:
                    density_data["grid"] = tuple(arg)
            elif isinstance(arg, str):
                if "format" not in density_data:
                    density_data["format"] = arg
                elif "data" not in density_data:
                    density_data["data"] = arg
        
        return {"density": Density(**density_data)}
    
    def density_data(self, args):
        """Transform density data"""
        if len(args) >= 1:
            return {"data": str(args[0])}
        return {"data": ""}
    
    def density_file(self, args):
        """Transform density file"""
        if len(args) >= 1:
            return {"data_file": str(args[0])}
        return {"data_file": ""}
    
    def environment(self, args):
        """Transform environment block"""
        env_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "temperature" in arg:
                    env_data["temperature"] = arg["temperature"]
                elif "pressure" in arg:
                    env_data["pressure"] = arg["pressure"]
                elif "e_field" in arg:
                    env_data["e_field"] = arg["e_field"]
                elif "e_grad" in arg:
                    env_data["e_grad"] = arg["e_grad"]
                elif "b_field" in arg:
                    env_data["b_field"] = arg["b_field"]
        
        return {"environment": Environment(**env_data)}
    
    def mat3(self, args):
        """Transform 3x3 matrix"""
        if len(args) >= 3:
            return [args[0], args[1], args[2]]
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
    def field_grad_unit(self, args):
        """Transform field gradient unit"""
        if args:
            return str(args[0])
        return "V/m2"
    
    def superspace(self, args):
        """Transform superspace block"""
        superspace_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "k_vectors" in arg:
                    superspace_data["k_vectors"] = arg["k_vectors"]
                elif "t_phase" in arg:
                    superspace_data["t_phase"] = arg["t_phase"]
            elif isinstance(arg, list):
                superspace_data["k_vectors"] = arg
            elif isinstance(arg, (int, float)):
                superspace_data["t_phase"] = float(arg)
        
        return {"superspace": Superspace(**superspace_data)}
    
    def adp(self, args):
        """Transform anisotropic displacement parameters"""
        if len(args) == 6:
            # Array format: [U11, U22, U33, U23, U13, U12]
            return {
                "U11": float(args[0]),
                "U22": float(args[1]),
                "U33": float(args[2]),
                "U23": float(args[3]),
                "U13": float(args[4]),
                "U12": float(args[5])
            }
        elif isinstance(args[0], dict):
            # Dictionary format
            return args[0]
        return {"U11": 0.0, "U22": 0.0, "U33": 0.0, "U23": 0.0, "U13": 0.0, "U12": 0.0}
    
    def moment(self, args):
        """Transform magnetic moment"""
        if len(args) >= 3:
            vec = [float(args[0]), float(args[1]), float(args[2])]
            frame = args[3] if len(args) > 3 else "cartesian"
            unit = args[4] if len(args) > 4 else "µB"
            return {"moment": {"vector": vec, "frame": frame, "unit": unit}}
        return {"moment": None}
    
    def constraint(self, args):
        """Transform constraint"""
        if args:
            return {"constraint": str(args[0])}
        return {"constraint": None}
    
    def label(self, args):
        """Transform label"""
        if args:
            return {"label": str(args[0])}
        return {"label": None}
    
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
    
    def layer_stack(self, args):
        """Transform layer stack"""
        layer_stack_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "layers" in arg:
                    layer_stack_data["layers"] = arg["layers"]
                elif "interlayer_distance" in arg:
                    layer_stack_data["interlayer_distance"] = arg["interlayer_distance"]
                elif "coupling_strength" in arg:
                    layer_stack_data["coupling_strength"] = arg["coupling_strength"]
        
        return {"layer_stack": LayerStack(**layer_stack_data)}
    
    def layer_spec(self, args):
        """Transform layer specification"""
        layer_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "material":
                    layer_data["material"] = value
                elif key == "twist_angle":
                    layer_data["twist_angle"] = value
                else:
                    if "properties" not in layer_data:
                        layer_data["properties"] = {}
                    layer_data["properties"][key] = value
        
        return {"layer_spec": LayerSpec(**layer_data)}
    
    def topology(self, args):
        """Transform topology"""
        topology_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "classification":
                    topology_data["classification"] = value
                elif key == "z2_invariant":
                    topology_data["z2_invariant"] = value
                elif key == "surface_states":
                    topology_data["surface_states"] = value
                elif key == "bulk_gap":
                    topology_data["bulk_gap"] = float(value)
        
        return {"topology": Topology(**topology_data)}
    
    def metamaterial(self, args):
        """Transform metamaterial"""
        metamaterial_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "metamaterial_type":
                    metamaterial_data["metamaterial_type"] = value
                elif key == "unit_cell_period":
                    metamaterial_data["unit_cell_period"] = value
                elif key == "effective_properties":
                    metamaterial_data["effective_properties"] = value
                else:
                    if "properties" not in metamaterial_data:
                        metamaterial_data["properties"] = {}
                    metamaterial_data["properties"][key] = value
        
        return {"metamaterial": Metamaterial(**metamaterial_data)}
    
    def quantum_state(self, args):
        """Transform quantum state"""
        quantum_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "quantum_state":
                    quantum_data["quantum_state"] = value
                elif key == "frustration":
                    quantum_data["frustration"] = value
                elif key == "entanglement":
                    quantum_data["entanglement"] = value
                else:
                    if "properties" not in quantum_data:
                        quantum_data["properties"] = {}
                    quantum_data["properties"][key] = value
        
        return {"quantum_state": QuantumState(**quantum_data)}
    
    def layer_property(self, args):
        """Transform layer property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def topology_property(self, args):
        """Transform topology property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def metamaterial_property(self, args):
        """Transform metamaterial property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def quantum_property(self, args):
        """Transform quantum property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def effective_property(self, args):
        """Transform effective property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def frustration_spec(self, args):
        """Transform frustration specification"""
        spec = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                spec[args[i]] = args[i + 1]
        return spec
    
    def entanglement_spec(self, args):
        """Transform entanglement specification"""
        spec = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                spec[args[i]] = args[i + 1]
        return spec
    
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
    
    def ai_integration_property(self, args):
        """Transform AI integration property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def graph_representation(self, args):
        """Transform graph representation"""
        graph_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "node_features":
                    graph_data["node_features"] = value
                elif key == "edge_features":
                    graph_data["edge_features"] = value
                elif key == "global_features":
                    graph_data["global_features"] = value
        
        return {"graph_representation": GraphRepresentation(**graph_data)}
    
    def generation_model(self, args):
        """Transform generation model"""
        model_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "type":
                    model_data["type"] = value
                elif key == "latent_dim":
                    model_data["latent_dim"] = int(value)
                elif key == "constraints":
                    model_data["constraints"] = value
                elif key == "sampling_temperature":
                    model_data["sampling_temperature"] = float(value)
        
        return {"generation_model": GenerationModel(**model_data)}
    
    def active_learning(self, args):
        """Transform active learning"""
        al_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "acquisition_function":
                    al_data["acquisition_function"] = value
                elif key == "surrogate_model":
                    al_data["surrogate_model"] = value
                elif key == "exploration_weight":
                    al_data["exploration_weight"] = float(value)
                elif key == "batch_size":
                    al_data["batch_size"] = int(value)
        
        return {"active_learning": ActiveLearning(**al_data)}
    
    def multi_fidelity(self, args):
        """Transform multi-fidelity"""
        mf_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "low_fidelity":
                    mf_data["low_fidelity"] = value
                elif key == "high_fidelity":
                    mf_data["high_fidelity"] = value
                elif key == "correlation_model":
                    mf_data["correlation_model"] = value
                elif key == "cost_ratio":
                    mf_data["cost_ratio"] = float(value)
        
        return {"multi_fidelity": MultiFidelity(**mf_data)}
    
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
    
    def template_generation(self, args):
        """Transform template generation"""
        template_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "template":
                    template_data["template"] = value
                elif key == "parameter_space":
                    template_data["parameter_space"] = value
                elif key == "constraints":
                    template_data["constraints"] = value
        
        return {"template_generation": TemplateGeneration(**template_data)}
    
    def parameter_sweep(self, args):
        """Transform parameter sweep"""
        sweep_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "base_structure":
                    sweep_data["base_structure"] = value
                elif key == "sweep_parameters":
                    sweep_data["sweep_parameters"] = value
                elif key == "generation_mode":
                    sweep_data["generation_mode"] = value
        
        return {"parameter_sweep": ParameterSweep(**sweep_data)}
    
    def ml_guided(self, args):
        """Transform ML-guided generation"""
        ml_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "target_properties":
                    ml_data["target_properties"] = value
                elif key == "generation_method":
                    ml_data["generation_method"] = value
                elif key == "population_size":
                    ml_data["population_size"] = int(value)
                elif key == "generations":
                    ml_data["generations"] = int(value)
        
        return {"ml_guided": MLGuided(**ml_data)}
    
    def hybridization(self, args):
        """Transform hybridization"""
        hybrid_data = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                key = args[i]
                value = args[i + 1]
                if key == "parent_selection":
                    hybrid_data["parent_selection"] = value
                elif key == "crossover_operations":
                    hybrid_data["crossover_operations"] = value
                elif key == "mutation_rate":
                    hybrid_data["mutation_rate"] = float(value)
                elif key == "fitness_function":
                    hybrid_data["fitness_function"] = value
        
        return {"hybridization": Hybridization(**hybrid_data)}
    
    def parameter_space(self, args):
        """Transform parameter space entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def sweep_parameter(self, args):
        """Transform sweep parameter entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def property_target(self, args):
        """Transform property target entry"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def parent_selection(self, args):
        """Transform parent selection"""
        selection = {}
        for i in range(0, len(args), 2):
            if i + 1 < len(args):
                selection[args[i]] = args[i + 1]
        return selection
    
    def generation_property(self, args):
        """Transform generation property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
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
    
    def benchmark_task(self, args):
        """Transform benchmark task"""
        task_data = {}
        for arg in args:
            if isinstance(arg, dict):
                if "input_modality" in arg:
                    task_data["input_modality"] = arg["input_modality"]
                elif "input_data" in arg:
                    task_data["input_data"] = arg["input_data"]
                elif "target_output" in arg:
                    task_data["target_output"] = arg["target_output"]
                elif "target_properties" in arg:
                    task_data["target_properties"] = arg["target_properties"]
                elif "difficulty_level" in arg:
                    task_data["difficulty_level"] = arg["difficulty_level"]
                elif "evaluation_metrics" in arg:
                    task_data["evaluation_metrics"] = arg["evaluation_metrics"]
                elif "constraints" in arg:
                    task_data["constraints"] = arg["constraints"]
        
        return {"benchmark_task": BenchmarkTask(**task_data)}
    
    def task_property(self, args):
        """Transform task property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def benchmark_property(self, args):
        """Transform benchmark property"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def property_spec(self, args):
        """Transform property specification"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def properties(self, args):
        """Transform properties block"""
        properties_data = {}
        for arg in args:
            if isinstance(arg, dict) and "property_entry" in arg:
                entry = arg["property_entry"]
                if "name" in entry and "value" in entry:
                    properties_data[entry["name"]] = entry["value"]
        
        return {"properties": Properties(properties=properties_data)}
    
    def property_entry(self, args):
        """Transform property entry"""
        if len(args) >= 2:
            return {"property_entry": {"name": args[0], "value": args[1]}}
        return {"property_entry": {}}
    
    def property_value(self, args):
        """Transform property value"""
        if len(args) == 1:
            return args[0]
        return args
    
    def dimless_value(self, args):
        """Transform dimensionless value"""
        if len(args) == 1:
            return args[0]
        return args
    
    def dimful_value(self, args):
        """Transform dimensionful value"""
        if len(args) == 1:
            return args[0]
        return args
    
    def validation(self, args):
        """Transform validation block (minimal core)."""
        # Core grammar (and generator) emit:
        #   tolerance = number,
        #   occupancy_clamp = bool,
        #   vector_unit_consistent = bool,
        #   max_transform_depth = int,
        #   enforce_units = bool
        scalar_values = [v for v in args if isinstance(v, (int, float, bool))]

        validation_data = {
            "tolerance": 1e-6,
            "occupancy_clamp": True,
            "vector_unit_consistent": True,
            "max_transform_depth": 64,
            "enforce_units": True,
        }

        if len(scalar_values) >= 1:
            validation_data["tolerance"] = float(scalar_values[0])
        if len(scalar_values) >= 2:
            validation_data["occupancy_clamp"] = bool(scalar_values[1])
        if len(scalar_values) >= 3:
            validation_data["vector_unit_consistent"] = bool(scalar_values[2])
        if len(scalar_values) >= 4:
            validation_data["max_transform_depth"] = int(scalar_values[3])
        if len(scalar_values) >= 5:
            validation_data["enforce_units"] = bool(scalar_values[4])

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
    
    def simplification_option(self, args):
        """Transform simplification option"""
        if len(args) >= 2:
            return {args[0]: args[1]}
        return {}
    
    def provenance(self, args):
        """Transform provenance block (minimal core)."""
        # Collect values by position (source, method, doi, url) and extensions
        string_values = [v for v in args if isinstance(v, str)]

        provenance_data = {
            "source": string_values[0] if len(string_values) >= 1 else "unknown",
            "method": string_values[1] if len(string_values) >= 2 else "method_unknown",
            "doi": string_values[2] if len(string_values) >= 3 else "",
            "url": string_values[3] if len(string_values) >= 4 else None,
            "extensions": {},
        }

        # Collect extensions
        for arg in args:
            if isinstance(arg, dict) and "provenance_extension" in arg:
                ext = arg["provenance_extension"]
                if isinstance(ext, dict):
                    provenance_data["extensions"].update(ext)

        return {"provenance": Provenance(**provenance_data)}
    
    def provenance_extension(self, args):
        """Transform provenance extension"""
        if len(args) >= 2:
            return {"provenance_extension": {args[0]: args[1]}}
        return {"provenance_extension": {}}
    
    def patch(self, args):
        """Transform patch block"""
        patch_data = {"operations": []}
        for arg in args:
            if isinstance(arg, dict) and "patch_op" in arg:
                patch_data["operations"].append(arg["patch_op"])
        
        return {"patch": Patch(**patch_data)}
    
    def patch_op(self, args):
        """Transform patch operation"""
        if not args:
            return {"patch_op": PatchOperation(type="unknown")}

        # Pass-through if child rules already produced a patch_op dict
        if len(args) == 1 and isinstance(args[0], dict) and "patch_op" in args[0]:
            return args[0]

        op_type = None
        site_obj = None
        path = None
        value = None

        for arg in args:
            if isinstance(arg, str) and arg in ("add", "add_site", "remove", "update"):
                op_type = arg
                continue
            if isinstance(arg, dict) and "site" in arg:
                site_obj = arg["site"]
                continue
            if path is None and isinstance(arg, str):
                path = arg
                continue
            if value is None:
                value = arg

        # Default op_type for add-site if not explicitly captured
        if op_type is None and site_obj is not None:
            op_type = "add"

        if op_type in ("add", "add_site") and site_obj is not None:
            return {"patch_op": PatchOperation(type="add", site=site_obj)}

        if op_type == "add" and path is not None:
            return {"patch_op": PatchOperation(type="add", path=path, value=value)}

        if op_type == "remove" and path is not None:
            return {"patch_op": PatchOperation(type="remove", path=path)}

        if op_type == "update" and path is not None:
            return {"patch_op": PatchOperation(type="update", path=path, value=value)}

        return {"patch_op": PatchOperation(type="unknown")}

    # Explicit handlers for patch sub-rules to ensure correct mapping
    def patch_add_site(self, args):
        """add site ..."""
        for arg in args:
            if isinstance(arg, dict) and "site" in arg:
                return {"patch_op": PatchOperation(type="add", site=arg["site"])}
        return {"patch_op": PatchOperation(type="unknown")}

    def patch_add_value(self, args):
        """add path = value"""
        if len(args) >= 2:
            return {"patch_op": PatchOperation(type="add", path=args[0], value=args[1])}
        return {"patch_op": PatchOperation(type="unknown")}

    def patch_remove(self, args):
        """remove path"""
        if args:
            return {"patch_op": PatchOperation(type="remove", path=args[0])}
        return {"patch_op": PatchOperation(type="unknown")}

    def patch_update(self, args):
        """update path = value"""
        if len(args) >= 2:
            return {"patch_op": PatchOperation(type="update", path=args[0], value=args[1])}
        return {"patch_op": PatchOperation(type="unknown")}
    
    def path(self, args):
        """Transform path"""
        if args:
            return ".".join([str(arg) for arg in args])
        return ""
    
    def path_segment(self, args):
        """Transform path segment"""
        if len(args) >= 1:
            segment = str(args[0])
            if len(args) >= 2:
                segment += f"[{args[1]}]"
            return segment
        return ""
    
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
        # Minimal core: safely handle empty arg list by defaulting to False.
        if not args:
            return False
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
        # Minimal core: lat_type is already a terminal like "cubic", "tetragonal", etc.
        # Some core parses may call this with an empty args list; fall back to a sentinel.
        if args:
            return str(args[0])
        return "triclinic"
    
    def space_group(self, args):
        return args[0]
    
    def wyckoff(self, args):
        return str(args[0])
    
    def element(self, args):
        return str(args[0])

    def frame(self, args):
        # Frame is either "fractional" or "cartesian"; default to "fractional"
        if args:
            return str(args[0])
        return "None"
    
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