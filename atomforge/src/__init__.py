"""
AtomForge DSL v2.1 Source Package

This package contains the source code for AtomForge DSL v2.1,
including parsers, IR definitions, compilers, and related components.
"""

from .atomforge_parser import (
    AtomForgeParser,
    AtomForgeTransformer,
    parse_atomforge,
    parse_atomforge_file
)

from .atomforge_compiler import (
    AtomForgeCompiler,
    compile_atomforge,
    compile_atomforge_file
)

from .atomforge_ir import (
    AtomForgeProgram,
    Header,
    Units,
    TypeSystem,
    Lattice,
    Symmetry,
    Basis,
    Site,
    Species,
    EmergingMaterials,
    AIIntegration,
    ProceduralGeneration,
    Benchmarking,
    Patch,
    Properties,
    Validation,
    Simplification,
    Provenance,
    Meta
)

from .crystal_calc import (
    CalculationTarget,
    CalcSettings,
    CalcInput,
    PrepReport,
    prepare_calc,
    estimate_kmesh,
)

__version__ = "2.1.0"

__all__ = [
    # Parser
    "AtomForgeParser",
    "AtomForgeTransformer", 
    "parse_atomforge",
    "parse_atomforge_file",
    
    # Compiler
    "AtomForgeCompiler",
    "compile_atomforge",
    "compile_atomforge_file",
    
    # Core data structures
    "AtomForgeProgram",
    "Header",
    "Units",
    "TypeSystem",
    "Lattice",
    "Symmetry",
    "Basis",
    "Site",
    "Species",
    
    # Advanced features
    "EmergingMaterials",
    "AIIntegration",
    "ProceduralGeneration",
    "Benchmarking",
    "Patch",
    "Properties",
    "Validation",
    "Simplification",
    "Provenance",
    "Meta",

    # Calculation prep
    "CalculationTarget",
    "CalcSettings",
    "CalcInput",
    "PrepReport",
    "prepare_calc",
    "estimate_kmesh",
] 