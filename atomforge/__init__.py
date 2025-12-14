"""
AtomForge DSL v2.1 - Production Standard

A revolutionary domain-specific language for comprehensive materials science specification,
featuring AI-native integration, procedural generation, benchmarking, and a revolutionary
patching system.

This is the main package for AtomForge DSL v2.1, which extends the core DSL framework
and provides advanced materials science capabilities.
"""

__version__ = "2.1.0"
__author__ = "AtomForge Development Team"
__description__ = "AtomForge DSL v2.1 - Revolutionary Domain-Specific Language for Materials Science"

# Import the main v2.1 components
try:
    from .src.atomforge_parser import (
        AtomForgeParser,
        AtomForgeTransformer,
        parse_atomforge,
        parse_atomforge_file
    )
    
    from .src.atomforge_compiler import (
        AtomForgeCompiler,
        compile_atomforge,
        compile_atomforge_file
    )
    
    from .src.atomforge_ir import (
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
    
    # Import Phase 1 components (Interop + Variant Pinning)
    from .src.phase1_interop import (
        VariantPolicy,
        VariantInfo,
        VariantSelectionResult,
        DatabaseMatchResult,
        CrystalDatabaseMatcher,
        from_cif,
        from_poscar,
        match_database,
        select_variant,
        create_variant_card,
        pin_variant
    )
    
    from .src.phase1_ui import (
        VariantCard,
        VariantSelector,
        SimpleVariantUI,
        create_variant_selector,
        run_variant_selection
    )
    
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
        
        # Phase 1: Interop + Variant Pinning
        "VariantPolicy",
        "VariantInfo", 
        "VariantSelectionResult",
        "DatabaseMatchResult",
        "CrystalDatabaseMatcher",
        "from_cif",
        "from_poscar",
        "match_database",
        "select_variant",
        "create_variant_card",
        "pin_variant",
        "VariantCard",
        "VariantSelector",
        "SimpleVariantUI",
        "create_variant_selector",
        "run_variant_selection"
    ]
    
except ImportError as e:
    print(f"Warning: Could not import AtomForge v2.1 components: {e}")
    __all__ = []

# Legacy v1 components are now in the atomforge_mvp folder
# To access v1 components, import from atomforge_mvp 