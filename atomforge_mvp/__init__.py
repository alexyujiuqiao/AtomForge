#!/usr/bin/env python3
"""
AtomForge - Domain Specific Language for Materials Science

This package provides tools for working with AtomForge DSL, including:
- Parser and compiler for AtomForge DSL
- Converters for various materials databases
- Database connectors for Materials Project, COD, AFLOW, OQMD, ICSD
- Intelligent conversion with LLM reasoning
"""

# Import main components
from .src.parser.atomforge_parser import parse_atomforge_string
from .src.compiler.code_generator import CodeGenerator

# Import converters
from .src.converters import (
    convert_material,
    API_KEY,
    AtomForgeConverter,
    convert_with_atomforge,
    UnifiedMaterialsDatabase,
    DatabaseResult,
    MaterialsProjectConnector,
    CODConnector,
    AFLOWConnector,
    OQMDConnector,
    ICSDConnector,
    MaterialQuery,
    MaterialData
)

__version__ = "1.0.0"

__all__ = [
    # Core functionality
    'parse_atomforge_string',
    'CodeGenerator',
    
    # Converters
    'convert_material',
    'API_KEY',
    'AtomForgeConverter',
    'convert_with_atomforge',
    'UnifiedMaterialsDatabase',
    'DatabaseResult',
    'MaterialsProjectConnector',
    'CODConnector',
    'AFLOWConnector',
    'OQMDConnector',
    'ICSDConnector',
    'MaterialQuery',
    'MaterialData'
] 