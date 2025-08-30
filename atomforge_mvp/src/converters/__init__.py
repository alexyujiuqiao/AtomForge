#!/usr/bin/env python3
"""
AtomForge Converters Package

This package contains all the converter modules for AtomForge DSL:
- converter.py: Basic Materials Project converter
- atomforge_converter.py: Main intelligent converter with LLM reasoning
- database_connectors.py: Database connector framework

Located in src/converters/ for better organization.
"""

from .converter import convert_material, API_KEY
from .atomforge_converter import (
    AtomForgeConverter,
    convert_with_atomforge,
    MaterialQuery,
    MaterialData
)
from .database_connectors import (
    UnifiedMaterialsDatabase,
    DatabaseResult,
    MaterialsProjectConnector,
    CODConnector,
    AFLOWConnector,
    OQMDConnector,
    ICSDConnector
)

__all__ = [
    'convert_material',
    'API_KEY',
    'AtomForgeConverter',
    'convert_with_atomforge',
    'MaterialQuery',
    'MaterialData',
    'UnifiedMaterialsDatabase',
    'DatabaseResult',
    'MaterialsProjectConnector',
    'CODConnector',
    'AFLOWConnector',
    'OQMDConnector',
    'ICSDConnector'
] 