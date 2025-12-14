#!/usr/bin/env python3

"""
Smoke test for AtomForge DSL v2.1 parser.

- Verifies that the grammar file can be loaded
- Initializes Lark via your DSLParser wrapper
- Parses a minimal but valid AtomForge program
"""

import os
import sys
from pathlib import Path

# --- Adjust these paths if your project structure is different ---

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent  # atomforge/src -> atomforge -> LLM-DSL

# Ensure core/ and atomforge/ are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Paths to grammar & parser
GRAMMAR_FILE = THIS_DIR / "atomforge.lark"  # Use full grammar

# Import your DSL infrastructure
try:
    from core.parser import DSLParser
except ImportError as e:
    print("Failed to import core.parser.DSLParser")
    print("Make sure your project structure is correct.")
    raise

# OPTIONAL: if you have AtomForgeParser class defined separately, you can also test it:
try:
    sys.path.insert(0, str(THIS_DIR))  # Add atomforge/src to path
    from atomforge_parser import AtomForgeParser
    HAVE_ATOMFORGE_PARSER = True
except ImportError:
    HAVE_ATOMFORGE_PARSER = False


# --- Minimal AtomForge program that should match your grammar ---

MINIMAL_PROGRAM = r'''
atom_spec "Si_diamond" {
  header {
    dsl_version = "2.1",
    title = "Silicon Diamond",
    created = 2025-05-08
  }

  units {
    system   = "crystallographic_default",
    length   = angstrom,
    angle    = degree,
    disp     = "angstrom^2",
    temp     = K,
    pressure = GPa
  }

  lattice {
    type  = cubic,
    a     = 5.431 angstrom,
    b     = 5.431 angstrom,
    c     = 5.431 angstrom,
    alpha = 90 degree,
    beta  = 90 degree,
    gamma = 90 degree
  }

  symmetry {
    space_group   = 227,
    origin_choice = 1
  }

  basis {
    site "Si1" {
      wyckoff  = "8a",
      position = (0.0, 0.0, 0.0),
      frame    = fractional,
      species  = ({ element = "Si", occupancy = 1.0 })
    }
  }
}
'''


def test_with_dslparser():
    print("== Testing via core.DSLParser ==")
    print(f"Using grammar file: {GRAMMAR_FILE}")

    if not GRAMMAR_FILE.exists():
        raise FileNotFoundError(f"Grammar file not found: {GRAMMAR_FILE}")

    parser = DSLParser(str(GRAMMAR_FILE), start_symbol="start")
    tree = parser.parse(MINIMAL_PROGRAM)
    print("Parse tree successfully created.")
    # If you want to see the structure:
    # print(tree.pretty())
    print("OK: DSLParser + grammar load and parse succeeded.\n")


def test_with_atomforgeparser():
    if not HAVE_ATOMFORGE_PARSER:
        print("AtomForgeParser not available; skipping that part.\n")
        return

    print("== Testing via AtomForgeParser (parse + transform) ==")
    parser = AtomForgeParser(grammar_file=str(GRAMMAR_FILE), start_symbol="start")
    program = parser.parse_and_transform(MINIMAL_PROGRAM)
    print("Program object successfully created.")
    try:
        print(f"Identifier: {program.identifier}")
        if program.header:
            print(f"DSL version: {program.header.dsl_version}")
            print(f"Title     : {program.header.title}")
    except Exception as e:
        print("Program object exists but attribute access raised an error:")
        print(repr(e))
    print("OK: AtomForgeParser end-to-end parse+transform succeeded.\n")


if __name__ == "__main__":
    print("=== AtomForge DSL v2.1 Parser Smoke Test ===\n")

    # 1) Direct test through core DSLParser
    test_with_dslparser()

    # 2) Optional: full AtomForgeParser test if available
    test_with_atomforgeparser()

    print("All tests done.")

