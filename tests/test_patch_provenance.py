import os
import sys
from pathlib import Path

# Ensure atomforge/src is on sys.path for direct module imports in tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "atomforge" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atomforge_parser import AtomForgeParser


def parse_program(text: str):
    parser = AtomForgeParser()
    return parser.parse_and_transform(text)


def test_provenance_extensions():
    program = parse_program(
        '''
atom_spec "prov_test" {
  header { dsl_version = "2.1", title = "t", created = "2025-01-01" }
  lattice { type = cubic, a = 5.0, b = 5.0, c = 5.0, alpha = 90.0, beta = 90.0, gamma = 90.0 }
  symmetry { space_group = 221, origin_choice = 1 }
  basis { site X1 { wyckoff = "1a", position = (0,0,0), frame = fractional, species = ({ element = "X", occupancy = 1.0 }) } }
  provenance { source = "ICSD", method = "DFT", doi = "10.0/xyz", url = "http://example.com",
               software = "VASP 6.3.0", prediction_confidence = 0.92, license = "CC BY 4.0" }
}
'''
    )
    assert program.provenance is not None
    ext = program.provenance.extensions
    assert ext["software"] == "VASP 6.3.0"
    assert ext["prediction_confidence"] == 0.92
    assert ext["license"] == "CC BY 4.0"


def test_patch_add_update_remove():
    program = parse_program(
        r'''
atom_spec "patch_test" {
  header { dsl_version = "2.1", title = "t", created = "2025-01-01" }
  lattice { type = cubic, a = 5.0, b = 5.0, c = 5.0, alpha = 90.0, beta = 90.0, gamma = 90.0 }
  symmetry { space_group = 221, origin_choice = 1 }
  basis { site O1 { wyckoff = "1a", position = (0,0,0), frame = fractional, species = ({ element = "O", occupancy = 1.0 }) } }
  provenance { source = "ICSD", method = "DFT", doi = "10.0/xyz" }
  patch {
    add site "Ti1" { wyckoff = "1a", position = (0.5,0.5,0.5), frame = fractional, species = ({ element = "Ti", occupancy = 1.0 }) },
    update lattice.a = 5.540,
    remove basis.O1,
    update provenance.method = "PBE+U relaxation"
  }
}
'''
    )
    patch = program.patch
    assert patch is not None
    ops = patch.operations
    assert len(ops) == 4
    assert ops[0].type == "add" and ops[0].site is not None and ops[0].site.name == "Ti1"
    assert ops[1].type == "update" and ops[1].path == "lattice.a" and ops[1].value == 5.540
    assert ops[2].type == "remove" and ops[2].path == "basis.O1"
    assert ops[3].type == "update" and ops[3].path == "provenance.method" and ops[3].value == "PBE+U relaxation"


def test_patch_indexed_path():
    program = parse_program(
        r'''
atom_spec "path_index_test" {
  header { dsl_version = "2.1", title = "t", created = "2025-01-01" }
  lattice { type = cubic, a = 5.0, b = 5.0, c = 5.0, alpha = 90.0, beta = 90.0, gamma = 90.0 }
  symmetry { space_group = 221, origin_choice = 1 }
  basis { site Al1 { wyckoff = "1a", position = (0,0,0), frame = fractional, species = ({ element = "Al", occupancy = 1.0 }) } }
  patch { update basis.Al1.species[0].occupancy = 0.95 }
}
'''
    )
    ops = program.patch.operations
    assert len(ops) == 1
    assert ops[0].type == "update"
    assert ops[0].path == "basis.Al1.species[0].occupancy"
    assert ops[0].value == 0.95

