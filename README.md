# AtomForge: Domain-Specific Language for Materials Science

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AtomForge is a comprehensive Domain-Specific Language (DSL) framework designed for modeling and manipulating inorganic crystalline materials. It provides a unified, modular approach to materials science workflows, from basic crystal structures to complex defect modeling and AI-integrated materials discovery.

## Key Features

- **Universal Materials Modeling**: Support for all inorganic repetitive structures (crystals, surfaces, interfaces, defects, amorphous regions, nanostructures)
- **Revolutionary Patching System**: Path-based addressing and version control for structure modifications
- **Multi-format I/O**: Native support for CIF, VASP/POSCAR, JSON, and custom formats
- **AI Integration**: Built-in support for AI/ML pipelines and materials property prediction
- **Advanced Type System**: Structural, chemical, and computational type validation
- **Comprehensive Validation**: Multi-level validation with configurable tolerance and error handling
- **Provenance Tracking**: Full reproducibility and metadata management

## Project Structure

```
AtomForge/
├── atomforge/                    # Core v2.1 implementation
│   ├── src/
│   │   ├── atomforge_compiler.py # Main compiler with multi-format output
│   │   ├── atomforge_parser.py   # Lark-based parser
│   │   ├── atomforge_ir.py       # Intermediate representation
│   │   └── crystal_v1_1.py       # Crystal v1.1 compatibility layer
│   ├── atomforge_v2.1.ebnf       # EBNF grammar specification
│   └── atomforge_crystal_schema.json
├── atomforge_materials/          # Example materials
│   ├── minimal_silicon.atomforge # Minimal silicon example
│   ├── silicon_v2.1.atomforge    # Full-featured silicon
│   └── graphene_2d_v2.1.atomforge # 2D graphene structure
├── atomforge_mvp/               # Legacy v1.0 implementation
├── core/                        # Shared DSL framework
├── data/                        # Test materials and outputs
└── docs/                        # Documentation
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/AtomForge.git
cd AtomForge

# Install dependencies
pip install -r atomforge_mvp/requirements.txt
pip install pymatgen  # For materials science formats
```

### Basic Usage

```python
from atomforge.src.atomforge_compiler import AtomForgeCompiler

# Create compiler with JSON output
compiler = AtomForgeCompiler(output_format="json")

# Minimal silicon example
silicon_code = '''
atom_spec "silicon_diamond" {
  header { 
    dsl_version = "2.1", 
    title = "Silicon Diamond", 
    created = 2025-05-08 
  }
  lattice { 
    type = cubic, 
    a = 5.431 
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

# Compile to JSON
result = compiler.compile(silicon_code)
print(result)
```

### Advanced Example

```python
# Compile to CIF format
cif_compiler = AtomForgeCompiler(output_format="cif")
cif_output = cif_compiler.compile(silicon_code)

# Compile to VASP/POSCAR format
vasp_compiler = AtomForgeCompiler(output_format="vasp")
vasp_output = vasp_compiler.compile(silicon_code)
```

## Language Features

### Core Syntax

AtomForge DSL v2.1 provides a rich syntax for materials specification:

```atomforge
atom_spec "material_name" {
  header { 
    dsl_version = "2.1",
    title = "Material Title",
    created = 2025-05-08
  }
  
  units {
    system = "crystallography_default",
    length = angstrom,
    angle = degree
  }
  
  lattice {
    type = cubic,
    a = 5.431 angstrom,
    b = 5.431 angstrom,
    c = 5.431 angstrom,
    alpha = 90.0 degree,
    beta = 90.0 degree,
    gamma = 90.0 degree
  }
  
  symmetry {
    space_group = 227,
    origin_choice = 2
  }
  
  basis {
    site "Si1" {
      wyckoff = "8a",
      position = (0.125, 0.125, 0.125),
      frame = fractional,
      species = ({ element = "Si", occupancy = 1.0 })
    }
  }
  
  properties {
    band_gap = 1.12 eV,
    density = 2.329 "g/cm3"
  }
}
```

### Advanced Features

- **Type System**: Structural, chemical, and computational type validation
- **Defect Modeling**: Comprehensive defect specification and manipulation
- **AI Integration**: Built-in support for ML property prediction
- **Procedural Generation**: Automated structure generation workflows
- **Benchmarking**: Multi-modal understanding and validation
- **Patching**: Revolutionary path-based structure modification

## Development

### Running Tests

```bash
# Test the complete materials pipeline
python test_online_materials_pipeline.py

# Test CIF/POSCAR import functionality
python test_cif_poscar_import.py
```

### Adding New Materials

Create `.atomforge` files in the `atomforge_materials/` directory:

```bash
# Example: Create a new material
cp atomforge_materials/minimal_silicon.atomforge atomforge_materials/my_material.atomforge
# Edit the file with your material specifications
```

## Documentation

- **Grammar Specification**: `atomforge/atomforge_v2.1.ebnf`
- **Schema Definition**: `atomforge/atomforge_crystal_schema.json`
- **Legacy Documentation**: `atomforge_mvp/AtomForge.pdf`
- **Converter Guide**: `atomforge_mvp/src/converters/AtomForge_Converter.md`

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on the core DSL framework for extensible domain-specific languages
- Integrates with pymatgen for materials science format support
- Supports materials from various databases (Materials Project, ICSD, COD)

## Related Projects

- **Core DSL Framework**: Modular framework for building domain-specific languages
- **Materials Project**: Database integration for materials discovery
- **pymatgen**: Python Materials Genomics library

---

**AtomForge v2.1** - Production Standard for comprehensive materials specification and manipulation.
