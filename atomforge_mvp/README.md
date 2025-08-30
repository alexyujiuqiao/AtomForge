# AtomForge MVP (v1.0) - Legacy Components

This package contains the AtomForge MVP v1.0 implementation and related components.

## Contents

### Core Files
- `atomforge_v1.0.ebnf` - EBNF grammar specification for v1.0
- `AtomForge.pdf` - Original AtomForge documentation
- `LLAMP_FRAMEWORK_PSEUDOCODE.md` - LLaMP framework pseudocode
- `requirements.txt` - Dependencies for v1.0
- `__init__.py` - Original package initialization

### Source Code (`src/`)
- `parser/` - Parser implementation for v1.0
  - `atomforge.lark` - Lark grammar for v1.0
  - `atomforge_parser.py` - Main parser implementation
  - `atomforge_ir.py` - Intermediate representation
  - `parser.py` - Base parser class
  - `atomforge_patch.py` - Patching system for v1.0

- `agents/` - AI agent implementation
  - `atomforge_agent.py` - Intelligent auto-converter agent
  - `__init__.py` - Agent package initialization

- `converters/` - File format converters
  - `atomforge_converter.py` - Main converter implementation
  - `converter.py` - Base converter class
  - `input2atomforge.py` - Input to AtomForge conversion
  - `database_connectors.py` - Database connectivity
  - `AtomForge_Converter.md` - Converter documentation
  - `__init__.py` - Converter package initialization

- `compiler/` - Compiler implementation
  - `code_generator.py` - Code generation logic
  - `__init__.py` - Compiler package initialization

### UI Components (`UI/`)
- `3d_pipeline.py` - 3D visualization pipeline

## Usage

To use the v1.0 components:

```python
# Import v1.0 components
from atomforge_mvp.src.parser.atomforge_parser import AtomForgeParser
from atomforge_mvp.src.agents.atomforge_agent import AtomForgeAgent
from atomforge_mvp.src.converters.atomforge_converter import AtomForgeConverter

# Use v1.0 parser
parser = AtomForgeParser()
result = parser.parse_and_transform(content)

# Use v1.0 agent
agent = AtomForgeAgent()
converted = agent.convert(input_format, output_format)

# Use v1.0 converter
converter = AtomForgeConverter()
output = converter.convert(input_file, output_format)
```

## Migration to v2.1

The v2.1 implementation provides significant improvements:

- **Core Framework Integration**: Proper inheritance from the core DSL framework
- **Advanced Features**: AI integration, emerging materials, procedural generation
- **Revolutionary Patching**: Path-based addressing and version control
- **Comprehensive Benchmarking**: Multi-modal understanding and property prediction
- **Enhanced Type System**: Structural, chemical, and computational type validation

To migrate from v1.0 to v2.1:

1. Update import statements to use v2.1 components
2. Update grammar syntax to v2.1 format
3. Leverage new advanced features as needed
4. Use the new validation and type system

## Legacy Support

The v1.0 components are maintained for:
- **Backward Compatibility**: Existing v1.0 code continues to work
- **Reference Implementation**: Study the evolution of the language
- **Migration Testing**: Compare v1.0 and v2.1 outputs
- **Documentation**: Historical context and design decisions

## File Structure

```
atomforge_mvp/
├── README.md                    # This file
├── atomforge_v1.0.ebnf         # v1.0 grammar
├── AtomForge.pdf               # v1.0 documentation
├── LLAMP_FRAMEWORK_PSEUDOCODE.md # LLaMP framework
├── requirements.txt            # v1.0 dependencies
├── __init__.py                 # v1.0 package init
└── src/
    ├── parser/
    │   ├── atomforge.lark      # v1.0 Lark grammar
    │   ├── atomforge_parser.py # v1.0 parser
    │   ├── atomforge_ir.py     # v1.0 IR
    │   ├── parser.py           # v1.0 base parser
    │   └── atomforge_patch.py  # v1.0 patching
    ├── agents/
    │   ├── atomforge_agent.py  # v1.0 AI agent
    │   └── __init__.py
    ├── converters/
    │   ├── atomforge_converter.py # v1.0 converter
    │   ├── converter.py        # v1.0 base converter
    │   ├── input2atomforge.py  # v1.0 input conversion
    │   ├── database_connectors.py # v1.0 DB connectors
    │   ├── AtomForge_Converter.md # v1.0 converter docs
    │   └── __init__.py
    ├── compiler/
    │   ├── code_generator.py   # v1.0 code generation
    │   └── __init__.py
    └── UI/
        └── 3d_pipeline.py      # v1.0 3D visualization
```

## Version History

- **v1.0**: Original MVP implementation with basic DSL features
- **v2.1**: Production standard with advanced AI integration and comprehensive features

For the latest features and improvements, use AtomForge DSL v2.1 from the main package. 