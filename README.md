# DSL Framework for Domain-Specific Modeling

This project defines a unified, modular framework for building and executing Domain-Specific Languages (DSLs) across various domains — including but not limited to statistics, medicine, music, and genetic editing.

Each DSL follows a shared lifecycle:  
**DSL code - Parsing - AST - Validation - Compilation/Execution**

## 🌐 Project Structure Overview

```
project_root/
 │
 ├── core/                     		  # Shared base classes for all DSLs
 │   ├── dataclass.py             # Abstract base class for a full DSLProgram
 │   ├── parser.py             	# Base parser and AST transformer using Lark
 │   └── compiler.py             # Abstract base class for DSL compilers
 │
 ├── statdsl/                  		# Concrete implementation for a statistics DSL
 │   ├── grammar.md          # Human-readable grammar documentation
 │   ├── statdsl.ebnf.txt       # EBNF grammar specification for Lark parser
 │   ├── statprogram.py      # Concrete subclass of DSLProgram for statistics
 │   ├── statparser.py          # Custom DSLTransformer for statistics
 │   └── statcompiler.py      # Compiler that outputs Stan code
 │
 └── other DSL ...
```

---

## 📦 `core/` Module Guide

The `core` folder defines abstract base classes shared across all DSLs.

### `dataclass.py`

- Defines the abstract `DSLProgram` class.
- All domain-specific programs must subclass this and implement a `validate()` method to check internal consistency.

### `parser.py`

- Provides two base classes:
  - `DSLParser`: Loads an EBNF grammar and parses DSL code into a Lark parse tree.
  - `DSLTransformer`: Transforms the parse tree into a structured AST or data classes.
- Each DSL should subclass `DSLTransformer` to define how grammar rules map to internal structures.

### `compiler.py`

- Defines the abstract class `DSLCompiler`:
  - Orchestrates parsing - transformation - validation - compilation.
  - Requires a concrete implementation of `_compile(program)` to generate output (e.g., code, SQL, executable).

---

## 🧪 `statdsl/`: An Example DSL for Statistical Modeling

This module implements a DSL for writing statistical models that can be compiled to Stan code and executed.

- `grammar.md`: Documentation explaining the DSL syntax and semantics for human readers.
- `statdsl.ebnf.txt`: Machine-readable Lark grammar defining the formal syntax.
- `statprogram.py`: Subclass of `DSLProgram` defining all blocks/entities in the statistics DSL.
- `statparser.py`: Subclass of `DSLTransformer` that builds a `StatProgram` from the parse tree.
- `statcompiler.py`: Subclass of `DSLCompiler` that compiles the `StatProgram` to Stan code.

---

## 🧑‍💻 How to Implement Your Own DSL

To add a new DSL module (e.g., for music, medicine, etc.), follow these steps:

1. **Create a New Module Folder**

   For example, `meddsl/`, `musicdsl/`, `genedsl/`.

2. **Define the DSL Grammar**

   - Write an EBNF grammar in `yourdsl.ebnf.txt`.
   - Optionally, provide a human-readable guide in `grammar.md`.

3. **Implement Required Components**

   Inside your DSL folder:

   - `yourprogram.py`:
     - Subclass `DSLProgram` and define your DSL-specific blocks and structure.
     - Implement a `validate()` method.
   
   - `yourparser.py`:
     - Subclass `DSLTransformer` to map parsed grammar nodes to your AST/data structures.
   
   - `yourcompiler.py`:
     - Subclass `DSLCompiler` and implement `_compile()` to output your target format (e.g., SQL, Python code, executable logic).

4. **Integrate and Test**

   - Instantiate your parser, transformer, and compiler.
   - Run your DSL code through the pipeline.
   - Validate, compile, and optionally execute the output.

---

## Minimal Example Entry Point

```python
from core import DSLParser
from statdsl import StatModelTrans, StatModelCompiler

grammar_file = "statdsl/statdsl_v1_0.ebnf.txt"
start_symbol = "stat_model_spec"
parser = DSLParser(grammar_file=grammar_file, start_symbol=start_symbol)
trans = StatModelTrans()
compiler = StatModelCompiler(parser, trans)

dsl_code = """
model linear_regression {
  data {
    real x;
    real y;
  }
  parameters {
    real beta;
  }
  model {
    y ~ normal(beta * x, 1);
  }
}
"""

compiled_output = compiler.compile(dsl_code)
print(compiled_output)
```

------

## 💡 Notes

- Each DSL is fully modular and independent, but shares a unified interface.
- The `validate()` method in each `DSLProgram` subclass ensures syntactic and semantic correctness before code generation.
- The framework is extensible: you can plug in interpreters, optimizers, simulators, etc
