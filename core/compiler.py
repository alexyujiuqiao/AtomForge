# compiler/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from pathlib import Path
import os

"""
This file defines the base class for all DSL compilers.

A `DSLCompiler` takes DSL source code through a complete compilation pipeline:
1. Parsing (code - parse tree)
2. Transformation (parse tree - program AST)
3. Validation (check internal consistency)
4. Code Generation (program AST - output code or executable logic)

Subclasses must implement the `_compile` method to produce domain-specific outputs.
"""

class DSLCompiler(ABC):
    """
    Base class for compilers that turn DSL source code into target code.
    Parses - transforms - validates - generates code.
    """

    def __init__(self, parser: Any, transformer: Any):
        """
        Initialize the compiler with a parser and a transformer.
        
        :param parser: Instance of a DSLParser
        :param transformer: Instance of a DSLTransformer
        """
        self.parser = parser
        self.transformer = transformer
        self.program = None  # Holds the internal AST representation of the program

    def to_dict(self, code) -> Dict[str, Any]:
        """
        Returns a dictionary representing all intermediate stages of compilation.
        Useful for debugging, visualization, or program introspection.
        
        :param code: DSL source code as a string
        :return: Dictionary with source code, parse tree, AST, and compiler output
        """
        parse_tree = self.parser.parse(code)
        self.program = self.transformer.transform(parse_tree)
        output = self._compile(self.program)

        return {
            "source_code": code,
            "parse_tree": parse_tree,
            "ast": self.program,
            "compiler_output": output
        }

    @abstractmethod
    def _compile(self, program: Any) -> str:
        """
        Subclasses must implement this method to define how the validated
        program is converted into the final output (e.g., Python code, SQL, etc.).
        
        :param program: The validated AST or program representation
        :return: Final compiled output as a string
        """
        pass

    def compile(self, code: Union[Path, str]) -> str:
        """
        Runs the full compilation pipeline: parse - transform - validate - compile.
        
        :param code: DSL source code as a string or a file path
        :return: Final compiled output as a string
        """
        if isinstance(code, Path) or os.path.isfile(code):
            code = Path(code).read_text()

        parse_tree = self.parser.parse(code)
        self.program = self.transformer.transform(parse_tree)

        # Ensure the program implements and passes validation
        if hasattr(self.program, "validate"):
            self.program.validate()
        else:
            raise NotImplementedError("Program does not implement `validate()`.")

        return self._compile(self.program)
