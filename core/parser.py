from lark import Lark
from lark import Transformer

"""
This file defines the core parsing logic for all DSLs.

`DSLParser` is responsible for loading an EBNF grammar and converting raw DSL code
into a parse tree using the Lark parsing library.

`DSLTransformer` is a base class for transforming the Lark parse tree into a custom
abstract syntax tree (AST) or structured Python dataclasses. Each DSL should subclass
this transformer to define how grammar rules map to program elements.
"""

class DSLParser():
    def __init__(self, grammar_file: str, start_symbol: str):
        """
        Initialize the DSL parser by loading the EBNF grammar 
        and setting up the Lark parser.
        
        :param grammar_file: Path to the EBNF grammar file
        :param start_symbol: The start symbol for grammar parsing
        """
        # Read the grammar definition from file
        with open(grammar_file, 'r') as f:
            grammar = f.read()
        
        # Initialize the Lark parser with the specified grammar and start symbol
        # Use 'earley' parser for grammars with many optional rules to avoid parse table explosion
        # Use 'dynamic' lexer for complex grammars (only works with earley)
        try:
            self.parser = Lark(grammar, start=start_symbol, parser='earley', lexer='dynamic', 
                             maybe_placeholders=False, propagate_positions=False, cache=False)
        except Exception:
            # Fallback to lalr with basic lexer if earley fails
            self.parser = Lark(grammar, start=start_symbol, parser='lalr', lexer='basic', 
                             maybe_placeholders=False, propagate_positions=False, cache=True)

    def parse(self, code: str):
        """
        Parse the given DSL code and return the Lark parse tree.
        
        :param code: DSL code as a string
        :return: The resulting parse tree (AST)
        """
        tree = self.parser.parse(code)
        return tree

class DSLTransformer(Transformer):
    """
    A generic base transformer class based on Lark's Transformer,
    used to convert a Lark parse tree into an AST or other formats.
    Subclasses should implement specific node transformation logic.
    """
    def __init__(self):
        super().__init__()

    def transform(self, tree):
        """
        Recursively traverse the tree and transform each node into the target format.
        Override this method if custom transformation behavior is needed.
        """
        return super().transform(tree)
    
    def generic(self, items):
        """
        Default transformation method for all nodes not explicitly handled.
        Subclasses can override this method for specific node types.
        
        :param items: List of child elements for the node
        :return: A generic dictionary structure representing the node
        """
        return {"type": str(type(self)), "data": items[0], "children": items[1:]}
