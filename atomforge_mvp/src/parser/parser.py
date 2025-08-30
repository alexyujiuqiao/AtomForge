from lark import Lark, Transformer
from pathlib import Path
import os
from typing import Any, Dict, List

from .atomforge_ir import (
    AtomForgeFile, LanguageVersionDecl, AtomSpec, Header, Units,
    Lattice, Symmetry, Site, Species, ConvergenceCriteria,
    TargetProperties, PropertyValidation, Provenance
)

"""
This file defines the AtomForge-specific parsing logic following the core DSL pattern.

`AtomForgeParser` is responsible for loading the AtomForge EBNF grammar and converting 
raw AtomForge DSL code into a parse tree using the Lark parsing library.

`AtomForgeTransformer` transforms the Lark parse tree into AtomForge-specific 
data structures (AtomForgeFile, AtomSpec, etc.).
"""

class AtomForgeParser:
    """
    AtomForge-specific DSL parser that loads the AtomForge grammar and parses DSL code.
    Follows the pattern from core/parser.py.
    """
    
    def __init__(self, grammar_file: str = None, start_symbol: str = "start"):
        """
        Initialize the AtomForge parser by loading the EBNF grammar 
        and setting up the Lark parser.
        
        :param grammar_file: Path to the AtomForge EBNF grammar file (optional)
        :param start_symbol: The start symbol for grammar parsing
        """
        if grammar_file is None:
            # Default to the grammar file in the same directory
            grammar_file = os.path.join(os.path.dirname(__file__), "atomforge.lark")
        
        # Read the grammar definition from file
        with open(grammar_file, 'r', encoding='utf-8') as f:
            grammar = f.read()
        
        # Initialize the Lark parser with the specified grammar and start symbol
        self.parser = Lark(grammar, start=start_symbol, parser='lalr', propagate_positions=False)

    def parse(self, code: str):
        """
        Parse the given AtomForge DSL code and return the Lark parse tree.
        
        :param code: AtomForge DSL code as a string
        :return: The resulting parse tree (AST)
        """
        tree = self.parser.parse(code)
        return tree

    def parse_and_transform(self, code: str):
        """
        Parse the DSL code and transform it into AtomForge data structures.
        
        :param code: AtomForge DSL code as a string
        :return: AtomForgeFile object
        """
        tree = self.parse(code)
        transformer = AtomForgeTransformer()
        return transformer.transform(tree)


class AtomForgeTransformer(Transformer):
    """
    AtomForge-specific transformer that converts Lark parse trees into 
    AtomForge data structures. Follows the pattern from core/parser.py.
    """
    
    def __init__(self):
        super().__init__()

    def start(self, *args):
        """Transform the start rule"""
        # args = [atomforge_file]
        return args[0]

    def atomforge_file(self, language_version=None, atom_spec=None):
        """Transform atomforge_file rule"""
        return AtomForgeFile(language_version=language_version, spec=atom_spec)

    def language_version_decl(self, ver):
        """Transform language_version_decl rule"""
        return LanguageVersionDecl(version=ver[1:-1])  # Remove quotes

    def atom_spec(self, name, header, description, units, lattice, symmetry, basis, property_validation, provenance):
        """Transform atom_spec rule"""
        return AtomSpec(
            name=str(name),
            header=header,
            description=description,
            units=units,
            lattice=lattice,
            symmetry=symmetry,
            basis=basis,
            property_validation=property_validation,
            provenance=provenance
        )

    def header(self, dsl_version, title, created, uuid=None):
        """Transform header rule"""
        return Header(
            dsl_version=dsl_version[1:-1],  # Remove quotes
            title=title[1:-1],  # Remove quotes
            created=created,
            uuid=uuid[1:-1] if uuid else None  # Remove quotes if present
        )

    def description(self, desc):
        """Transform description rule"""
        return desc[1:-1] if desc else None  # Remove quotes

    def units(self, system, length, angle):
        """Transform units rule"""
        return Units(
            system=system[1:-1],  # Remove quotes
            length=length,
            angle=angle
        )

    def lattice(self, lattice_type, a, b, c, alpha, beta, gamma):
        """Transform lattice rule"""
        return Lattice(
            type=lattice_type,
            a=float(a),
            b=float(b),
            c=float(c),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma)
        )

    def symmetry(self, space_group, origin_choice=None):
        """Transform symmetry rule"""
        return Symmetry(
            space_group=space_group[1:-1],  # Remove quotes
            origin_choice=int(origin_choice) if origin_choice else None
        )

    def basis(self, sites):
        """Transform basis rule"""
        return sites if isinstance(sites, list) else [sites]

    def site(self, identifier, wyckoff, position, frame, species, adp_iso=None, label=None):
        """Transform site rule"""
        return Site(
            identifier=str(identifier),
            wyckoff=wyckoff[1:-1],  # Remove quotes
            position=position,
            frame=frame,
            species=species,
            adp_iso=float(adp_iso) if adp_iso else None,
            label=label[1:-1] if label else None  # Remove quotes if present
        )

    def species_list(self, species_items):
        """Transform species_list rule"""
        return species_items if isinstance(species_items, list) else [species_items]

    def species(self, element, occupancy, charge=None):
        """Transform species rule"""
        return Species(
            element=element[1:-1],  # Remove quotes
            occupancy=float(occupancy),
            charge=float(charge) if charge else None
        )

    def property_validation(self, computational_backend, convergence_criteria=None, target_properties=None):
        """Transform property_validation rule"""
        return PropertyValidation(
            computational_backend=computational_backend,
            convergence_criteria=convergence_criteria,
            target_properties=target_properties
        )

    def computational_backend(self, functional, energy_cutoff, k_point_density):
        """Transform computational_backend rule"""
        return {
            "functional": functional[1:-1],  # Remove quotes
            "energy_cutoff": float(energy_cutoff),
            "k_point_density": float(k_point_density)
        }

    def convergence_criteria(self, energy_tolerance=None, force_tolerance=None, stress_tolerance=None):
        """Transform convergence_criteria rule"""
        return ConvergenceCriteria(
            energy_tolerance=float(energy_tolerance) if energy_tolerance else None,
            force_tolerance=float(force_tolerance) if force_tolerance else None,
            stress_tolerance=float(stress_tolerance) if stress_tolerance else None
        )

    def target_properties(self, formation_energy=None, band_gap=None, elastic_constants=None):
        """Transform target_properties rule"""
        return TargetProperties(
            formation_energy=formation_energy,
            band_gap=band_gap,
            elastic_constants=elastic_constants
        )

    def provenance(self, source, method=None, doi=None):
        """Transform provenance rule"""
        return Provenance(
            source=source[1:-1],  # Remove quotes
            method=method[1:-1] if method else None,  # Remove quotes if present
            doi=doi[1:-1] if doi else None  # Remove quotes if present
        )

    def vector3(self, x, y, z):
        """Transform vector3 rule"""
        return (float(x), float(y), float(z))

    def generic(self, items):
        """
        Default transformation method for all nodes not explicitly handled.
        Follows the pattern from core/parser.py.
        
        :param items: List of child elements for the node
        :return: A generic dictionary structure representing the node
        """
        return {"type": str(type(self)), "data": items[0] if items else None, "children": items[1:]}


# Convenience function for backward compatibility
def parse_atomforge_string(code: str) -> AtomForgeFile:
    """
    Parse AtomForge DSL string and return AtomForgeFile object.
    This maintains compatibility with existing code.
    
    :param code: AtomForge DSL code as string
    :return: AtomForgeFile object
    """
    parser = AtomForgeParser()
    return parser.parse_and_transform(code) 