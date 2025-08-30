from lark import Lark, Transformer, v_args
from pathlib import Path
from datetime import datetime
import os

from .atomforge_ir import (
    AtomForgeFile, LanguageVersionDecl, AtomSpec, Header, Units,
    Lattice, Symmetry, Site, Species, ConvergenceCriteria,
    TargetProperties, PropertyValidation, Provenance
)

# load the grammar from the file
GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "atomforge.lark")
with open(GRAMMAR_PATH, "r", encoding="utf-8") as f:
    ATOMFORGE_GRAMMAR = f.read()

parser = Lark(ATOMFORGE_GRAMMAR, start="start", parser="lalr", propagate_positions=False)

@v_args(inline=True)
class AtomForgeTransformer(Transformer):
    def start(self, *args):
        # args = [atomforge_file]
        return args[0]
    def atomforge_file(self, language_version=None, atom_spec=None):
        from .atomforge_ir import AtomForgeFile
        return AtomForgeFile(language_version=language_version, spec=atom_spec)

    def language_version_decl(self, ver):
        return LanguageVersionDecl(version=ver[1:-1])

    def atom_spec(self, name, header, description=None, units=None, lattice=None, symmetry=None, basis=None, property_validation=None, provenance=None):
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

    def header(self, dsl_version_val, title_val, created_val, uuid_val=None):
        # dsl_version_val, title_val: quoted strings; created_val: Tree(date, [y,m,d]); uuid_val: quoted string or None
        from datetime import datetime
        if isinstance(created_val, list) or hasattr(created_val, '__iter__'):
            # likely Tree(date, [y, m, d])
            y, m, d = created_val
            created = datetime(int(y), int(m), int(d))
        else:
            created = created_val
        uuid = uuid_val[1:-1] if uuid_val else None
        return Header(
            dsl_version=dsl_version_val[1:-1],
            title=title_val[1:-1],
            created=created,
            uuid=uuid
        )

    def description(self, desc_val):
        return desc_val[1:-1]

    def units(self, system_val, length_val, angle_val):
        return Units(
            system=system_val[1:-1] if system_val and system_val.startswith('"') else str(system_val),
            length=length_val,
            angle=angle_val
        )

    def lattice(self, type_val, a_val, b_val, c_val, alpha_val, beta_val, gamma_val):
        return Lattice(
            type=type_val,
            a=float(a_val),
            b=float(b_val),
            c=float(c_val),
            alpha=float(alpha_val),
            beta=float(beta_val),
            gamma=float(gamma_val)
        )

    def symmetry(self, space_group, origin_choice=None):
        return Symmetry(
            space_group=space_group,
            origin_choice=origin_choice
        )

    def basis(self, *sites):
        return list(sites)

    def site(self, *args):
        # Parse the site arguments manually since the grammar produces many tokens
        # Expected order: name, wyckoff, position, frame, species, label (adp_iso is optional)
        name = str(args[0])
        wyckoff = args[1][1:-1] if args[1] and args[1].startswith('"') else str(args[1])
        position = args[2]
        frame = args[3]
        species = args[4]
        
        # Check if adp_iso is present (it would be before label)
        adp_iso = None
        label = None
        
        if len(args) > 5:
            # Check if the next argument looks like a float (adp_iso)
            try:
                if isinstance(args[5], (int, float)) or (isinstance(args[5], str) and args[5].replace('.', '').replace('-', '').isdigit()):
                    adp_iso = float(args[5])
                    if len(args) > 6:
                        label = args[6][1:-1] if args[6] and args[6].startswith('"') else str(args[6])
                else:
                    # No adp_iso, this is the label
                    label = args[5][1:-1] if args[5] and args[5].startswith('"') else str(args[5])
            except (ValueError, IndexError):
                # If conversion fails, treat as label
                if len(args) > 5:
                    label = args[5][1:-1] if args[5] and args[5].startswith('"') else str(args[5])
        
        return Site(
            name=name,
            wyckoff=wyckoff,
            position=position,
            frame=frame,
            species=species,
            adp_iso=adp_iso,
            label=label
        )

    def species_list(self, *species):
        return list(species)

    def species(self, element, occupancy, charge=None):
        # Handle element symbol - remove quotes if it's a string literal
        element_str = element[1:-1] if isinstance(element, str) and element.startswith('"') else str(element)
        return Species(
            element=element_str,
            occupancy=float(occupancy),
            charge=float(charge) if charge is not None else None
        )

    def property_validation(self, computational_backend, convergence_criteria=None, target_properties=None):
        return PropertyValidation(
            computational_backend=computational_backend,
            convergence_criteria=convergence_criteria,
            target_properties=target_properties
        )

    def computational_backend(self, functional, energy_cutoff, k_point_density):
        #  IR  dict  dataclass
        return {
            'functional': functional[1:-1] if functional and functional.startswith('"') else str(functional),
            'energy_cutoff': float(energy_cutoff),
            'k_point_density': float(k_point_density)
        }

    def convergence_criteria(self, *args):
        # Parse keyword arguments from the grammar
        energy_tolerance = None
        force_tolerance = None
        stress_tolerance = None
        
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == "energy_tolerance":
                energy_tolerance = float(args[i + 1]) if i + 1 < len(args) else None
            elif isinstance(arg, str) and arg == "force_tolerance":
                force_tolerance = float(args[i + 1]) if i + 1 < len(args) else None
            elif isinstance(arg, str) and arg == "stress_tolerance":
                stress_tolerance = float(args[i + 1]) if i + 1 < len(args) else None
        
        return ConvergenceCriteria(
            energy_tolerance=energy_tolerance,
            force_tolerance=force_tolerance,
            stress_tolerance=stress_tolerance
        )

    def target_properties(self, *args):
        # Parse keyword arguments from the grammar
        formation_energy = None
        band_gap = None
        elastic_constants = None
        
        for i, arg in enumerate(args):
            if isinstance(arg, str) and arg == "formation_energy":
                formation_energy = args[i + 1] if i + 1 < len(args) else None
            elif isinstance(arg, str) and arg == "band_gap":
                band_gap = args[i + 1] if i + 1 < len(args) else None
            elif isinstance(arg, str) and arg == "elastic_constants":
                elastic_constants = args[i + 1] if i + 1 < len(args) else None
        
        return TargetProperties(
            formation_energy=formation_energy,
            band_gap=band_gap,
            elastic_constants=elastic_constants
        )

    def provenance(self, source, method=None, doi=None):
        return Provenance(
            source=source[1:-1] if source and source.startswith('"') else str(source),
            method=method[1:-1] if method else None,
            doi=doi[1:-1] if doi else None
        )

    def vector3(self, x, y, z):
        return (float(x), float(y), float(z))

    def lattice_type(self, val=None):
        return str(val) if val is not None else None

    def space_group(self, val):
        return str(val[1:-1]) if val.startswith('"') else str(val)

    def coordinate_frame(self, val=None):
        return str(val) if val is not None else None

    def length_unit(self, val=None):
        return str(val) if val is not None else None

    def angle_unit(self, val=None):
        return str(val) if val is not None else None

    def element_symbol(self, val):
        return str(val[1:-1]) if val.startswith('"') else str(val)

    def STRING_LITERAL(self, s):
        return str(s)

    def REAL_LITERAL(self, n):
        return float(n)

    def INTEGER_LITERAL(self, n):
        return int(n)

    def BOOL(self, b):
        return b == "true"

# Usage:
def parse_atomforge_string(text: str) -> AtomForgeFile:
    tree = parser.parse(text)
    return AtomForgeTransformer().transform(tree)