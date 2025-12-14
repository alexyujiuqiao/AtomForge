#!/usr/bin/env python3
"""
AtomForge Program Generator 
=====================================================

This module provides functionality to generate AtomForge DSL programs
from crystal structures and operations. It creates complete, executable
AtomForge programs that represent the current state of the crystal
after each operation.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


def _composition_formula(crystal) -> str:
    """
    Convert crystal.composition.reduced (e.g. {'Li':7,'La':3,'Zr':2,'O':12})
    into a simple chemical formula string "Li7La3Zr2O12".
    """
    if not hasattr(crystal, "composition") or not crystal.composition:
        return "Unknown"

    reduced = getattr(crystal.composition, "reduced", None) or {}
    # Sort by element symbol for determinism
    items = sorted(reduced.items(), key=lambda kv: kv[0])

    parts = []
    for elem, count in items:
        # count may be float or int; show integer if it is very close to an int
        try:
            f = float(count)
            if abs(f - round(f)) < 1e-6:
                n = int(round(f))
            else:
                n = f
        except Exception:
            n = count

        if n == 1:
            parts.append(f"{elem}")
        else:
            parts.append(f"{elem}{n}")
    return "".join(parts) if parts else "Unknown"

def generate_atomforge_program(crystal, 
                              material_name: str = "crystal",
                              description: str = "Generated crystal structure",
                              operations: List[Dict[str, Any]] = None,
                              supercell_info: Dict[str, Any] = None,
                              defects: List[Dict[str, Any]] = None) -> str:
    """
    Generate a complete AtomForge DSL program from a crystal structure.
    
    Args:
        crystal: Crystal object with lattice, sites, and other properties
        material_name: Name for the material in the program
        description: Description of the crystal structure
        operations: List of operations that have been applied
        supercell_info: Information about supercell creation
        
    Returns:
        Complete AtomForge DSL program as string
    """
    
    # Generate unique ID and timestamp
    program_uuid = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    # Start building the program
    program_lines = []
    
    # Header (complete according to FullLanguage.tex)
    program_lines.append(f'atom_spec "{material_name}" {{')
    program_lines.append('  header {')
    program_lines.append('    dsl_version = "2.1",')
    program_lines.append('    content_schema_version = "atomforge_crystal_v1.0",')
    program_lines.append(f'    uuid = "{program_uuid}",')
    program_lines.append(f'    title = "{crystal.composition}",')
    program_lines.append(f'    created = {timestamp}')
    program_lines.append('  },')
    
    # Description
    program_lines.append(f'  description = "{description}",')
    
    # Units (complete according to FullLanguage.tex)
    program_lines.append('  units {')
    program_lines.append('    system = "crystallographic_default",')
    program_lines.append('    length = angstrom,')
    program_lines.append('    angle = degree,')
    program_lines.append('    disp = "angstrom^2",')
    program_lines.append('    temp = K,')
    program_lines.append('    pressure = GPa')
    program_lines.append('  },')
    
    # Lattice (complete with description)
    program_lines.append('  lattice {')
    program_lines.append(f'    description = "{_get_crystal_system(crystal).title()} crystal system",')
    program_lines.append(f'    type = {_get_crystal_system(crystal)},')
    program_lines.append(f'    a = {crystal.lattice.a:.6f},')
    program_lines.append(f'    b = {crystal.lattice.b:.6f},')
    program_lines.append(f'    c = {crystal.lattice.c:.6f},')
    program_lines.append(f'    alpha = {crystal.lattice.alpha:.1f},')
    program_lines.append(f'    beta = {crystal.lattice.beta:.1f},')
    program_lines.append(f'    gamma = {crystal.lattice.gamma:.1f}')
    program_lines.append('  },')
    
    # Symmetry (complete with description)
    program_lines.append('  symmetry {')
    program_lines.append('    description = "Crystallographic symmetry",')
    if hasattr(crystal, 'space_group') and crystal.space_group:
        if isinstance(crystal.space_group, int):
            program_lines.append(f'    space_group = {crystal.space_group},')
        else:
            program_lines.append(f'    space_group = "{crystal.space_group}",')
    else:
        program_lines.append('    space_group = 1,')
    program_lines.append('    origin_choice = 1')
    program_lines.append('  },')
    
    # Basis (complete with all sites)
    program_lines.append('  basis {')
    program_lines.append('    description = "Atomic basis with all unique sites",')
    
    # Group sites by element for better organization
    sites_by_element = {}
    for i, site in enumerate(crystal.sites):
        if hasattr(site, 'species') and site.species:
            for element in site.species.keys():
                if element not in sites_by_element:
                    sites_by_element[element] = []
                sites_by_element[element].append((i, site))
    
    site_counter = 1
    for element, element_sites in sites_by_element.items():
        for i, site in element_sites:
            site_name = f"{element}{site_counter}"
            program_lines.append(f'    site "{site_name}" {{')
            program_lines.append(f'      description = "{element} site in {_get_crystal_system(crystal)} structure",')
            
            if hasattr(site, 'wyckoff') and site.wyckoff:
                program_lines.append(f'      wyckoff = "{site.wyckoff}",')
            else:
                # Generate a reasonable Wyckoff position
                program_lines.append(f'      wyckoff = "1a",')
            
            if hasattr(site, 'frac') and site.frac:
                program_lines.append(f'      position = ({site.frac[0]:.6f}, {site.frac[1]:.6f}, {site.frac[2]:.6f}),')
            else:
                program_lines.append('      position = (0.0, 0.0, 0.0),')
            
            program_lines.append('      frame = fractional,')
            
            # Species with complete specification
            if hasattr(site, 'species') and site.species:
                species_list = []
                for elem, occupancy in site.species.items():
                    species_list.append(f'{{ element = "{elem}", occupancy = {occupancy:.6f} }}')
                program_lines.append(f'      species = ({", ".join(species_list)}),')
            
            # Add thermal displacement parameters
            program_lines.append('      adp_iso = 0.005,')
            program_lines.append(f'      label = "{element}_site_{site_counter}"')
            program_lines.append('    },')
            site_counter += 1
    
    program_lines.append('  },')
    
    # Defects block (optional per FullLanguage.tex line 765-768)
    # Defects ::= "defects" "{" DefEntry ( "," DefEntry )* "}"
    # DefEntry ::= "{" "site_ref" "=" Id "," "type" "=" ( "vacancy" | "interstitial" | "substitution" ) "," "prob" "=" Num ( "," "species" "=" Species )? "}"
    if defects:
        program_lines.append('  defects {')
        defect_lines = generate_defects_block(defects, crystal)
        program_lines.extend(defect_lines)
        # Remove trailing comma from last line
        if program_lines[-1].endswith(','):
            program_lines[-1] = program_lines[-1][:-1]
        program_lines.append('  },')
    
    # Add patch operations if provided
    if operations:
        patch_lines = generate_patch_operations(operations)
        if patch_lines:
            program_lines.append('  patch {')
            program_lines.extend(patch_lines)
            # Remove trailing comma from last line
            if program_lines[-1].endswith(','):
                program_lines[-1] = program_lines[-1][:-1]
            program_lines.append('  },')
    
    # Add supercell information if provided (using tile block)
    if supercell_info:
        program_lines.append('  tile {')
        repeat = supercell_info.get('repeat', (1, 1, 1))
        program_lines.append(f'    repeat = ({repeat[0]}, {repeat[1]}, {repeat[2]}),')
        origin_shift = supercell_info.get('origin_shift', (0.0, 0.0, 0.0))
        program_lines.append(f'    origin_shift = ({origin_shift[0]:.6f}, {origin_shift[1]:.6f}, {origin_shift[2]:.6f})')
        program_lines.append('  },')
    
    # Properties block (comprehensive)
    program_lines.append('  properties {')
    program_lines.append('    band_gap = 4.5,  (* eV *)')
    program_lines.append('    formation_energy = -0.85,  (* eV/atom *)')
    program_lines.append('    density = 5.1,  (* g/cm³ *)')
    program_lines.append('    melting_point = 1873,  (* K *)')
    program_lines.append('    ionic_conductivity = 1e-4,  (* S/cm *)')
    program_lines.append('    electronic_conductivity = 1e-8,  (* S/cm *)')
    program_lines.append('    thermal_stability = 1273  (* K *)')
    program_lines.append('  },')
    
    # Validation block
    program_lines.append('  validation {')
    program_lines.append('    tolerance = 1e-6,')
    program_lines.append('    occupancy_clamp = true,')
    program_lines.append('    vector_unit_consistent = true,')
    program_lines.append('    max_transform_depth = 64,')
    program_lines.append('    enforce_units = true')
    program_lines.append('  },')
    
    # Provenance (comprehensive)
    program_lines.append('  provenance {')
    program_lines.append('    source = "AtomForge Crystal Generator",')
    program_lines.append('    method = "crystal_to_dsl_conversion",')
    program_lines.append('    software = "AtomForge v2.1",')
    program_lines.append('    computational_details = "Generated from crystal object",')
    program_lines.append('    doi = "",')
    program_lines.append('    url = "https://github.com/atomforge/atomforge"')
    program_lines.append('  }')
    
    # Close the program
    program_lines.append('}')
    
    return '\n'.join(program_lines)


def generate_minimal_atomforge_program(
    crystal,
    material_name: str,
    description: str,
    properties: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Minimal AtomForge program compatible with atomforge_core.lark.

    Blocks:
      - header
      - description (1-line, trailing comma)
      - units
      - lattice
      - symmetry
      - basis
      - (optional) properties
      - validation
      - provenance
    """

    program_uuid = str(uuid.uuid4())
    created = datetime.now().strftime("%Y-%m-%d")  # stored as string in header

    lines: List[str] = []

    # ------------------------------------------------------------------
    # atom_spec + header
    #   program: "atom_spec" identifier "{" ...
    #   identifier -> string | id
    # ------------------------------------------------------------------
    # Always emit material_name as a string identifier so we can use formula names
    lines.append(f'atom_spec "{material_name}" {{')

    lines.append("  header {")
    lines.append('    dsl_version = "2.1",')
    lines.append('    content_schema_version = "atomforge_crystal_v1.1",')
    lines.append(f'    uuid = "{program_uuid}",')

    title = _composition_formula(crystal)
    lines.append(f'    title = "{title}",')
    # created is stored as a string "YYYY-MM-DD" per core grammar
    lines.append(f'    created = "{created}"')
    lines.append("  }")

    # ------------------------------------------------------------------
    # description (rule ends with a comma)
    # ------------------------------------------------------------------
    lines.append(f'  description = "{description}",')

    # ------------------------------------------------------------------
    # units  (all fields required, last one without comma)
    #   We serialize units as strings to align with core grammar
    # ------------------------------------------------------------------
    lines.append("  units {")
    # Must match Units.validate() accepted systems in atomforge_ir.py
    lines.append('    system = "crystallography_default",')
    lines.append('    length = "angstrom",')
    lines.append('    angle = "degree",')
    lines.append('    disp = "angstrom^2",')
    lines.append('    temp = "K",')
    lines.append('    pressure = "GPa"')
    lines.append("  }")

    # ------------------------------------------------------------------
    # lattice  (bravais form)
    #   lattice: "lattice" "{" description? (bravais | vectors) "}"
    #   If we emit description, it must end with a comma.
    # ------------------------------------------------------------------
    cs = _get_crystal_system(crystal)  # "cubic", "tetragonal", ...
    L = crystal.lattice

    lines.append("  lattice {")
    lines.append(f'    description = "{cs.title()} crystal system",')
    # lat_type is an enum token, so no quotes
    lines.append(f"    type = {cs},")
    lines.append(f"    a = {L.a:.6f},")
    lines.append(f"    b = {L.b:.6f},")
    lines.append(f"    c = {L.c:.6f},")
    lines.append(f"    alpha = {L.alpha:.6f},")
    lines.append(f"    beta = {L.beta:.6f},")
    lines.append(f"    gamma = {L.gamma:.6f}")
    lines.append("  }")

    # ------------------------------------------------------------------
    # symmetry
    #   symmetry: "symmetry" "{" description? "space_group" "=" space_group ","
    #                           "origin_choice" "=" int ... "}"
    # ------------------------------------------------------------------
    sym = getattr(crystal, "symmetry", None)
    sg_value = None
    if sym is not None:
        # prefer explicit number if present; otherwise use symbol
        sg_value = getattr(sym, "number", None) or getattr(sym, "space_group", None)

    if isinstance(sg_value, int):
        space_group_repr = str(sg_value)  # int branch
    elif sg_value:
        # string branch: ESCAPED_STRING, so we quote it
        space_group_repr = f'"{sg_value}"'
    else:
        space_group_repr = "1"  # default P1

    lines.append("  symmetry {")
    lines.append(f"    space_group = {space_group_repr},")
    lines.append("    origin_choice = 1")
    lines.append("  }")

    # ------------------------------------------------------------------
    # basis / sites
    #   basis: "basis" "{" description? site ("," site)* "}"
    #   site:  "site" id "{" ... "}"
    #   id is ident or bt_ident -> NO quotes around site name
    #   species: element is an enum token: element = Li, O, ...
    # ------------------------------------------------------------------
    lines.append("  basis {")
    lines.append('    description = "Atomic basis with unique sites",')

    sites_by_element: Dict[str, List[Any]] = {}
    for idx, site in enumerate(crystal.sites):
        if getattr(site, "species", None):
            for elem in site.species.keys():
                sites_by_element.setdefault(str(elem), []).append((idx, site))

    site_counter = 1
    for elem, elem_sites in sites_by_element.items():
        for _, site in elem_sites:
            # bare identifier; ensure it's valid ident (letters/digits/_)
            site_name = getattr(site, "label", None) or f"{elem}{site_counter}"

            lines.append(f"    site {site_name} {{")
            lines.append(f'      description = "{elem} site in {cs} structure",')

            wyck = getattr(site, "wyckoff", None) or "1a"
            lines.append(f'      wyckoff = "{wyck}",')

            if getattr(site, "frac", None):
                fx, fy, fz = site.frac
                lines.append(f"      position = ({fx:.6f}, {fy:.6f}, {fz:.6f}),")
            else:
                lines.append("      position = (0.0, 0.0, 0.0),")

            lines.append("      frame = fractional,")

            species_entries: List[str] = []
            if getattr(site, "species", None):
                for s_elem, occ in site.species.items():
                    s_elem_str = str(s_elem)
                    species_entries.append(
                        f'{{ element = "{s_elem_str}", occupancy = {float(occ):.6f} }}'
                    )

            lines.append(f"      species = ({', '.join(species_entries)})")

            lines.append("    },")
            site_counter += 1

    # remove trailing comma after the last site
    if lines[-1].strip().endswith(","):
        lines[-1] = lines[-1].rstrip(",")

    lines.append("  }")

    # ------------------------------------------------------------------
    # optional properties  (simple numeric/string values, no comments)
    #   property_entry: qname "=" property_value
    #   property_value: number | string | "(" vec3 frame? ")" | vec3i
    #                   | "[" number ("," number)* "]" | bool
    # ------------------------------------------------------------------
    if properties:
        prop_lines: List[str] = []
        if "band_gap" in properties and properties["band_gap"] is not None:
            prop_lines.append(f"    band_gap = {properties['band_gap']}")
        if "formation_energy" in properties and properties["formation_energy"] is not None:
            prop_lines.append(f"    formation_energy = {properties['formation_energy']}")
        if "density" in properties and properties["density"] is not None:
            prop_lines.append(f"    density = {properties['density']}")

        if prop_lines:
            lines.append("  properties {")
            lines.extend(prop_lines)
            lines.append("  }")

    # ------------------------------------------------------------------
    # close atom_spec
    # ------------------------------------------------------------------
    lines.append("}")

    return "\n".join(lines)

def _get_crystal_system(crystal) -> str:
    """Determine crystal system, preferring MP / symmetry metadata when available."""
    # If crystal.symmetry has an explicit crystal_system string from MP, use it.
    sym = getattr(crystal, "symmetry", None)
    cs_str = getattr(sym, "crystal_system", None) if sym is not None else None
    if isinstance(cs_str, str) and cs_str:
        # Normalize to our token set
        cs_norm = cs_str.strip().lower()
        mapping = {
            "triclinic": "triclinic",
            "monoclinic": "monoclinic",
            "orthorhombic": "orthorhombic",
            "tetragonal": "tetragonal",
            "trigonal": "trigonal",
            "hexagonal": "hexagonal",
            "cubic": "cubic",
        }
        if cs_norm in mapping:
            return mapping[cs_norm]

    # Otherwise, if symmetry has a known number, map that to a crystal system
    sym = getattr(crystal, "symmetry", None)
    number = getattr(sym, "number", None) if sym is not None else None
    if isinstance(number, int) and 1 <= number <= 230:
        n = number
        if 1 <= n <= 2:
            return "triclinic"
        elif 3 <= n <= 15:
            return "monoclinic"
        elif 16 <= n <= 74:
            return "orthorhombic"
        elif 75 <= n <= 142:
            return "tetragonal"
        elif 143 <= n <= 167:
            return "trigonal"
        elif 168 <= n <= 194:
            return "hexagonal"
        else:
            return "cubic"

    # Fallback: infer from lattice metric if symmetry metadata is missing
    a, b, c = crystal.lattice.a, crystal.lattice.b, crystal.lattice.c
    alpha, beta, gamma = crystal.lattice.alpha, crystal.lattice.beta, crystal.lattice.gamma

    if abs(a - b) < 0.01 and abs(b - c) < 0.01 and abs(alpha - 90) < 0.1:
        return "cubic"
    elif abs(a - b) < 0.01 and abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1:
        return "tetragonal"
    elif abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 90) < 0.1:
        return "orthorhombic"
    elif abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1:
        return "monoclinic"
    elif abs(alpha - 90) < 0.1 and abs(gamma - 120) < 0.1:
        return "hexagonal"
    elif abs(alpha - 90) < 0.1 and abs(beta - 90) < 0.1 and abs(gamma - 120) < 0.1:
        return "trigonal"
    else:
        return "triclinic"

def generate_defects_block(defects: List[Dict[str, Any]], crystal) -> List[str]:
    """
    Generate defects block per FullLanguage.tex line 765-768.
    
    Defects ::= "defects" "{" DefEntry ( "," DefEntry )* "}"
    DefEntry ::= "{" "site_ref" "=" Id "," 
                 "type" "=" ( "vacancy" | "interstitial" | "substitution" ) "," 
                 "prob" "=" Num ( "," "species" "=" Species )? "}"
    
    Args:
        defects: List of defect dictionaries with keys:
            - site_ref: Site identifier (e.g., "Li1", "O1")
            - type: "vacancy", "interstitial", or "substitution"
            - prob: Probability (0.0 to 1.0)
            - species: Optional Species dict for substitution/interstitial
        crystal: Crystal object for reference
    
    Returns:
        List of defect entry lines
    """
    defect_lines = []
    
    for i, defect in enumerate(defects):
        site_ref = defect.get("site_ref", "site1")
        defect_type = defect.get("type", "vacancy")  # vacancy, interstitial, or substitution
        prob = defect.get("prob", 0.0)
        species = defect.get("species", None)
        
        # Start defect entry
        defect_lines.append('    {')
        defect_lines.append(f'      site_ref = "{site_ref}",')
        defect_lines.append(f'      type = "{defect_type}",')
        defect_lines.append(f'      prob = {prob:.6f}')
        
        # Optional species for substitution/interstitial (per line 768)
        if species and defect_type in ["substitution", "interstitial"]:
            defect_lines[-1] = defect_lines[-1] + ','
            # Species format: { element = "X", occupancy = N }
            element = species.get("element", "X")
            occupancy = species.get("occupancy", 1.0)
            species_str = f'{{ element = "{element}", occupancy = {occupancy:.6f} }}'
            defect_lines.append(f'      species = {species_str}')
        
        defect_lines.append('    }')
        
        # Add comma after each defect except the last
        if i < len(defects) - 1:
            defect_lines[-1] = defect_lines[-1] + ','
    
    return defect_lines


def generate_patch_operations(operations: List[Dict[str, Any]]) -> List[str]:
    """
    Generate proper patch operations according to FullLanguage.tex specification.
    
    Args:
        operations: List of operations to convert to patch format
        
    Returns:
        List of patch operation lines
    """
    patch_lines = []
    
    for op in operations:
        op_type = op.get("operation", "unknown")
        
        if op_type == "vacancy":
            # Create vacancy by updating occupancy
            target_site = op.get("params", {}).get("site_sel", "Species:Li")
            occupancy = op.get("params", {}).get("occupancy", 0.964)
            
            # Extract element from target (e.g., "Species:Li" -> "Li")
            element = target_site.split(":")[-1] if ":" in target_site else target_site
            
            patch_lines.append(f'    update basis.{element}1.species[0].occupancy = {occupancy:.6f},')
            
        elif op_type == "substitution":
            # Add substitution site
            target_element = op.get("params", {}).get("target_element", "Li")
            new_element = op.get("params", {}).get("new_element", "Ti")
            occupancy = op.get("params", {}).get("occupancy", 0.1)
            
            patch_lines.append(f'    update basis.{target_element}1.species[0].occupancy = {1.0 - occupancy:.6f},')
            patch_lines.append(f'    add site "{new_element}_dopant" {{')
            patch_lines.append('      wyckoff = "4a",')
            patch_lines.append('      position = (0.125, 0.125, 0.125),')
            patch_lines.append('      frame = fractional,')
            patch_lines.append(f'      species = ({{ element = "{new_element}", occupancy = {occupancy:.6f} }}),')
            patch_lines.append(f'      label = "{new_element}_substitution"')
            patch_lines.append('    },')
            
        elif op_type == "interstitial":
            # Add interstitial site
            element = op.get("params", {}).get("element", "Li")
            position = op.get("params", {}).get("position", (0.5, 0.5, 0.5))
            occupancy = op.get("params", {}).get("occupancy", 0.1)
            
            patch_lines.append(f'    add site "{element}_inter" {{')
            patch_lines.append('      wyckoff = "1a",')
            patch_lines.append(f'      position = ({position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}),')
            patch_lines.append('      frame = fractional,')
            patch_lines.append(f'      species = ({{ element = "{element}", occupancy = {occupancy:.6f} }}),')
            patch_lines.append(f'      label = "{element}_interstitial"')
            patch_lines.append('    },')
            
        elif op_type == "canonicalize":
            # Canonicalization doesn't need patch operations, it's already applied
            continue
            
        else:
            # Generic operation
            patch_lines.append(f'    (* {op_type} operation applied *)')
    
    return patch_lines

def generate_operation_program(base_program: str, 
                              operation: Dict[str, Any],
                              operation_name: str) -> str:
    """
    Generate a new AtomForge program by adding an operation to an existing program.
    
    Args:
        base_program: Existing AtomForge program as string
        operation: Operation to add (patch, tile, etc.)
        operation_name: Name of the operation
        
    Returns:
        Updated AtomForge program with the new operation
    """
    # Parse the base program and add the operation
    # This is a simplified version - in practice, you'd want to use the parser
    
    lines = base_program.split('\n')
    
    # Find where to insert the operation (before the closing brace)
    insert_index = -1
    for i, line in enumerate(lines):
        if line.strip() == '}':
            insert_index = i
            break
    
    if insert_index == -1:
        return base_program  # Fallback
    
    # Generate proper patch operations
    patch_lines = generate_patch_operations([operation])
    
    if patch_lines:
        # Insert the patch block
        operation_lines = ['  patch {']
        operation_lines.extend(patch_lines)
        # Remove trailing comma from last line
        if operation_lines[-1].endswith(','):
            operation_lines[-1] = operation_lines[-1][:-1]
        operation_lines.append('  },')
        
        # Insert the operation before the closing brace
        for i, op_line in enumerate(operation_lines):
            lines.insert(insert_index + i, op_line)
    
    return '\n'.join(lines)

if __name__ == "__main__":
    # Example usage
    print("AtomForge Program Generator")
    print("=" * 30)
    
    # This would be used with actual crystal objects in practice
    print("This module provides functions to generate AtomForge DSL programs")
    print("from crystal structures and operations.")
