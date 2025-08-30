#!/usr/bin/env python3
"""
Converter for Materials Project materials to AtomForge DSL format.

This script provides a function to convert Materials Project material data
to AtomForge DSL format, which can be used to create materials in the AtomForge
system.
"""

from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import uuid
import datetime
import numpy as np

API_KEY = "MP_API_KEY_REDACTED"  # Replace with your Materials Project API key

def get_lattice_type(sga):
    """Determine the lattice type based on space group analysis."""
    crystal_system = sga.get_crystal_system().lower()
    valid_lattice_types = [
        "cubic", "tetragonal", "orthorhombic", "hexagonal",
        "rhombohedral", "monoclinic", "triclinic"
    ]
    return crystal_system if crystal_system in valid_lattice_types else "triclinic"

def get_wyckoff_label(structure, site_index):
    """Get Wyckoff label for a site using SpacegroupAnalyzer."""
    sga = SpacegroupAnalyzer(structure)
    try:
        # Get symmetry dataset for accurate Wyckoff labels
        sym_dataset = sga.get_symmetry_dataset()
        wyckoff_letter = sym_dataset.wyckoffs[site_index]
        
        # Get multiplicity from symmetrized structure
        sym_struct = sga.get_symmetrized_structure()
        target_site = structure[site_index]
        target_coords = target_site.frac_coords
        target_species = target_site.species_string
        
        # Find the equivalent site group to get multiplicity
        for i, equiv_sites in enumerate(sym_struct.equivalent_sites):
            for equiv_site in equiv_sites:
                # Check if species match and coordinates are close
                if (equiv_site.species_string == target_species and 
                    np.allclose(equiv_site.frac_coords, target_coords, atol=1e-3)):
                    multiplicity = len(equiv_sites)
                    return f"{multiplicity}{wyckoff_letter}"
        
        # If no exact match found, try to find the closest match
        for i, equiv_sites in enumerate(sym_struct.equivalent_sites):
            for equiv_site in equiv_sites:
                if equiv_site.species_string == target_species:
                    # Check if coordinates are within tolerance
                    if np.allclose(equiv_site.frac_coords, target_coords, atol=0.1):
                        multiplicity = len(equiv_sites)
                        return f"{multiplicity}{wyckoff_letter}"
        
        # Fallback: use multiplicity from equivalent atoms
        equivalent_atoms = sym_dataset.equivalent_atoms
        multiplicity = np.sum(equivalent_atoms == equivalent_atoms[site_index])
        return f"{multiplicity}{wyckoff_letter}"
        
    except Exception as e:
        print(f"Warning: Could not determine Wyckoff label for site {site_index}: {e}")
        return "?"

def get_site_info(structure, site_index, sga):
    """Get detailed information about a site including occupancy and multiplicity."""
    try:
        # Get symmetry dataset for accurate Wyckoff labels
        sym_dataset = sga.get_symmetry_dataset()
        wyckoff_letter = sym_dataset.wyckoffs[site_index]
        
        # Get multiplicity from symmetrized structure
        sym_struct = sga.get_symmetrized_structure()
        target_site = structure[site_index]
        target_coords = target_site.frac_coords
        target_species = target_site.species_string
        
        # Find the equivalent site group to get multiplicity
        for i, equiv_sites in enumerate(sym_struct.equivalent_sites):
            for equiv_site in equiv_sites:
                if (equiv_site.species_string == target_species and 
                    np.allclose(equiv_site.frac_coords, target_coords, atol=1e-3)):
                    # Found the site group
                    multiplicity = len(equiv_sites)
                    
                    return {
                        "occupancy": 1.0 / multiplicity,  # Site occupancy
                        "multiplicity": multiplicity,
                        "wyckoff": f"{multiplicity}{wyckoff_letter}",
                        "label": f"{target_species}"
                    }
        
        # Fallback: use multiplicity from equivalent atoms
        equivalent_atoms = sym_dataset.equivalent_atoms
        multiplicity = np.sum(equivalent_atoms == equivalent_atoms[site_index])
        
        return {
            "occupancy": 1.0 / multiplicity,
            "multiplicity": multiplicity,
            "wyckoff": f"{multiplicity}{wyckoff_letter}",
            "label": f"{target_species}"
        }
    except Exception as e:
        print(f"Warning: Could not get detailed site info for site {site_index}: {e}")
        return {
            "occupancy": 1.0,
            "multiplicity": 1,
            "wyckoff": "?",
            "label": f"{target_species}?"
        }

def convert_material(
    material_id_or_formula,
    units_config=None,
    description="No description available.",
    computational_backend=None,
    convergence_criteria=None,
    target_properties=None,
    provenance_extra=None
):
    """
    Convert a Materials Project material to AtomForge DSL format.

    Args:
        material_id (str): Materials Project ID (e.g., 'mp-149').
        units_config (dict): Optional units configuration. Defaults to crystallographic_default, angstrom, degree.
        description (str): Optional description string.
        computational_backend (dict): Optional computational backend settings (e.g., VASP parameters).
        convergence_criteria (dict): Optional convergence criteria for property validation.
        target_properties (dict): Optional target properties to validate (e.g., formation_energy, band_gap).
        provenance_extra (dict): Optional additional provenance data (e.g., doi).
    """
    # Default units configuration
    default_units = {
        "system": "crystallographic_default",
        "length": "angstrom",
        "angle": "degree"
    }
    units_config = units_config or default_units
    # Validate units_config against grammar
    valid_length_units = ["angstrom", "nm", "pm", "bohr"]
    valid_angle_units = ["degree", "radian"]
    if units_config["length"] not in valid_length_units:
        raise ValueError(f"Invalid length unit: {units_config['length']}. Must be one of {valid_length_units}")
    if units_config["angle"] not in valid_angle_units:
        raise ValueError(f"Invalid angle unit: {units_config['angle']}. Must be one of {valid_angle_units}")

    # Default provenance extras
    default_provenance_extra = {}
    provenance_extra = provenance_extra or default_provenance_extra

    with MPRester(API_KEY) as m:
        # Handle different input types: MP ID, element name, or chemical formula
        if material_id_or_formula.startswith('mp-'):
            # Direct Materials Project ID
            material_id = material_id_or_formula
            doc = m.summary.search(material_ids=[material_id])[0]
        else:
            # Element name or chemical formula - search for it
            search_query = material_id_or_formula.strip()
            
            # Try to find the material
            try:
                # First, try to parse the input as a chemical formula and extract elements
                import re
                
                # Pattern to match chemical formulas like TaTe4, LiFePO4, etc.
                formula_pattern = r'([A-Z][a-z]?\d*)'
                elements_found = re.findall(formula_pattern, search_query)
                
                if elements_found:
                    # Extract unique element symbols (remove numbers)
                    element_symbols = []
                    for element in elements_found:
                        # Remove numbers and get just the element symbol
                        element_symbol = re.sub(r'\d+', '', element)
                        if element_symbol not in element_symbols:
                            element_symbols.append(element_symbol)
                    
                    # Try searching by elements first
                    try:
                        docs = m.summary.search(elements=element_symbols)
                        if docs:
                            # Filter to find the best match for the formula
                            best_match = None
                            for doc in docs:
                                if doc['formula_pretty'] == search_query:
                                    best_match = doc
                                    break
                            
                            if not best_match and docs:
                                # If no exact match, use the first result
                                best_match = docs[0]
                            
                            if best_match:
                                doc = best_match
                                material_id = doc['material_id']
                                print(f"Found material: {doc['formula_pretty']} (ID: {material_id})")
                            else:
                                raise ValueError(f"No material found for '{search_query}'")
                        else:
                            raise ValueError(f"No material found for '{search_query}'")
                    except Exception as element_error:
                        # If element search fails, try chemical system search
                        try:
                            # Create chemical system string (e.g., "Ta-Te" for TaTe4)
                            if len(element_symbols) == 2:
                                chemsys = f"{element_symbols[0]}-{element_symbols[1]}"
                            elif len(element_symbols) > 2:
                                chemsys = "-".join(element_symbols)
                            else:
                                chemsys = element_symbols[0]
                            
                            docs = m.summary.search(chemsys=chemsys)
                            if docs:
                                # Find best match
                                best_match = None
                                for doc in docs:
                                    if doc['formula_pretty'] == search_query:
                                        best_match = doc
                                        break
                                
                                if not best_match and docs:
                                    best_match = docs[0]
                                
                                if best_match:
                                    doc = best_match
                                    material_id = doc['material_id']
                                    print(f"Found material: {doc['formula_pretty']} (ID: {material_id})")
                                else:
                                    raise ValueError(f"No material found for '{search_query}'")
                            else:
                                raise ValueError(f"No material found for '{search_query}'")
                        except Exception as chemsys_error:
                            raise ValueError(f"Could not find material '{search_query}'. Tried elements: {element_symbols}")
                else:
                    # If it's not a chemical formula, try as a single element or chemical system
                    try:
                        # Try as a single element first
                        docs = m.summary.search(elements=[search_query])
                        if not docs:
                            # Try as chemical system
                            docs = m.summary.search(chemsys=search_query)
                        
                        if not docs:
                            raise ValueError(f"No material found for '{search_query}'. Please check the spelling or try a different search term.")
                        
                        # Use the first (most relevant) result
                        doc = docs[0]
                        material_id = doc['material_id']
                        print(f"Found material: {doc['formula_pretty']} (ID: {material_id})")
                        
                    except Exception as e:
                        raise ValueError(f"Error searching for '{search_query}': {e}")
                
            except Exception as e:
                raise ValueError(f"Error searching for '{search_query}': {e}")
        
        # Now we have the doc, continue with the rest of the processing
        structure = doc["structure"]
        sga = SpacegroupAnalyzer(structure)
        
        # Get available properties
        has_props = doc.get('has_props', {})
        
        # Extract computational backend from available data
        # Based on typical VASP calculations in Materials Project
        extracted_computational_backend = {
            "functional": "PBE",  # Most common in MP
            "energy_cutoff": 520,  # Typical for most materials
            "k_point_density": 1000.0  # Standard k-point density
        }
        
        # Try to extract from entry parameters if available
        try:
            entries = m.get_entries(material_id)
            if entries:
                entry = entries[0]
                if hasattr(entry, 'parameters') and entry.parameters:
                    if 'run_type' in entry.parameters:
                        extracted_computational_backend["functional"] = entry.parameters['run_type']
        except:
            pass
        
        computational_backend = computational_backend or extracted_computational_backend

        # Extract convergence criteria based on available properties
        extracted_convergence_criteria = {
            "energy_tolerance": 1e-5,  # Standard for MP calculations
            "force_tolerance": 0.01,   # Standard for MP calculations
            "stress_tolerance": 0.1    # Standard for MP calculations
        }
        convergence_criteria = convergence_criteria or extracted_convergence_criteria

        # Extract target properties based on available data
        extracted_target_properties = {
            "formation_energy": has_props.get('thermo', False),
            "band_gap": has_props.get('electronic_structure', False),
            "elastic_constants": has_props.get('elasticity', False)
        }
        target_properties = target_properties or extracted_target_properties

        # Header
        today = datetime.date.today()
        header = {
            "dsl_version": "1.0",
            "title": doc["formula_pretty"],
            "created": f"{today.year}-{today.month}-{today.day}",  # Unquoted: 2025-7-12
            "uuid": str(uuid.uuid4())
        }

        # Lattice
        lattice = structure.lattice
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        lattice_type = get_lattice_type(sga)

        # Symmetry
        space_group = sga.get_space_group_symbol()

        # Build DSL
        dsl = []
        dsl.append('#atomforge_version "1.0";')
        dsl.append(f'atom_spec {header["title"]} {{')
        
        # Header section (all fields required)
        dsl.append("  header {")
        dsl.append(f'    dsl_version = "{header["dsl_version"]}",')
        dsl.append(f'    title = "{header["title"]}",')
        dsl.append(f'    created = {header["created"]},')  # Unquoted date
        dsl.append(f'    uuid = "{header["uuid"]}",')
        dsl.append("  }")

        # Description section (all fields required)
        dsl.append(f' description = "{description}",')

        # Units section (all fields required)
        dsl.append("  units {")
        dsl.append(f'    system = "{units_config["system"]}",')
        dsl.append(f'    length = {units_config["length"]},')  # Unquoted
        dsl.append(f'    angle = {units_config["angle"]},')   # Unquoted
        dsl.append("  }")

        # Lattice section (all fields required)
        dsl.append("  lattice {")
        dsl.append(f'    type = {lattice_type},')
        dsl.append(f'    a = {a:.10f},')
        dsl.append(f'    b = {b:.10f},')
        dsl.append(f'    c = {c:.10f},')
        dsl.append(f'    alpha = {alpha:.10f},')
        dsl.append(f'    beta = {beta:.10f},')
        dsl.append(f'    gamma = {gamma:.10f},')
        dsl.append("  }")

        # Symmetry section (space_group required, origin_choice optional)
        dsl.append("  symmetry {")
        dsl.append(f'    space_group = "{space_group}",')
        dsl.append('    origin_choice = 1,')  # Included as default
        dsl.append("  }")

        # Basis section (required)
        dsl.append("  basis {")
        for i, site in enumerate(structure):
            pos = site.frac_coords
            wyckoff_label = get_wyckoff_label(structure, i)
            
            # Get site multiplicity and other details
            site_info = get_site_info(structure, i, sga)
            
            dsl.append(f'    site {site.species_string}{i+1} {{')
            dsl.append(f'      wyckoff = "{wyckoff_label}",')
            dsl.append(f'      position = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}),')
            dsl.append('      frame = fractional,')
            dsl.append(f'      species = ({{ element = "{site.species_string}", occupancy = {site_info["occupancy"]:.3f} }}),')
            if site_info.get("label"):
                dsl.append(f'      label = "{site_info["label"]}",')
            dsl.append("    }")
        dsl.append("  }")

        # Property validation section (all subfields required)
        dsl.append("  property_validation {")
        # Computational backend (required, non-empty)
        dsl.append("    computational_backend: VASP {")
        for k, v in computational_backend.items():
            value = f'"{v}"' if isinstance(v, str) else v
            dsl.append(f'      {k}: {value},')
        dsl.append("    },")
        # Convergence criteria (required, non-empty)
        dsl.append("    convergence_criteria: {")
        for k, v in convergence_criteria.items():
            dsl.append(f'      {k}: {v},')
        dsl.append("    },")
        # Target properties (required, non-empty)
        dsl.append("    target_properties: {")
        for k, v in target_properties.items():
            dsl.append(f'      {k}: {str(v).lower()},')
        dsl.append("    },")
        dsl.append("  }")

        # Provenance section (required, source mandatory, others optional)
        dsl.append("  provenance {")
        dsl.append('    source = "Materials Project",')
        
        # Extract method from available data
        method = "VASP DFT calculation"
        if has_props.get('elasticity', False):
            method += " with elastic properties"
        if has_props.get('electronic_structure', False):
            method += " with electronic structure"
        if has_props.get('dielectric', False):
            method += " with dielectric properties"
        
        dsl.append(f'    method = "{method}",')
        
        # Try to extract DOI from available data
        doi_to_include = None
        
        # First check if DOI is provided in provenance_extra
        if "doi" in provenance_extra:
            doi_to_include = provenance_extra["doi"]
        else:
            # Try to construct DOI from ICSD IDs if available
            if 'database_IDs' in doc and 'icsd' in doc['database_IDs']:
                icsd_ids = doc['database_IDs']['icsd']
                if icsd_ids:
                    # Use the first ICSD ID to construct a potential DOI
                    # ICSD entries often have DOIs in the format: 10.17188/XXXXXXX
                    icsd_number = icsd_ids[0].replace('icsd-', '')
                    doi_to_include = f"10.17188/{icsd_number}"
        
        if doi_to_include:
            dsl.append(f'    doi = "{doi_to_include}",')
        dsl.append("  }")

        dsl.append("}")

        return "\n".join(dsl)

# Example run:
if __name__ == "__main__":
    # Test different input types
    test_cases = [
        "mp-149",      # Materials Project ID
        "Si",          # Element name
        "Fe",          # Element name
        "Cu",          # Element name
        "TaTe4",       # Chemical formula
    ]
    
    for test_input in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing input: {test_input}")
        print(f"{'='*60}")
        
        try:
            output = convert_material(
                test_input,
                description=f"{test_input} crystal structure from Materials Project"
            )
            print(output)
        except Exception as e:
            print(f"Error: {e}")
        
        print(f"\n{'-'*60}")