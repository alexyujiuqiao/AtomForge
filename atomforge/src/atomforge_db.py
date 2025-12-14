#!/usr/bin/env python3
"""
Minimal MP -> Crystal v1.1 -> AtomForge program pipeline.

- Fetch structures from Materials Project
- Convert pymatgen.Structure -> Crystal (v1.1)
- Optionally canonicalize/validate
- Generate minimal AtomForge program text
- Write <material_name>.atomforge into atomforge_data/
"""

import os
from pathlib import Path
from datetime import datetime

from mp_api.client import MPRester          # pip install mp-api
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crystal_v1_1 import (
    Crystal,
    Lattice,
    Symmetry,
    Site,
    Composition,
    Provenance,
    canonicalize,
    validate,
)
from atomforge_generator import generate_minimal_atomforge_program


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Optional curated list of MP IDs (leave empty to auto-sample from MP).
MP_IDS = []

# Default number of materials to auto-sample when no mp_ids are provided
DEFAULT_LIMIT = 30

OUT_DIR = Path("atomforge_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# pymatgen.Structure -> Crystal v1.1
# (same spirit as from_cif(), but starting from an in-memory Structure)
# ---------------------------------------------------------------------

def structure_to_crystal(struct: Structure, mp_id: str, mp_symmetry=None) -> Crystal:
    """
    Convert a pymatgen Structure into a Crystal v1.1 object.

    This mirrors the logic of from_cif() in crystal_v1_1.py:
    - Lattice params from struct.lattice
    - Symmetry inferred via SpacegroupAnalyzer
    - Sites with species dict + frac coords
    - Composition from struct.composition
    - Provenance with database="MP", id=mp_id
    """

    # --- LATTICE ---
    lattice = Lattice(
        a=struct.lattice.a,
        b=struct.lattice.b,
        c=struct.lattice.c,
        alpha=struct.lattice.alpha,
        beta=struct.lattice.beta,
        gamma=struct.lattice.gamma,
    )

    # --- SYMMETRY ---
    # Prefer symmetry information from MP summary docs when provided;
    # fall back to SpacegroupAnalyzer if not.
    if mp_symmetry is not None:
        sg_symbol = getattr(mp_symmetry, "symbol", None)
        sg_number = getattr(mp_symmetry, "number", None)
        hall = getattr(mp_symmetry, "hall", None)
        origin_choice = getattr(mp_symmetry, "choice", None)
        crystal_system = getattr(mp_symmetry, "crystal_system", None)
        symmetry = Symmetry(
            space_group=str(sg_symbol) if sg_symbol is not None else "P1",
            number=int(sg_number) if sg_number is not None else 1,
            hall_symbol=str(hall) if hall is not None else None,
            origin_choice=str(origin_choice) if origin_choice is not None else None,
            symmetry_source="provided",
            crystal_system=str(crystal_system).lower() if crystal_system is not None else None,
        )
    else:
        analyzer = SpacegroupAnalyzer(struct, symprec=1e-5)
        space_group_symbol = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()

        symmetry = Symmetry(
            space_group=space_group_symbol,
            number=space_group_number,
            symmetry_source="inferred",
        )

    # --- SITES ---
    sites = []
    for site in struct:
        species_dict = {}
        for species, occupancy in site.species.items():
            species_dict[str(species)] = occupancy

        frac_coords = tuple(site.frac_coords)

        site_obj = Site(
            species=species_dict,
            frac=frac_coords,
            wyckoff=None,          # filled by canonicalize() if you want
            multiplicity=None,
            label=getattr(site, "label", None),
        )
        sites.append(site_obj)

    # --- COMPOSITION ---
    composition_dict = struct.composition.as_dict()

    reduced = {}
    for element, count in composition_dict.items():
        # mimic from_cif(): use int if the float is effectively integer
        try:
            reduced[element] = int(count) if float(count).is_integer() else float(count)
        except Exception:
            reduced[element] = count

    composition = Composition(
        reduced=reduced,
        # You *might* want true atomic fractions here; for now we just reuse
        # the raw composition dict like from_cif().
        atomic_fractions=composition_dict,
    )

    # --- PROVENANCE ---
    provenance = Provenance(
        database="MP",
        id=mp_id,
        retrieved_at=datetime.now().isoformat(),
        schema_version="AtomForge/Crystal/1.1",
        external_ids={"mp_id": mp_id},
    )

    crystal = Crystal(
        lattice=lattice,
        symmetry=symmetry,
        sites=tuple(sites),
        composition=composition,
        oxidation_states=None,
        constraints=None,
        provenance=provenance,
        notes=None,
    )

    return crystal


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def mp_to_atomforge(mp_ids=None, limit: int = DEFAULT_LIMIT):
    """
    For each mp-id:
      1. Fetch MP summary (with structure + formula).
      2. Convert pymatgen.Structure -> Crystal v1.1.
      3. Canonicalize + validate (optional but recommended).
      4. Generate minimal AtomForge program.
      5. Write <material_name>.atomforge into OUT_DIR.
    """
    api_key = os.environ.get("MP_API_KEY", "MP_API_KEY_REDACTED")

    with MPRester(api_key) as mpr:
        # Include symmetry so we can use MP's crystal_system / space group directly
        fields = ["material_id", "structure", "formula_pretty", "symmetry"]

        # If explicit IDs are provided, use those; otherwise auto-sample
        # the first `limit` materials from the summary search.
        docs = mpr.materials.summary.search(
            material_ids=mp_ids if mp_ids else None,
            fields=fields,
            num_chunks=1,
        )

        if not mp_ids and limit is not None:
            docs = docs[:limit]

        print(f"Fetched {len(docs)} materials from MP")

        for doc in docs:
            mp_id = str(doc.material_id)
            formula = doc.formula_pretty
            struct: Structure = doc.structure
            mp_symmetry = getattr(doc, "symmetry", None)

            print(f"\n=== {mp_id} ({formula}) ===")

            # 1) Structure -> Crystal (using MP symmetry where available)
            crystal = structure_to_crystal(struct, mp_id=mp_id, mp_symmetry=mp_symmetry)

            # 2) Optional: canonicalize + validate
            try:
                crystal, canon_report = canonicalize(crystal, policy="conventional")
                v_report = validate(crystal)

                if not v_report.ok:
                    print(f"  [skip] validation failed: {v_report.errors}")
                    continue
            except Exception as e:
                print(f"  [skip] canonicalize/validate error for {mp_id}: {e}")
                continue

            # 3) Crystal -> minimal AtomForge program (no defects/operations)
            # Use the chemical formula as the material name for atom_spec/header
            material_name = formula  # e.g. "Fe2", "Si", "NaCl"
            description = f"{formula} from Materials Project ({mp_id})"

            program_text = generate_minimal_atomforge_program(
                crystal=crystal,
                material_name=material_name,
                description=description,
            )

            # 4) Write to disk (sanitize formula for filename)
            file_stem = formula.replace(" ", "_")
            out_path = OUT_DIR / f"{file_stem}.atomforge"
            out_path.write_text(program_text, encoding="utf-8")
            print(f"  -> wrote {out_path}")

if __name__ == "__main__":
    mp_to_atomforge(MP_IDS or None, limit=DEFAULT_LIMIT)


