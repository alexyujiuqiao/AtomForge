#!/usr/bin/env python3
"""
Level-1 enrichment for AtomForge programs using LLM and Materials Project metadata.

This module provides functions to enrich minimal AtomForge programs with:
- Improved descriptions (using MP metadata + optional LLM)
- Properties blocks (from MP numeric data)
- Enhanced provenance information
"""

import json
from typing import Dict, List, Optional, Any

# Import composition formula helper from generator
from atomforge_generator import _composition_formula


def _extract_properties_from_mp(mp_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic extraction of a few useful scalar properties from a Materials Project
    summary document (mp_api.materials.summary).

    Expected keys (if present):
      - band_gap
      - formation_energy_per_atom
      - density

    Returns a dict with AtomForge-style property keys:
      {
          "band_gap": float,
          "formation_energy": float,
          "density": float
      }
    """
    if not mp_doc:
        return {}

    props: Dict[str, Any] = {}

    bg = mp_doc.get("band_gap")
    if bg is not None:
        props["band_gap"] = float(bg)

    fe = (
        mp_doc.get("formation_energy_per_atom")
        or mp_doc.get("formation_energy")
    )
    if fe is not None:
        props["formation_energy"] = float(fe)

    rho = mp_doc.get("density")
    if rho is not None:
        props["density"] = float(rho)

    return props


def _inject_or_replace_description(program: str, new_desc: str) -> str:
    """
    Replace the `description = "...",` line in a core AtomForge program
    with an enriched description. If not found, insert after header.
    """
    lines = program.splitlines()
    desc_line_idx = None

    for i, line in enumerate(lines):
        if "description" in line and "=" in line:
            desc_line_idx = i
            break

    desc_str = f'  description = "{new_desc}",'

    if desc_line_idx is not None:
        lines[desc_line_idx] = desc_str
        return "\n".join(lines)

    # If there is no description line yet, insert after the header block
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "}":  # end of header { ... }
            insert_idx = i + 1
            break

    if insert_idx is None:
        # Fallback: prepend near top
        lines.insert(1, desc_str)
    else:
        lines.insert(insert_idx, desc_str)

    return "\n".join(lines)


def _inject_properties_block(program: str, properties: Dict[str, Any]) -> str:
    """
    Insert a `properties { ... }` block (or replace existing content)
    using the given properties dict. Only called if properties is non-empty.
    """
    if not properties:
        return program

    lines = program.splitlines()

    # If there's already a properties block, we'll wipe its interior and replace.
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("properties {"):
            start_idx = i
        if start_idx is not None and line.strip() == "}":
            end_idx = i
            break

    prop_lines: List[str] = []
    prop_lines.append("  properties {")
    
    # Properties are separated by commas (grammar: property_entry ("," property_entry)*)
    prop_items = list(properties.items())
    for i, (key, value) in enumerate(prop_items):
        if i < len(prop_items) - 1:
            prop_lines.append(f"    {key} = {value},")
        else:
            prop_lines.append(f"    {key} = {value}")

    prop_lines.append("  }")

    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        # Replace existing properties block
        new_lines = lines[:start_idx] + prop_lines + lines[end_idx + 1 :]
        return "\n".join(new_lines)

    # Otherwise insert the block just before provenance { ... } if present
    prov_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("provenance {"):
            prov_idx = i
            break

    if prov_idx is None:
        # If we can't find provenance, insert before the final closing brace
        # (before the last line that is just "}" for atom_spec)
        closing_idx = len(lines) - 1
        for i in reversed(range(len(lines))):
            if lines[i].strip() == "}":
                closing_idx = i
                break
        new_lines = lines[:closing_idx] + prop_lines + lines[closing_idx:]
        return "\n".join(new_lines)

    new_lines = lines[:prov_idx] + prop_lines + lines[prov_idx:]
    return "\n".join(new_lines)


def _enrich_provenance_block(program: str, mp_doc: Optional[Dict[str, Any]]) -> str:
    """
    Optionally enrich the provenance block with a brief note and MP id if present.

    This assumes your core generator already emits a `provenance { ... }` block.
    We just insert some extensions before the closing brace.
    """
    if not mp_doc:
        return program

    mp_id = mp_doc.get("material_id")
    if mp_id is None:
        return program

    lines = program.splitlines()
    prov_start = None
    prov_end = None

    for i, line in enumerate(lines):
        if line.strip().startswith("provenance {"):
            prov_start = i
        if prov_start is not None and line.strip() == "}":
            prov_end = i
            break

    if prov_start is None or prov_end is None:
        # If no provenance block exists, create one before the final closing brace
        closing_idx = len(lines) - 1
        for i in reversed(range(len(lines))):
            if lines[i].strip() == "}":
                closing_idx = i
                break
        prov_block = [
            "  provenance {",
            f'    source = "Materials Project",',
            f'    method = "database_lookup",',
            f'    mp_id = "{mp_id}",',
            '    notes = "Enriched via LLM + MP metadata"',
            "  }"
        ]
        new_lines = lines[:closing_idx] + prov_block + lines[closing_idx:]
        return "\n".join(new_lines)

    # Insert just before the closing brace of provenance
    insert_idx = prov_end
    extra = [
        f'    mp_id = "{mp_id}",',
        '    notes = "Enriched via LLM + MP metadata"'
    ]

    new_lines = lines[:insert_idx] + extra + lines[insert_idx:]
    return "\n".join(new_lines)


def enrich_atomforge_description(
    crystal,
    base_program: str,
    mp_doc: Optional[Dict[str, Any]] = None,
    llm_client: Optional[Any] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Level-1 enrichment for a core AtomForge program:

    - Improves the `description` field (using MP metadata + optional LLM).
    - Adds a `properties { ... }` block if we have numeric MP data.
    - Adds a short MP-based note into `provenance { ... }`.

    Args:
        crystal: Crystal v1.1 object (used for formula + symmetry context).
        base_program: The level-0 AtomForge DSL program (string).
        mp_doc: Optional Materials Project summary document (dict).
        llm_client: Optional OpenAI/MCP-compatible client for LLM enrichment.
        model: Model name to call if llm_client is provided.

    Returns:
        Enriched AtomForge DSL program string 
    """

    # ----- 1) Build a default, deterministic description from local data -----
    formula = _composition_formula(crystal)
    sg_number = getattr(getattr(crystal, "symmetry", None), "number", None)
    sg_label = getattr(getattr(crystal, "symmetry", None), "space_group", None)

    base_desc_parts = [f"{formula} crystal structure"]
    if sg_label or sg_number:
        sg_str = sg_label or f"space group {sg_number}"
        base_desc_parts.append(f"in {sg_str}")
    base_desc = ", ".join(base_desc_parts)

    # ----- 2) If we have MP metadata, decorate the description a bit -----
    if mp_doc:
        backend = "Materials Project"
        mid = mp_doc.get("material_id")
        if mid:
            base_desc += f" (retrieved from {backend}, id={mid})"
        else:
            base_desc += f" (retrieved from {backend})"

    enriched_desc = base_desc

    # ----- 3) Optionally ask an LLM to make the description more human/physics-friendly -----
    # IMPORTANT: This is optional. If llm_client is None, we just keep base_desc.
    if llm_client is not None:
        context = {
            "formula": formula,
            "space_group_number": sg_number,
            "space_group_label": sg_label,
            "mp": mp_doc or {},
        }

        prompt = (
            "You are helping to enrich an AtomForge DSL crystal specification.\n"
            "Given the following JSON context, produce a short, single-sentence\n"
            "scientific but readable description of the material. Keep <= 200 characters.\n\n"
            f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
            "Return ONLY a JSON object of the form:\n"
            '{ "description": "... human readable description ..." }\n'
        )

        try:
            # Try OpenAI-compatible interface first
            if hasattr(llm_client, 'chat') and hasattr(llm_client.chat, 'completions'):
                # OpenAI-style client
                resp = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a materials science assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=300,
                )
                raw = resp.choices[0].message.content.strip()
            elif hasattr(llm_client, 'responses') and hasattr(llm_client.responses, 'create'):
                # MCP-style client (as in original code)
                resp = llm_client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": "You are a materials science assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                # Extract the first text output from the response
                text_chunks = []
                for item in resp.output:
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "output_text":
                            text_chunks.append(c.text.value)
                raw = "".join(text_chunks).strip() if text_chunks else ""
            else:
                # Fallback: assume it's callable or has a different interface
                raw = str(llm_client(prompt))

            data = json.loads(raw)
            if isinstance(data, dict) and "description" in data and data["description"]:
                enriched_desc = str(data["description"])
        except Exception:
            # If anything fails (no MCP, no JSON, etc.), silently fall back.
            enriched_desc = base_desc

    # ----- 4) Inject/replace description line in the program -----
    program = _inject_or_replace_description(base_program, enriched_desc)

    # ----- 5) Extract numeric properties from MP and inject properties block -----
    inferred_props = _extract_properties_from_mp(mp_doc)
    program = _inject_properties_block(program, inferred_props)

    # ----- 6) Enrich provenance with MP id + note -----
    program = _enrich_provenance_block(program, mp_doc)

    return program

