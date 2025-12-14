#!/usr/bin/env python3
"""
Phase 3 Demonstration Script (Initial Slice)

This demo exercises the new calculation preparation pipeline introduced at the
start of Phase 3. It reuses the Phase 2 editing demo crystal and showcases how
`prepare_calc` produces deterministic VASP-ready settings along with a detailed
`PrepReport` while keeping requested calculation targets typed and separate
from the crystal structure.

Requirements addressed (Phase 3 plan):
* Deterministic policy for functional, ENCUT, k-mesh, smearing, spin
* Separation of calculation targets from structure metadata
* Guarding property types (band_gap/formation_energy as numeric)
* Exposed `estimate_kmesh` helper

Steps demonstrated:
1. Build a test crystal via Phase 2 helper
2. Prepare calculation inputs with default policy
3. Prepare calculation inputs with user overrides (spin + ENCUT)
4. Show typed calculation targets, including band gap & formation energy
5. Display audit reports and highlight auto-detected magnetic spin enabling
"""

import sys
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crystal_v1_1 import create_simple_crystal, canonicalize
from crystal_edit import substitute
from crystal_calc import (
    prepare_calc,
    estimate_kmesh,
    CalculationTarget,
)


def build_demo_crystal():
    """Reuse Phase 2 capabilities to produce a doped Li-Ti spinful crystal."""

    crystal = create_simple_crystal(
        lattice_params=(4.2, 4.2, 4.2, 90.0, 90.0, 90.0),
        sites=[
            ("Li", (0.0, 0.0, 0.0)),
            ("O", (0.5, 0.5, 0.5)),
        ],
        space_group="Fm-3m",
        space_group_number=225,
    )

    crystal, _ = canonicalize(crystal)

    doped_crystal, _ = substitute(crystal, "Species:Li", {"Li": 0.9, "Fe": 0.1}, 1.0)

    return doped_crystal


def main():
    print("ATOMFORGE CRYSTAL PHASE 3 DEMONSTRATION (Initial Slice)")
    print("=" * 80)

    crystal = build_demo_crystal()

    print("1. Estimating k-point grid from policy density...")
    kmesh = estimate_kmesh(crystal)
    print(f"   Estimated k-mesh: {kmesh}")

    print("\n2. Preparing calculation input with default policy...")
    targets = [
        "total_energy",
        CalculationTarget(property="band_gap", unit="eV", accuracy=0.05),
        {"property": "formation_energy", "unit": "eV", "accuracy": 0.02},
    ]
    calc_input, prep_report = prepare_calc(crystal, backend_caps="vasp", targets=targets)

    print("   Settings:")
    pprint(calc_input.settings)
    print("\n   Targets:")
    for target in calc_input.targets:
        print(f"    - {target.property} ({target.unit}) [type={target.expected_type}]")

    print("\n   Prep Report Decisions:")
    pprint(prep_report.decisions)
    if prep_report.warnings:
        print("   Warnings:")
        for warning in prep_report.warnings:
            print(f"    * {warning}")

    print("\n3. Preparing calculation input with overrides (ENCUT=600, force spin off)...")
    overrides = {"encut": 600, "spin_polarized": False}
    calc_input2, prep_report2 = prepare_calc(
        crystal,
        backend_caps={"name": "vasp"},
        targets=targets,
        defaults=overrides,
    )

    print("   Settings:")
    pprint(calc_input2.settings)
    print("\n   Prep Report Decisions:")
    pprint(prep_report2.decisions)

    if prep_report2.warnings:
        print("   Warnings:")
        for warning in prep_report2.warnings:
            print(f"    * {warning}")

    print("\nPhase 3 initial demo complete!")


if __name__ == "__main__":
    main()


