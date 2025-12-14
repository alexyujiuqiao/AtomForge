#!/usr/bin/env python3
"""
AtomForge Crystal Calculation Prep - Phase 3 Foundations

This module establishes the first slice of Phase 3 from the AtomForge Crystal
MVP plan. It focuses on deterministic calculation preparation targeting a VASP
backend and introduces typed calculation targets that stay separate from the
Crystal structure itself.

Covered requirements from plan-v-2-0 Phase 3:
* prepare_calc: deterministic policy for functional, ENCUT, k-mesh, smearing, spin
* estimate_kmesh helper following a density-based policy
* Separate structured CalcTarget definitions so requested properties are typed
* Guard against Bool/Float ambiguity for properties like band_gap / formation_energy
* Attach an auditable PrepReport describing policy decisions

Additional safeguards:
* Automatic spin detection for magnetic species when not explicitly requested
* Policy constants collected in DEFAULT_VASP_POLICY for future tuning
* Units captured explicitly in CalculationTarget to avoid implicit assumptions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from datetime import datetime

from crystal_v1_1 import Crystal


# ==========================================================================
# Calculation Target & Settings Data Structures
# ==========================================================================


@dataclass(frozen=True)
class CalculationTarget:
    """Structured representation of a requested calculation observable."""

    property: str
    unit: str
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected_type: str = "float"  # Enforce numeric outputs for Phase 3 targets
    value: Optional[float] = None


@dataclass(frozen=True)
class CalcSettings:
    """Deterministic calculation settings for a backend run."""

    backend: str
    functional: str
    encut: int
    kpoint_grid: Tuple[int, int, int]
    smearing: str
    sigma: float
    spin_polarized: bool
    additional: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalcInput:
    """Complete calculation bundle that can be handed to execution tooling."""

    crystal: Crystal
    settings: CalcSettings
    targets: Tuple[CalculationTarget, ...]
    policy_version: str = "AtomForge/CalcPrep/0.1"


@dataclass
class PrepReport:
    """Audit report describing how prepare_calc filled in settings."""

    backend: str
    decisions: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    policy_version: str = "AtomForge/CalcPrep/0.1"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ==========================================================================
# Policy Defaults & Utilities
# ==========================================================================


DEFAULT_VASP_POLICY: Dict[str, Any] = {
    "policy_name": "vasp_default_v1",
    "functional": "PBE",
    "encut": 520,
    "smearing": "fermi",
    "sigma": 0.05,
    "spin_polarized": False,
    "magnetic_elements": {
        "Fe",
        "Co",
        "Ni",
        "Mn",
        "Cr",
        "V",
        "Ti",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
    },
    "kpoint_density_per_angstrom": 0.25,
    "kpoint_min_grid": 1,
    "kpoint_max_grid": 12,
    "energy_tolerance": 1e-5,
    "force_tolerance": 0.02,
    "stress_tolerance": 0.5,
    "max_steps": 200,
    "mixing_scheme": "Pulay",
    "mixing_parameter": 0.4,
    "electronic_temperature": 0.02,
}


def estimate_kmesh(crystal: Crystal, policy: Optional[Mapping[str, Any]] = None) -> Tuple[int, int, int]:
    """Estimate a Monkhorst-Pack grid using lattice lengths and a density policy."""

    if policy is None:
        policy = DEFAULT_VASP_POLICY

    density = float(policy.get("kpoint_density_per_angstrom", 0.25))
    min_grid = int(policy.get("kpoint_min_grid", 1))
    max_grid = int(policy.get("kpoint_max_grid", 12))

    lattice = crystal.lattice
    lengths = (lattice.a, lattice.b, lattice.c)
    grid: List[int] = []

    for axis, length in zip(["a", "b", "c"], lengths):
        if length <= 0:
            grid.append(min_grid)
            continue

        raw = int(round(max(min_grid, density * length)))
        capped = min(max_grid, max(min_grid, raw))
        grid.append(capped)

    return tuple(grid)  # type: ignore[arg-type]


def _contains_magnetic_species(crystal: Crystal, magnetic_elements: Iterable[str]) -> bool:
    lookup = {elem.capitalize() for elem in magnetic_elements}
    for site in crystal.sites:
        for species in site.species:
            if species.capitalize() in lookup:
                return True
    return False


def _normalize_backend(backend_caps: Union[str, Mapping[str, Any], None]) -> Tuple[str, Dict[str, Any]]:
    if backend_caps is None:
        return "VASP", {}

    if isinstance(backend_caps, str):
        return backend_caps.upper(), {}

    backend = backend_caps.get("name") or backend_caps.get("backend") or "VASP"
    return str(backend).upper(), dict(backend_caps)


def _normalize_targets(
    targets: Optional[Sequence[Union[str, CalculationTarget, Mapping[str, Any]]]]
) -> Tuple[CalculationTarget, ...]:
    """Create a typed target tuple, applying default units and type fixes."""

    if not targets:
        return (
            CalculationTarget(property="total_energy", unit="eV"),
            CalculationTarget(property="band_gap", unit="eV"),
        )

    normalized: List[CalculationTarget] = []

    for entry in targets:
        if isinstance(entry, CalculationTarget):
            normalized.append(entry)
            continue

        if isinstance(entry, str):
            property_name = entry
            default_unit = "eV" if entry in {"band_gap", "formation_energy", "total_energy"} else "unknown"
            normalized.append(CalculationTarget(property=property_name, unit=default_unit))
            continue

        if isinstance(entry, Mapping):
            property_name = str(entry.get("property"))
            unit = str(entry.get("unit", "unknown"))
            accuracy = entry.get("accuracy")
            metadata = dict(entry.get("metadata", {}))
            expected_type = str(entry.get("expected_type", "float"))
            value = entry.get("value")

            if isinstance(value, bool):
                metadata.setdefault("converted_from_bool", True)
                value = float(value)

            if property_name in {"band_gap", "formation_energy"} and value is None:
                metadata.setdefault("expected_numeric", True)

            normalized.append(
                CalculationTarget(
                    property=property_name,
                    unit=unit,
                    accuracy=accuracy,
                    metadata=metadata,
                    expected_type=expected_type,
                    value=float(value) if isinstance(value, (int, float)) else value,
                )
            )
            continue

        raise TypeError(f"Unsupported calculation target type: {type(entry)!r}")

    return tuple(normalized)


# ==========================================================================
# Public API: prepare_calc
# ==========================================================================


def prepare_calc(
    crystal: Crystal,
    backend_caps: Union[str, Mapping[str, Any], None] = None,
    targets: Optional[Sequence[Union[str, CalculationTarget, Mapping[str, Any]]]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
) -> Tuple[CalcInput, PrepReport]:
    """Prepare deterministic calculation inputs following Phase 3 policy."""

    backend, backend_details = _normalize_backend(backend_caps)
    backend = backend.upper()

    if backend not in {"VASP"}:
        raise ValueError(f"Unsupported backend '{backend}'.")

    policy: Dict[str, Any] = dict(DEFAULT_VASP_POLICY)
    if defaults:
        policy.update(defaults)

    decisions: Dict[str, Any] = {
        "backend": backend,
        "backend_details": backend_details,
        "policy_name": policy.get("policy_name", "vasp_default_v1"),
    }

    # Functional selection
    functional = str(policy.get("functional", "PBE"))
    decisions["functional"] = {
        "value": functional,
        "source": "user_override" if defaults and "functional" in defaults else "policy_default",
    }

    # ENCUT selection
    encut = int(policy.get("encut", 520))
    decisions["encut"] = {
        "value": encut,
        "source": "user_override" if defaults and "encut" in defaults else "policy_default",
    }

    # Spin configuration
    spin_source = "policy_default"
    if defaults and "spin_polarized" in defaults:
        spin_polarized = bool(defaults["spin_polarized"])
        spin_source = "user_override"
    else:
        spin_polarized = bool(policy.get("spin_polarized", False))
        if not spin_polarized:
            magnetic_elements = policy.get("magnetic_elements", set())
            if _contains_magnetic_species(crystal, magnetic_elements):
                spin_polarized = True
                spin_source = "auto_detect_magnetic_species"

    decisions["spin_polarized"] = {"value": spin_polarized, "source": spin_source}

    # K-point grid
    kpoint_grid = estimate_kmesh(crystal, policy)
    decisions["kpoint_grid"] = {
        "value": kpoint_grid,
        "source": "user_override" if defaults and "kpoint_grid" in defaults else "density_policy",
        "density": policy.get("kpoint_density_per_angstrom"),
    }

    # Smearing parameters
    smearing = str(policy.get("smearing", "fermi"))
    sigma = float(policy.get("sigma", 0.05))
    decisions["smearing"] = {
        "value": smearing,
        "sigma": sigma,
        "source": "user_override" if defaults and ("smearing" in defaults or "sigma" in defaults) else "policy_default",
    }

    additional: Dict[str, Any] = {
        "energy_tolerance": float(policy.get("energy_tolerance", 1e-5)),
        "force_tolerance": float(policy.get("force_tolerance", 0.02)),
        "stress_tolerance": float(policy.get("stress_tolerance", 0.5)),
        "max_steps": int(policy.get("max_steps", 200)),
        "mixing_scheme": policy.get("mixing_scheme", "Pulay"),
        "mixing_parameter": float(policy.get("mixing_parameter", 0.4)),
        "electronic_temperature": float(policy.get("electronic_temperature", 0.02)),
    }

    decisions["additional"] = additional

    calc_settings = CalcSettings(
        backend=backend,
        functional=functional,
        encut=encut,
        kpoint_grid=kpoint_grid,
        smearing=smearing,
        sigma=sigma,
        spin_polarized=spin_polarized,
        additional=additional,
    )

    calc_targets = _normalize_targets(targets)

    calc_input = CalcInput(
        crystal=crystal,
        settings=calc_settings,
        targets=calc_targets,
    )

    warnings: List[str] = []
    notes: List[str] = []

    if spin_source == "auto_detect_magnetic_species":
        warnings.append("Spin polarization enabled due to magnetic species detection")

    if min(calc_settings.kpoint_grid) < policy.get("kpoint_min_grid", 1):
        warnings.append("K-point grid fell below minimum policy value")

    if crystal.provenance.hash:
        notes.append(f"identity_hash={crystal.provenance.hash}")

    report = PrepReport(
        backend=backend,
        decisions=decisions,
        warnings=warnings,
        notes=notes,
    )

    return calc_input, report


__all__ = [
    "CalculationTarget",
    "CalcSettings",
    "CalcInput",
    "PrepReport",
    "prepare_calc",
    "estimate_kmesh",
]


