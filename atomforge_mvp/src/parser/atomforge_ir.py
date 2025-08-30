from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime

# language version declaration
@dataclass
class LanguageVersionDecl:
    version: str

# Top-level file, may include a version declaration and exactly one AtomSpec
@dataclass
class AtomForgeFile:
    language_version: Optional[LanguageVersionDecl]
    spec: 'AtomSpec' = None

# Core AtomForge specification
@dataclass
class AtomSpec:
    name: str
    header: 'Header'
    description: Optional[str] = None
    units: Optional['Units'] = None
    lattice: 'Lattice' = None
    symmetry: 'Symmetry' = None
    basis: List['Site'] = field(default_factory=list)
    property_validation: Optional['PropertyValidation'] = None
    provenance: Optional['Provenance'] = None

# Metadata header 
@dataclass
class Header:
    dsl_version: str
    title: str
    created: datetime
    uuid: Optional[str] = None

# Units system declaration
@dataclass
class Units:
    system: str
    length: str
    angle: str

# Crystallographic lattice parameters
@dataclass
class Lattice:
    type: str
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

# Space group symmetry info
@dataclass
class Symmetry:
    space_group: str
    origin_choice: Optional[int] = None

# Species entry within a site
@dataclass
class Species:
    element: str
    occupancy: float
    charge: Optional[float] = None

# Atomic site definition
@dataclass
class Site:
    name: str
    wyckoff: str
    position: Tuple[float, float, float]
    frame: str
    species: List[Species]
    adp_iso: Optional[float] = None
    label: Optional[str] = None

# Convergence Criteria
@dataclass
class ConvergenceCriteria:
    energy_tolerance: Optional[float] = None
    force_tolerance: Optional[float] = None
    stress_tolerance: Optional[float] = None

# target properties
@dataclass
class TargetProperties:
    formation_energy: Optional[bool] = None
    band_gap: Optional[bool] = None
    elastic_constants: Optional[bool] = None

# Validation settings for computational properties
@dataclass
class PropertyValidation:
    computational_backend: str
    convergence_criteria: Optional[ConvergenceCriteria] = None
    target_properties: Optional[TargetProperties] = None

# Provenance information for reproducibility
@dataclass
class Provenance:
    source: str
    method: str
    doi: Optional[str] = None
    computational_cost: Optional[str] = None
    validation: Optional[str] = None
    literature_support: Optional[List[str]] = field(default_factory=list)


    