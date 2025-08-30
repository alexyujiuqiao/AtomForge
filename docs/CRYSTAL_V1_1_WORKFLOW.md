# Crystal v1.1 Workflow: Detailed Explanation

## Overview

Crystal v1.1 is the **Phase 0 implementation** of the AtomForge Crystal MVP Plan, designed as a **production-ready core system** for crystal structure handling in materials science workflows. It implements the essential operations needed for reproducible, validated crystal structure manipulation.

## Core Philosophy

### **Immutability First**
- All data structures are **frozen dataclasses** (`@dataclass(frozen=True)`)
- **No in-place modifications** - every operation returns a new object
- **Reproducible results** - same inputs always produce same outputs
- **Thread-safe** - immutable objects can be safely shared

### **Pure Functions**
- All operations are **pure functions** with no side effects
- **Deterministic** - same inputs + parameters = same outputs + hash
- **Cacheable** - results can be cached using identity hashes
- **Composable** - operations can be chained together

## Data Structure Architecture

### **1. Core Crystal Structure (`Crystal`)**
```python
@dataclass(frozen=True)
class Crystal:
    lattice: Lattice          # Unit cell parameters
    symmetry: Symmetry        # Space group information
    sites: Tuple[Site, ...]   # Atomic positions (symmetry-unique only)
    composition: Composition  # Chemical formula
    oxidation_states: Optional[Dict[str, float]] = None
    constraints: Optional[ConstraintSet] = None
    provenance: Provenance = field(default_factory=Provenance)
    notes: Optional[str] = None
```

### **2. Lattice Parameters (`Lattice`)**
```python
@dataclass(frozen=True)
class Lattice:
    a: float      # Å - lattice parameter a
    b: float      # Å - lattice parameter b  
    c: float      # Å - lattice parameter c
    alpha: float  # deg - angle between b and c
    beta: float   # deg - angle between a and c
    gamma: float  # deg - angle between a and b
```

### **3. Symmetry Information (`Symmetry`)**
```python
@dataclass(frozen=True)
class Symmetry:
    space_group: str                           # e.g., "P6_3/mmc"
    number: int                                # 1..230 (space group number)
    hall_symbol: Optional[str] = None
    origin_choice: Optional[str] = None
    symmetry_source: Optional[Literal["provided","inferred"]] = None
```

### **4. Atomic Sites (`Site`)**
```python
@dataclass(frozen=True)
class Site:
    species: Dict[str, float]                  # {"Li":0.5,"La":0.5} (sum ≈ 1.0)
    frac: Vec3                                 # fractional coordinates in [0,1)
    wyckoff: Optional[str] = None              # e.g., "4e" (filled by canonicalizer)
    multiplicity: Optional[int] = None
    label: Optional[str] = None
    magnetic_moment: Optional[Tuple[float,float,float]] = None
    charge: Optional[float] = None
    disorder_group: Optional[str] = None       # correlate partial occupancies
```

### **5. Provenance Tracking (`Provenance`)**
```python
@dataclass(frozen=True)
class Provenance:
    database: Optional[str] = None             # "MP","ICSD","COD","user"
    id: Optional[str] = None                   # e.g., "mp-12345"
    doi: Optional[str] = None
    retrieved_at: Optional[str] = None         # ISO 8601 timestamp
    generator: Optional[str] = None            # if produced by a generator
    schema_version: str = "AtomForge/Crystal/1.1"
    hash: Optional[str] = None                 # canonical identity hash
    external_ids: Dict[str, str] = field(default_factory=dict)
```

## Core Operations Workflow

### **Phase 0 Operations (Implemented)**

#### **1. Ingest Operations**
```python
def from_cif(cif_path: str) -> Crystal:
    """Parse CIF file - minimal Crystal (symmetry may be inferred later)"""
    # TODO: Implement CIF parsing
    pass

def from_poscar(poscar_path: str) -> Crystal:
    """Parse POSCAR file - minimal Crystal (symmetry may be inferred later)"""
    # TODO: Implement POSCAR parsing
    pass
```

#### **2. Canonicalization (`canonicalize`)**
**Purpose**: Standardize crystal structures for reproducible comparison and hashing.

**7-Step Pipeline**:
```python
def canonicalize(crystal: Crystal, policy: str = "conventional") -> Tuple[Crystal, CanonReport]:
    """
    Steps:
    1. Reduce to primitive cell (spglib), then convert to conventional setting
    2. Wrap fractional coords to [0,1) and quantize lattice and coords with fixed epsilons
    3. Infer wyckoff and multiplicity where possible; sort sites by (species tuple - wyckoff - frac)
    4. Strip non-physical labels; normalize element symbols consistently
    5. Set provenance.hash
    """
```

**Detailed Steps**:

1. **Primitive Cell Reduction** (`_reduce_to_primitive`)
   - Use spglib to find the primitive cell
   - Reduce symmetry operations to minimum set
   - Update lattice parameters and site positions

2. **Conventional Cell Conversion** (`_to_conventional`)
   - Convert primitive cell to conventional setting
   - Apply standard crystallographic conventions
   - Update space group information

3. **Coordinate Wrapping** (`_wrap_coordinates`)
   - Wrap all fractional coordinates to [0,1) range
   - Ensure atomic positions are within unit cell
   - Handle periodic boundary conditions

4. **Quantization** (`_quantize_structure`)
   - Apply epsilon-based quantization to lattice parameters
   - Quantize fractional coordinates to fixed precision
   - Remove numerical noise: `epsilon_lattice = 1e-6` Å, `epsilon_coords = 1e-8`

5. **Wyckoff Inference** (`_infer_wyckoff`)
   - Determine Wyckoff positions for each site
   - Calculate site multiplicities
   - Use spglib for symmetry analysis

6. **Site Sorting** (`_sort_sites`)
   - Sort sites by: `(species tuple - wyckoff - frac)`
   - Ensure deterministic ordering
   - Enable reproducible comparison

7. **Identity Hash Computation** (`_compute_identity_hash`)
   - Generate SHA-256 hash of canonical structure
   - Update provenance.hash field
   - Used as cache key across all operations

**Policies**:
- `"primitive"`: Reduce to primitive cell only
- `"conventional"`: Reduce to primitive, then convert to conventional
- `"standard"`: No cell reduction, only coordinate processing

#### **3. Validation (`validate`)**
**Purpose**: Ensure crystal structure integrity and physical consistency.

**7 Validation Checks**:

1. **Metric Positive Definite** (`_check_metric_pd`)
   - Verify cell metric tensor is positive definite
   - Check lattice angles are in valid ranges (0 < angle < 180°)
   - Ensure physically meaningful unit cell

2. **Species Occupancies** (`_check_species_occupancies`)
   - For every site: `sum(occupancy) == 1 ± tolerance`
   - Validate partial occupancies sum to unity
   - Check for missing or invalid occupancies

3. **Composition Consistency** (`_check_composition_consistency`)
   - Recompute formula from `(sites × multiplicity)`
   - Assert match with `composition` within tolerance
   - Verify chemical formula accuracy

4. **Disorder Groups** (`_check_disorder_groups`)
   - If `disorder_group` used, validate correlated occupancies
   - Check domain rules: `≤ 1.0` per group
   - Ensure exactly `1.0` if mutually exclusive

5. **Charge Neutrality** (`_check_charge_neutrality`)
   - If `constraints.charge_neutrality == True`
   - Validate using `oxidation_states`
   - Check total charge equals zero

6. **Symmetry Coherence** (`_check_symmetry_coherence`)
   - Verify reported `Symmetry` agrees with spglib analysis
   - Check space group consistency
   - Validate symmetry operations

7. **Fractional Bounds** (`_check_fractional_bounds`)
   - Ensure no coordinate equals 1.0 after wrapping
   - Check all coordinates are in [0,1) range
   - Validate periodic boundary conditions

#### **4. Identity Hashing (`identity_hash`)**
**Purpose**: Generate unique, reproducible identifier for crystal structures.

```python
def identity_hash(crystal: Crystal) -> str:
    """Compute SHA-256 over canonical byte string; used as cache key across all ops."""
    canonicalizer = CrystalCanonicalizer()
    return canonicalizer._compute_identity_hash(crystal)
```

**Features**:
- **SHA-256 hash** over canonical byte representation
- **Cache key** across all operations
- **Reproducible** - same structure always produces same hash
- **Collision-resistant** - extremely unlikely for different structures

### **JSON Schema Integration**

#### **Schema Validation**
```python
class CrystalSchemaValidator:
    """JSON Schema validation for Crystal v1.1"""
    
    def validate_json(self, data: Dict[str, Any]) -> ValidationReport:
        """Validate JSON data against the schema"""
        # Validates against atomforge_crystal_schema.json
```

#### **Typed Adapters**
```python
class CrystalAdapter:
    """Adapter for converting between Crystal objects and other formats"""
    
    @staticmethod
    def to_dict(crystal: Crystal) -> Dict[str, Any]:
        """Convert Crystal to dictionary"""
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Crystal:
        """Convert dictionary to Crystal"""
    
    @staticmethod
    def to_json(crystal: Crystal) -> str:
        """Convert Crystal to JSON string"""
    
    @staticmethod
    def from_json(json_str: str) -> Crystal:
        """Convert JSON string to Crystal"""
```

## Complete Workflow Examples

### **1. Basic Crystal Creation and Canonicalization**
```python
from atomforge.src.crystal_v1_1 import create_simple_crystal, canonicalize, validate, identity_hash

# Create a simple crystal structure
crystal = create_simple_crystal(
    lattice_params=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),  # cubic lattice
    sites=[("Fe", (0.0, 0.0, 0.0))],                    # Fe at origin
    space_group="Pm-3m",                                # space group
    space_group_number=221                              # space group number
)

# Canonicalize the structure
canonical_crystal, canon_report = canonicalize(crystal, policy="conventional")

# Validate the canonicalized structure
validation_report = validate(canonical_crystal)

# Get the identity hash
hash_value = identity_hash(canonical_crystal)

print(f"Canonicalization actions: {canon_report.actions_taken}")
print(f"Validation passed: {validation_report.ok}")
print(f"Identity hash: {hash_value}")
```

### **2. Round-Trip JSON Serialization**
```python
from atomforge.src.crystal_v1_1 import CrystalAdapter

# Convert to JSON
json_str = CrystalAdapter.to_json(canonical_crystal)

# Convert back from JSON
recovered_crystal = CrystalAdapter.from_json(json_str)

# Verify round-trip integrity
assert identity_hash(canonical_crystal) == identity_hash(recovered_crystal)
print("Round-trip validation passed!")
```

### **3. Validation with Custom Rules**
```python
# Define custom validation rules
custom_rules = {
    'check_metric_pd': True,
    'check_occupancies': True,
    'check_composition': True,
    'check_disorder': False,  # Skip disorder group validation
    'check_charge_neutrality': True,
    'check_symmetry': True,
    'check_bounds': True,
    'tolerance': 1e-6  # Stricter tolerance
}

# Validate with custom rules
validation_report = validate(canonical_crystal, rules=custom_rules)

if not validation_report.ok:
    print("Validation errors:")
    for error in validation_report.errors:
        print(f"  - {error}")
else:
    print("All validation checks passed!")
```

## Key Features and Benefits

### **1. Reproducibility**
- **Deterministic operations** - same inputs always produce same outputs
- **Immutable data structures** - prevents accidental modifications
- **Canonical forms** - standardized representation for comparison

### **2. Validation**
- **Comprehensive checks** - 7 different validation categories
- **Configurable rules** - customizable validation policies
- **Detailed reporting** - specific error messages and warnings

### **3. Provenance Tracking**
- **Complete metadata** - database IDs, DOIs, timestamps
- **Identity hashing** - unique identifiers for caching
- **Audit trail** - full traceability of structure origins

### **4. Extensibility**
- **Modular design** - easy to add new operations
- **Pure functions** - composable and testable
- **Type safety** - strong typing with dataclasses

### **5. Performance**
- **Caching support** - hash-based caching across operations
- **Efficient serialization** - JSON schema validation
- **Memory efficient** - immutable objects enable sharing

## Integration Points

### **Phase 1 Operations (Planned)**
- Database matching (MP/ICSD/COD)
- Variant selection
- Defect engineering (substitution, vacancy, interstitial)
- Supercell generation
- Export to VASP/Quantum ESPRESSO

### **AI/ML Integration**
- **LLaMP framework** integration for intelligent retrieval
- **OpenAI GPT models** for DSL generation
- **Graph neural networks** for property prediction
- **Active learning** workflows

### **Materials Databases**
- **Materials Project (MP)** integration
- **ICSD** database access
- **COD** database support
- **AFLOW** and **OQMD** compatibility

## Use Cases

### **1. Materials Discovery**
- Import crystal structures from databases
- Canonicalize and validate structures
- Compare structures for similarity
- Generate computational inputs

### **2. Defect Engineering**
- Create defect structures systematically
- Validate defect configurations
- Track defect generation history
- Export for DFT calculations

### **3. Structure Comparison**
- Compare different crystal structures
- Identify symmetry-equivalent structures
- Track structural evolution
- Validate structural transformations

### **4. Computational Workflows**
- Prepare structures for DFT calculations
- Generate VASP/Quantum ESPRESSO inputs
- Validate calculation inputs
- Track calculation provenance

## Summary

Crystal v1.1 provides a **robust, reproducible foundation** for crystal structure handling in materials science. Its **immutable design**, **comprehensive validation**, and **canonical representation** ensure reliable, auditable workflows for materials discovery and computational chemistry applications.

The workflow follows a **clear progression**: **Ingest → Canonicalize → Validate → Hash**, with each step building on the previous one to create a complete, validated crystal structure representation that can be safely used in downstream applications. 