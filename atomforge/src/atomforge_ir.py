from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod
import sys
import os

# Add core to path for inheritance
core_path = os.path.join(os.path.dirname(__file__), '..', '..', 'core')
if core_path not in sys.path:
    sys.path.append(core_path)

try:
    from dataclass import DSLProgram
except ImportError:
    # Fallback: create a simple base class if core module not available
    class DSLProgram:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        def validate(self):
            pass

"""
AtomForge DSL v2.1 Intermediate Representation (IR)

This module defines the complete data structures for AtomForge DSL v2.1,
extending the core DSLProgram pattern and supporting all advanced features:
- AI integration and machine learning
- Emerging materials (2D, topological, metamaterials)
- Revolutionary patching system
- Comprehensive benchmarking
- Advanced type system
- Procedural generation
"""

# ============================================================================
# CORE ATOMFORGE V2.1 PROGRAM STRUCTURE
# ============================================================================

@dataclass
class AtomForgeProgram(DSLProgram):
    """
    Complete AtomForge DSL v2.1 program representation.
    Extends the core DSLProgram and implements validation.
    """
    identifier: str
    header: 'Header'
    lattice: 'Lattice'
    symmetry: 'Symmetry'
    basis: 'Basis'
    description: Optional[str] = None
    units: Optional['Units'] = None
    type_system: Optional['TypeSystem'] = None
    emerging_materials: Optional['EmergingMaterials'] = None
    defects: Optional['Defects'] = None
    tile: Optional['Tile'] = None
    bonds: Optional['Bonds'] = None
    elastic: Optional['Elastic'] = None
    phonon: Optional['Phonon'] = None
    density: Optional['Density'] = None
    environment: Optional['Environment'] = None
    ai_integration: Optional['AIIntegration'] = None
    procedural_generation: Optional['ProceduralGeneration'] = None
    benchmarking: Optional['Benchmarking'] = None
    properties: Optional['Properties'] = None
    validation: Optional['Validation'] = None
    simplification: Optional['Simplification'] = None
    provenance: Optional['Provenance'] = None
    patch: Optional['Patch'] = None
    meta: Optional['Meta'] = None

    def validate(self) -> None:
        """Validate the integrity of the AtomForge v2.1 program"""
        # Validate required components
        if not self.header:
            raise ValueError("Header is required")
        if not self.lattice:
            raise ValueError("Lattice is required")
        if not self.symmetry:
            raise ValueError("Symmetry is required")
        if not self.basis:
            raise ValueError("Basis is required")
        
        # Validate header components
        self.header.validate()
        
        # Validate lattice and symmetry consistency
        self.lattice.validate()
        self.symmetry.validate()
        
        # Validate basis
        self.basis.validate()
        
        # Validate optional components if present
        if self.units:
            self.units.validate()
        if self.type_system:
            self.type_system.validate()
        if self.emerging_materials:
            self.emerging_materials.validate()
        if self.ai_integration:
            self.ai_integration.validate()
        if self.procedural_generation:
            self.procedural_generation.validate()
        if self.benchmarking:
            self.benchmarking.validate()
        if self.patch:
            self.patch.validate()

# ============================================================================
# HEADER & METADATA
# ============================================================================

@dataclass
class Header:
    """Enhanced header with versioning and UUID support"""
    dsl_version: str
    title: str
    created: datetime
    content_schema_version: Optional[str] = None
    uuid: Optional[str] = None
    modified: Optional[datetime] = None

    def validate(self) -> None:
        if not self.dsl_version:
            raise ValueError("DSL version is required")
        if not self.title:
            raise ValueError("Title is required")
        if not self.created:
            raise ValueError("Created date is required")

@dataclass
class Units:
    """Enhanced units system with dimensional analysis"""
    system: str
    length: str
    angle: str
    disp: str
    temp: str
    pressure: str

    def validate(self) -> None:
        # Accept both legacy and current spellings for crystallography units.
        valid_systems = [
            "crystallography_default",
            "crystallographic_default",
            "SI",
            "atomic",
            "custom",
        ]
        if self.system not in valid_systems:
            raise ValueError(f"Invalid unit system: {self.system}")

# ============================================================================
# ADVANCED TYPE SYSTEM
# ============================================================================

@dataclass
class TypeSystem:
    """Advanced type system for structural, chemical, and computational compatibility"""
    structural_types: Optional['StructuralTypes'] = None
    chemical_types: Optional['ChemicalTypes'] = None
    computational_types: Optional['ComputationalTypes'] = None
    auto_inference: Optional['AutoInference'] = None

    def validate(self) -> None:
        if self.structural_types:
            self.structural_types.validate()
        if self.chemical_types:
            self.chemical_types.validate()
        if self.computational_types:
            self.computational_types.validate()
        if self.auto_inference:
            self.auto_inference.validate()

@dataclass
class StructuralTypes:
    """Structural type specifications"""
    coordination_environment: Optional[str] = None
    crystal_system: Optional[str] = None
    space_group_family: Optional[str] = None
    connectivity: Optional[str] = None
    compatibility_rules: List[str] = field(default_factory=list)

    def validate(self) -> None:
        valid_environments = ["tetrahedral", "octahedral", "square_planar", "trigonal_planar"]
        if self.coordination_environment and self.coordination_environment not in valid_environments:
            raise ValueError(f"Invalid coordination environment: {self.coordination_environment}")

@dataclass
class ChemicalTypes:
    """Chemical type specifications"""
    element_category: Optional[str] = None
    oxidation_states: List[int] = field(default_factory=list)
    coordination_preference: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)

    def validate(self) -> None:
        pass

@dataclass
class ComputationalTypes:
    """Computational type specifications"""
    calculation_method: Optional[str] = None
    basis_requirements: Optional[str] = None
    k_point_density: Optional[str] = None
    resource_requirements: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        pass

@dataclass
class AutoInference:
    """Automatic type inference settings"""
    enabled: bool
    inference_methods: List[str] = field(default_factory=list)
    validation_strictness: str = "moderate"  # strict, moderate, permissive
    error_handling: str = "warn_and_continue"  # error, warn_and_continue, silent

    def validate(self) -> None:
        valid_strictness = ["strict", "moderate", "permissive"]
        if self.validation_strictness not in valid_strictness:
            raise ValueError(f"Invalid validation strictness: {self.validation_strictness}")

# ============================================================================
# LATTICE & SYMMETRY
# ============================================================================

@dataclass
class Lattice:
    """Enhanced lattice with Bravais and vector formats"""
    description: Optional[str] = None
    bravais: Optional['Bravais'] = None
    vectors: Optional[List[Tuple[float, float, float]]] = None

    def validate(self) -> None:
        if not self.bravais and not self.vectors:
            raise ValueError("Lattice must specify either Bravais parameters or vectors")
        if self.bravais:
            self.bravais.validate()
        if self.vectors and len(self.vectors) != 3:
            raise ValueError("Lattice vectors must have exactly 3 vectors")

@dataclass
class Bravais:
    """Bravais lattice parameters"""
    type: str
    a: 'Length'
    b: 'Length'
    c: 'Length'
    alpha: 'Angle'
    beta: 'Angle'
    gamma: 'Angle'

    def validate(self) -> None:
        valid_types = ["cubic", "tetragonal", "orthorhombic", "hexagonal", 
                      "rhombohedral", "monoclinic", "triclinic"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid lattice type: {self.type}")

@dataclass
class Symmetry:
    """Enhanced symmetry with magnetic and superspace support"""
    space_group: Union[int, str]
    origin_choice: int
    description: Optional[str] = None
    magnetic_group: Optional[str] = None
    superspace: Optional['Superspace'] = None

    def validate(self) -> None:
        if isinstance(self.space_group, int) and (self.space_group < 1 or self.space_group > 230):
            raise ValueError(f"Invalid space group number: {self.space_group}")

@dataclass
class Superspace:
    """Superspace structure specifications"""
    k_vectors: List[Tuple[float, float, float]]
    t_phase: float

    def validate(self) -> None:
        pass

# ============================================================================
# ATOMIC BASIS
# ============================================================================

@dataclass
class Basis:
    """Enhanced atomic basis with comprehensive site specification"""
    description: Optional[str] = None
    sites: List['Site'] = field(default_factory=list)

    def validate(self) -> None:
        if not self.sites:
            raise ValueError("Basis must contain at least one site")
        for site in self.sites:
            site.validate()

@dataclass
class Site:
    """Enhanced atomic site with comprehensive metadata"""
    name: str
    wyckoff: str
    position: Tuple[float, float, float]
    frame: str
    species: List['Species']
    description: Optional[str] = None
    moment: Optional[Tuple[float, float, float]] = None
    constraint: Optional[str] = None
    adp_iso: Optional[float] = None
    adp_aniso: Optional[Dict[str, float]] = None
    label: Optional[str] = None

    def validate(self) -> None:
        if not self.species:
            raise ValueError("Site must have at least one species")
        if self.frame not in ["fractional", "cartesian"]:
            raise ValueError(f"Invalid frame: {self.frame}")
        for species in self.species:
            species.validate()

@dataclass
class Species:
    """Enhanced species with comprehensive metadata"""
    element: str
    occupancy: float
    isotope: Optional[int] = None
    charge: Optional[float] = None
    valence: Optional[int] = None

    def validate(self) -> None:
        if self.occupancy < 0.0 or self.occupancy > 1.0:
            raise ValueError(f"Invalid occupancy: {self.occupancy}")

# ============================================================================
# EMERGING MATERIALS
# ============================================================================

@dataclass
class EmergingMaterials:
    """Support for emerging materials classes"""
    type: str
    layer_stack: Optional['LayerStack'] = None
    topology: Optional['Topology'] = None
    metamaterial: Optional['Metamaterial'] = None
    quantum_state: Optional['QuantumState'] = None

    def validate(self) -> None:
        valid_types = ["2D_heterostructure", "topological_insulator", "topological_semimetal",
                      "metamaterial", "quantum_material", "van_der_waals", "moire_system"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid emerging materials type: {self.type}")

@dataclass
class LayerStack:
    """2D materials layer stacking"""
    layers: List['LayerSpec']
    interlayer_distance: 'Length'
    coupling_strength: str

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("Layer stack must have at least one layer")

@dataclass
class LayerSpec:
    """Individual layer specification"""
    material: str
    twist_angle: 'Angle'
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Topology:
    """Topological materials specifications"""
    classification: str
    z2_invariant: str
    surface_states: str
    bulk_gap: float

@dataclass
class Metamaterial:
    """Metamaterial specifications"""
    metamaterial_type: str
    unit_cell_period: Tuple[float, float, float]
    effective_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QuantumState:
    """Quantum materials specifications"""
    quantum_state: str
    frustration: Dict[str, Any] = field(default_factory=dict)
    entanglement: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# AI INTEGRATION
# ============================================================================

@dataclass
class AIIntegration:
    """AI-native integration constructs"""
    graph_representation: Optional['GraphRepresentation'] = None
    generation_model: Optional['GenerationModel'] = None
    active_learning: Optional['ActiveLearning'] = None
    multi_fidelity: Optional['MultiFidelity'] = None

    def validate(self) -> None:
        if self.graph_representation:
            self.graph_representation.validate()
        if self.generation_model:
            self.generation_model.validate()
        if self.active_learning:
            self.active_learning.validate()
        if self.multi_fidelity:
            self.multi_fidelity.validate()

@dataclass
class GraphRepresentation:
    """Graph neural network representation"""
    node_features: List[str]
    edge_features: List[str]
    global_features: List[str]

    def validate(self) -> None:
        if not self.node_features:
            raise ValueError("Graph representation must have node features")

@dataclass
class GenerationModel:
    """Generative model integration"""
    type: str
    latent_dim: int
    constraints: List[str]
    sampling_temperature: float

    def validate(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("Latent dimension must be positive")

@dataclass
class ActiveLearning:
    """Active learning workflow"""
    acquisition_function: str
    surrogate_model: str
    exploration_weight: float
    batch_size: int

    def validate(self) -> None:
        if self.exploration_weight < 0.0 or self.exploration_weight > 1.0:
            raise ValueError("Exploration weight must be between 0 and 1")

@dataclass
class MultiFidelity:
    """Multi-fidelity learning"""
    low_fidelity: str
    high_fidelity: str
    correlation_model: str
    cost_ratio: float

    def validate(self) -> None:
        if self.cost_ratio <= 0:
            raise ValueError("Cost ratio must be positive")

# ============================================================================
# PROCEDURAL GENERATION
# ============================================================================

@dataclass
class ProceduralGeneration:
    """Procedural generation system"""
    generator_type: str
    template_generation: Optional['TemplateGeneration'] = None
    parameter_sweep: Optional['ParameterSweep'] = None
    ml_guided: Optional['MLGuided'] = None
    hybridization: Optional['Hybridization'] = None

    def validate(self) -> None:
        valid_types = ["template_based", "parameter_sweep", "combinatorial", "ml_guided", "hybridization"]
        if self.generator_type not in valid_types:
            raise ValueError(f"Invalid generator type: {self.generator_type}")

@dataclass
class TemplateGeneration:
    """Template-based generation"""
    template: str
    parameter_space: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)

@dataclass
class ParameterSweep:
    """Parameter sweep generation"""
    base_structure: str
    generation_mode: str
    sweep_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MLGuided:
    """ML-guided generation"""
    generation_method: str
    population_size: int
    generations: int
    target_properties: Dict[str, str] = field(default_factory=dict)

@dataclass
class Hybridization:
    """Hybridization system"""
    mutation_rate: float
    fitness_function: str
    parent_selection: Dict[str, Any] = field(default_factory=dict)
    crossover_operations: List[str] = field(default_factory=list)

# ============================================================================
# BENCHMARKING
# ============================================================================

@dataclass
class Benchmarking:
    """Comprehensive benchmarking system"""
    benchmark_type: str
    tasks: List['BenchmarkTask'] = field(default_factory=list)

    def validate(self) -> None:
        valid_types = ["structure_reconstruction", "property_prediction", "inverse_design",
                      "materials_understanding", "multi_modal", "comprehensive"]
        if self.benchmark_type not in valid_types:
            raise ValueError(f"Invalid benchmark type: {self.benchmark_type}")

@dataclass
class BenchmarkTask:
    """Individual benchmark task"""
    input_modality: Optional[List[str]] = None
    input_data: Optional[str] = None
    target_output: Optional[str] = None
    target_properties: Optional[Dict[str, Any]] = None
    difficulty_level: Optional[str] = None
    evaluation_metrics: Optional[List[str]] = None
    constraints: Optional[List[str]] = None

# ============================================================================
# REVOLUTIONARY PATCHING SYSTEM
# ============================================================================

@dataclass
class Patch:
    """Revolutionary patching system"""
    operations: List['PatchOperation'] = field(default_factory=list)

    def validate(self) -> None:
        for operation in self.operations:
            operation.validate()

@dataclass
class PatchOperation:
    """Individual patch operation"""
    type: str  # "add", "remove", "update"
    path: Optional[str] = None
    value: Optional[Any] = None
    site: Optional[Site] = None

    def validate(self) -> None:
        valid_types = ["add", "remove", "update"]
        if self.type not in valid_types:
            raise ValueError(f"Invalid patch operation type: {self.type}")

# ============================================================================
# SUPPORTING DATA TYPES
# ============================================================================

@dataclass
class Length:
    """Length with units"""
    value: float
    unit: Optional[str] = None

@dataclass
class Angle:
    """Angle with units"""
    value: float
    unit: Optional[str] = None

@dataclass
class Temperature:
    """Temperature with units"""
    value: float
    unit: Optional[str] = None

@dataclass
class Pressure:
    """Pressure with units"""
    value: float
    unit: Optional[str] = None

@dataclass
class Displacement:
    """Displacement with units"""
    value: float
    unit: Optional[str] = None

# ============================================================================
# ADDITIONAL COMPONENTS
# ============================================================================

@dataclass
class Defects:
    """Defect modeling"""
    defects: List['DefectEntry'] = field(default_factory=list)

@dataclass
class DefectEntry:
    """Individual defect entry"""
    site_ref: str
    type: str  # "vacancy", "interstitial", "substitution"
    prob: float
    species: Optional[Species] = None

@dataclass
class Tile:
    """Transformations and supercells"""
    repeat: Tuple[int, int, int]
    origin_shift: Optional[Tuple[float, float, float]] = None
    transforms: Optional[List[str]] = None

@dataclass
class Bonds:
    """Chemical bonding"""
    bonds: List['Bond'] = field(default_factory=list)

@dataclass
class Bond:
    """Individual bond"""
    site1: str
    site2: str
    length: Length

@dataclass
class Elastic:
    """Elastic properties"""
    c_ijkl: List[float] = field(default_factory=list)

@dataclass
class Phonon:
    """Phonon properties"""
    q_grid: Tuple[int, int, int]
    frequencies: List[float] = field(default_factory=list)

@dataclass
class Density:
    """Density maps and volumetric data"""
    grid: Tuple[int, int, int]
    format: str
    data: Optional[str] = None
    data_file: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Environment:
    """Environmental conditions"""
    temperature: Optional[Temperature] = None
    pressure: Optional[Pressure] = None
    e_field: Optional[Tuple[float, float, float]] = None
    e_grad: Optional[List[List[float]]] = None  # 3x3 matrix
    b_field: Optional[Tuple[float, float, float]] = None

@dataclass
class Properties:
    """Extensible properties"""
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Validation:
    """Validation settings"""
    tolerance: float
    occupancy_clamp: bool
    vector_unit_consistent: bool
    max_transform_depth: int
    enforce_units: bool

@dataclass
class Simplification:
    """Complexity management"""
    complexity_level: str
    auto_complete: bool
    suggest_defaults: bool
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Provenance:
    """Enhanced provenance tracking"""
    source: str
    method: str
    doi: str
    url: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Meta:
    """Extensible metadata"""
    metadata: Dict[str, Any] = field(default_factory=dict) 