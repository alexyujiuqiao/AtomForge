#!/usr/bin/env python3
"""
AtomForge Interop Module - Phase 1 Implementation

This module implements the interop functionality for Phase 1:
- from_cif / from_poscar parsing
- match_database and select_variant functions
- Enhanced database matching with provenance
- Variant selection with policies
- UI-ready variant cards with space group, energy hull, site count

Based on Phase 1 requirements from plan-v-2-0.tex and operations-and-workflows-v-2-0.tex
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

# Import database connectors
from atomforge_database_connector import (
    MaterialsProjectConnector, CODConnector, ICSDConnector,
    DatabaseMatch, MatchReport, SelectionReport
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Phase 1 Data Structures
# ============================================================================

@dataclass
class Crystal:
    """Basic crystal structure representation for Phase 1"""
    formula: str
    structure_data: Dict[str, Any]
    provenance: Dict[str, Any]
    metadata: Dict[str, Any]
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Crystal':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class VariantCard:
    """UI-ready variant card with space group, energy hull, site count"""
    database_name: str
    material_id: str
    formula: str
    space_group: Optional[str]
    energy_hull: Optional[float]
    site_count: Optional[int]
    is_experimental: bool
    is_stable: Optional[bool]
    provenance: Dict[str, Any]
    properties: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI display"""
        return asdict(self)
    
    def get_display_summary(self) -> str:
        """Get human-readable summary for UI"""
        parts = [f"{self.database_name}:{self.material_id}"]
        if self.space_group:
            parts.append(f"SG: {self.space_group}")
        if self.energy_hull is not None:
            parts.append(f"E_hull: {self.energy_hull:.3f}")
        if self.site_count:
            parts.append(f"Sites: {self.site_count}")
        if self.is_experimental:
            parts.append("Exp")
        if self.is_stable:
            parts.append("Stable")
        return " | ".join(parts)

# ============================================================================
# Phase 1 Core Functions
# ============================================================================

def from_cif(file_path: str) -> 'CrystalV1_1':
    """
    Parse CIF file using crystal v1.1 schema and SpaceGroupAnalyzer.
    
    Args:
        file_path: Path to CIF file
    
    Returns:
        Crystal v1.1 object with proper symmetry analysis
    """
    try:
        from crystal_v1_1 import from_cif as crystal_from_cif
        crystal = crystal_from_cif(file_path)
        logger.info(f"Successfully parsed CIF file: {file_path}")
        logger.info(f"Space group: {crystal.symmetry.space_group} ({crystal.symmetry.number})")
        logger.info(f"Formula: {crystal.composition.reduced}")
        return crystal
        
    except Exception as e:
        logger.error(f"Error parsing CIF file {file_path}: {e}")
        raise

def from_poscar(file_path: str) -> 'CrystalV1_1':
    """
    Parse POSCAR file using crystal v1.1 schema and SpaceGroupAnalyzer.
    
    Args:
        file_path: Path to POSCAR file
    
    Returns:
        Crystal v1.1 object with proper symmetry analysis
    """
    try:
        from crystal_v1_1 import from_poscar as crystal_from_poscar
        crystal = crystal_from_poscar(file_path)
        logger.info(f"Successfully parsed POSCAR file: {file_path}")
        logger.info(f"Space group: {crystal.symmetry.space_group} ({crystal.symmetry.number})")
        logger.info(f"Formula: {crystal.composition.reduced}")
        return crystal
        
    except Exception as e:
        logger.error(f"Error parsing POSCAR file {file_path}: {e}")
        raise

def match_database(crystal_formula: str, sources: List[str] = None, 
                  tolerance: float = 0.1) -> MatchReport:
    """
    Find similar structures in MP/ICSD/COD with provenance and similarity score.
    
    Args:
        crystal_formula: Chemical formula to search for
        sources: List of databases to search (default: ["MP", "COD"])
        tolerance: Similarity tolerance (not used in current implementation)
    
    Returns:
        MatchReport with all matches found
    """
    if sources is None:
        sources = ["MP", "COD"]
    
    start_time = time.time()
    matches_found = {}
    errors = []
    
    # Initialize connectors
    connectors = {}
    if "MP" in sources:
        try:
            connectors["MP"] = MaterialsProjectConnector("MP_API_KEY_REDACTED")
        except Exception as e:
            errors.append(f"Failed to initialize MP connector: {e}")
    
    if "COD" in sources:
        try:
            connectors["COD"] = CODConnector()
        except Exception as e:
            errors.append(f"Failed to initialize COD connector: {e}")
    
    if "ICSD" in sources:
        try:
            connectors["ICSD"] = ICSDConnector()  # No auth for demo
        except Exception as e:
            errors.append(f"Failed to initialize ICSD connector: {e}")
    
    # Search each database
    for db_name, connector in connectors.items():
        try:
            matches = connector.search_by_formula(crystal_formula, limit=10)
            matches_found[db_name] = matches
            logger.info(f"Found {len(matches)} matches in {db_name}")
            
            # Add small delay to be respectful to APIs
            time.sleep(0.1)
            
        except Exception as e:
            error_msg = f"Error searching {db_name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            matches_found[db_name] = []
    
    total_matches = sum(len(matches) for matches in matches_found.values())
    search_time = time.time() - start_time
    
    return MatchReport(
        query_formula=crystal_formula,
        matches_found=matches_found,
        total_matches=total_matches,
        search_time=search_time,
        databases_searched=sources,
        errors=errors
    )

def select_variant(candidates: List[DatabaseMatch], 
                  policy: str = "prefer_low_hull_then_experimental",
                  reference_space_group: Optional[Union[str, int]] = None) -> Tuple[DatabaseMatch, SelectionReport]:
    """
    Choose best variant by policy: energy_hull, experimental, completeness, explicit_space_group.
    
    Args:
        candidates: List of DatabaseMatch objects to choose from
        policy: Selection policy to use
    
    Returns:
        Tuple of (selected_variant, SelectionReport)
    """
    if not candidates:
        raise ValueError("No candidates provided for variant selection")
    
    # Flatten candidates from all databases
    all_candidates = []
    for db_matches in candidates if isinstance(candidates[0], list) else [candidates]:
        all_candidates.extend(db_matches)
    
    if not all_candidates:
        raise ValueError("No valid candidates found")
    
    # Apply selection policy
    if policy == "low_hull_experimental_completeness_explicit_space_group":
        selected = _select_by_comprehensive_policy(all_candidates)
    elif policy == "prefer_low_hull_then_experimental":
        selected = _select_by_energy_hull_then_experimental(all_candidates)
    elif policy == "prefer_experimental":
        selected = _select_by_experimental(all_candidates)
    elif policy == "prefer_completeness":
        selected = _select_by_completeness(all_candidates)
    elif policy == "prefer_explicit_space_group":
        selected = _select_by_explicit_space_group(all_candidates, reference_space_group)
    else:
        # Default: first candidate
        selected = all_candidates[0]
    
    # Create selection report
    report = SelectionReport(
        selected_variant=selected,
        ranking_criteria=_get_ranking_criteria(policy),
        policy_used=policy,
        alternatives_considered=all_candidates,
        selection_reason=_get_selection_reason(selected, policy),
        timestamp=datetime.now()
    )
    
    logger.info(f"Selected variant: {selected.database_name}:{selected.material_id} using policy: {policy}")
    return selected, report

def match_database_with_provenance(crystal, sources: List[str] = None, 
                                 tolerance: float = 0.1) -> Tuple[MatchReport, List[VariantCard]]:
    """
    Find similar structures in MP/ICSD/COD with provenance and create variant cards.
    
    Args:
        crystal: Crystal v1.1 object to match
        sources: List of databases to search
        tolerance: Similarity tolerance
    
    Returns:
        Tuple of (MatchReport, List[VariantCard])
    """
    # Extract formula from crystal v1.1 composition
    if hasattr(crystal, 'composition') and hasattr(crystal.composition, 'reduced'):
        # Convert reduced formula dict to string
        formula_parts = []
        for element, count in crystal.composition.reduced.items():
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        formula = "".join(formula_parts)
    else:
        # Fallback for old Crystal objects
        formula = getattr(crystal, 'formula', 'Unknown')
    
    # Use the match_database function
    match_report = match_database(formula, sources, tolerance)
    
    # Convert matches to variant cards
    variant_cards = []
    for db_name, matches in match_report.matches_found.items():
        for match in matches:
            card = VariantCard(
                database_name=match.database_name,
                material_id=match.material_id,
                formula=match.formula,
                space_group=match.properties.get('space_group'),
                energy_hull=match.properties.get('energy_hull'),
                site_count=match.metadata.get('nsites'),
                is_experimental=match.metadata.get('is_experimental', False),
                is_stable=match.properties.get('is_stable'),
                provenance=match.provenance,
                properties=match.properties,
                metadata=match.metadata
            )
            variant_cards.append(card)
    
    logger.info(f"Created {len(variant_cards)} variant cards from {match_report.total_matches} matches")
    return match_report, variant_cards

def select_variant_with_policy(variant_cards: List[VariantCard], 
                              policy: str = "prefer_low_hull_then_experimental",
                              reference_space_group: Optional[Union[str, int]] = None) -> Tuple[VariantCard, SelectionReport]:
    """
    Choose best variant by policy and return variant card with selection report.
    
    Args:
        variant_cards: List of VariantCard objects
        policy: Selection policy to use
    
    Returns:
        Tuple of (selected_variant_card, SelectionReport)
    """
    if not variant_cards:
        raise ValueError("No variant cards provided for selection")
    
    # Convert variant cards to database matches for selection
    candidates = []
    for card in variant_cards:
        match = DatabaseMatch(
            database_name=card.database_name,
            material_id=card.material_id,
            formula=card.formula,
            similarity_score=1.0,
            provenance=card.provenance,
            properties=card.properties,
            metadata=card.metadata
        )
        candidates.append(match)
    
    # Use the select_variant function
    selected_match, report = select_variant(candidates, policy, reference_space_group)
    
    # Find the corresponding variant card
    selected_card = None
    for card in variant_cards:
        if (card.database_name == selected_match.database_name and 
            card.material_id == selected_match.material_id):
            selected_card = card
            break
    
    if selected_card is None:
        raise ValueError("Selected match not found in variant cards")
    
    logger.info(f"Selected variant: {selected_card.get_display_summary()}")
    return selected_card, report

def create_variant_cards_ui(variant_cards: List[VariantCard], query_formula: str = None) -> List[Dict[str, Any]]:
    """
    Create UI-ready variant cards with space group, energy hull, site count, and similarity scores.
    
    Args:
        variant_cards: List of VariantCard objects
        query_formula: Original formula for similarity calculation
    
    Returns:
        List of dictionaries ready for UI display with similarity scores
    """
    ui_cards = []
    for card in variant_cards:
        # Calculate similarity score
        similarity_score = 0.0
        if query_formula:
            similarity_score = calculate_similarity_score(query_formula, card.formula, card.properties)
        
        ui_card = {
            "id": f"{card.database_name}:{card.material_id}",
            "database": card.database_name,
            "material_id": card.material_id,
            "formula": card.formula,
            "similarity_score": similarity_score,
            "display_summary": card.get_display_summary(),
            "space_group": card.space_group,
            "energy_hull": card.energy_hull,
            "site_count": card.site_count,
            "is_experimental": card.is_experimental,
            "is_stable": card.is_stable,
            "provenance": card.provenance,
            "properties": card.properties,
            "metadata": card.metadata
        }
        ui_cards.append(ui_card)
    
    # Sort by similarity score (highest first), then by energy hull, then by experimental status
    ui_cards.sort(key=lambda x: (
        -x["similarity_score"],  # Negative for descending order
        x["energy_hull"] if x["energy_hull"] is not None else float('inf'),
        not x["is_experimental"]
    ))
    
    return ui_cards

# ============================================================================
# Selection Policy Helper Functions
# ============================================================================

def _select_by_energy_hull_then_experimental(candidates: List[DatabaseMatch]) -> DatabaseMatch:
    """Select by lowest energy hull, then prefer experimental"""
    # Filter candidates with energy hull data
    hull_candidates = [c for c in candidates if c.properties.get('energy_hull') is not None]
    
    if hull_candidates:
        # Sort by energy hull (lower is better)
        hull_candidates.sort(key=lambda x: x.properties.get('energy_hull', float('inf')))
        
        # Among candidates with same energy hull, prefer experimental
        best_hull = hull_candidates[0].properties.get('energy_hull')
        same_hull = [c for c in hull_candidates if c.properties.get('energy_hull') == best_hull]
        
        # Prefer experimental over theoretical
        experimental = [c for c in same_hull if c.metadata.get('is_experimental', False)]
        if experimental:
            return experimental[0]
        else:
            return same_hull[0]
    else:
        # No energy hull data, fall back to experimental preference
        return _select_by_experimental(candidates)

def _select_by_experimental(candidates: List[DatabaseMatch]) -> DatabaseMatch:
    """Select by preferring experimental over theoretical"""
    experimental = [c for c in candidates if c.metadata.get('is_experimental', False)]
    if experimental:
        return experimental[0]
    else:
        return candidates[0]

def _select_by_completeness(candidates: List[DatabaseMatch]) -> DatabaseMatch:
    """Select by data completeness"""
    def completeness_score(candidate):
        score = 0
        # Count non-None properties
        for prop in candidate.properties.values():
            if prop is not None:
                score += 1
        # Count non-None metadata
        for meta in candidate.metadata.values():
            if meta is not None:
                score += 1
        return score
    
    return max(candidates, key=completeness_score)

def _select_by_explicit_space_group(candidates: List[DatabaseMatch], reference_space_group: Optional[Union[str, int]] = None) -> DatabaseMatch:
    """Select candidates whose space group explicitly matches the reference (symbol or number).

    Ranking tiers:
      1) Exact match (symbol or number) with reference_space_group
      2) Has explicit space group but not a match
      3) Missing space group
    Within a tier, keep original order.
    If no reference provided, prefer tier 2 over 3.
    """
    exact: List[DatabaseMatch] = []
    has_sg: List[DatabaseMatch] = []
    missing: List[DatabaseMatch] = []

    ref_norm_str = str(reference_space_group).strip() if reference_space_group is not None else None

    for c in candidates:
        sg_symbol = c.properties.get('space_group')
        sg_number = c.properties.get('space_group_number')
        has = sg_symbol is not None or sg_number is not None
        if ref_norm_str is not None and has:
            # Compare as string both symbol and number
            if (sg_symbol is not None and str(sg_symbol).strip() == ref_norm_str) or (
                sg_number is not None and str(sg_number).strip() == ref_norm_str
            ):
                exact.append(c)
            else:
                has_sg.append(c)
        else:
            if has:
                has_sg.append(c)
            else:
                missing.append(c)

    if exact:
        return exact[0]
    if has_sg:
        return has_sg[0]
    return missing[0] if missing else candidates[0]


def _select_by_comprehensive_policy(candidates: List[DatabaseMatch]) -> DatabaseMatch:
    """Select by policy: low_hull, experimental, completeness, explicit_space_group"""
    if not candidates:
        raise ValueError('No candidates provided')
    
    # Score each candidate based on all four criteria
    def calculate_score(candidate):
        score = 0.0
        
        # 1. Energy hull (lower is better)
        energy_hull = candidate.properties.get('energy_hull')
        if energy_hull is not None:
            # Normalize: 0 eV = 1.0, higher values get lower scores
            hull_score = max(0, 1.0 - abs(energy_hull) * 10)  # Scale factor
            score += hull_score * 0.4
        
        # 2. Experimental status
        is_experimental = candidate.metadata.get('is_experimental', False)
        score += (1.0 if is_experimental else 0.0) * 0.3
        
        # 3. Data completeness
        completeness = 0
        total_props = 0
        for prop in candidate.properties.values():
            total_props += 1
            if prop is not None:
                completeness += 1
        for meta in candidate.metadata.values():
            total_props += 1
            if meta is not None:
                completeness += 1
        
        if total_props > 0:
            completeness_score = completeness / total_props
            score += completeness_score * 0.2
        
        # 4. Explicit space group
        has_space_group = candidate.properties.get('space_group') is not None
        score += (1.0 if has_space_group else 0.0) * 0.1
        
        return score
    
    # Score all candidates and return the best one
    scored_candidates = [(candidate, calculate_score(candidate)) for candidate in candidates]
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_candidate = scored_candidates[0][0]
    best_score = scored_candidates[0][1]
    
    logger.info(f'Selected candidate with comprehensive score: {best_score:.3f}')
    return best_candidate

def _get_ranking_criteria(policy: str) -> List[str]:
    """Get ranking criteria for a given policy"""
    criteria_map = {
        "low_hull_experimental_completeness_explicit_space_group": ["energy_hull", "is_experimental", "data_completeness", "space_group"],
        "prefer_low_hull_then_experimental": ["energy_hull", "is_experimental"],
        "prefer_experimental": ["is_experimental"],
        "prefer_completeness": ["data_completeness"],
        "prefer_explicit_space_group": ["space_group"]
    }
    return criteria_map.get(policy, ["default"])

def _get_selection_reason(selected: DatabaseMatch, policy: str) -> str:
    """Generate human-readable selection reason"""
    if policy == "low_hull_experimental_completeness_explicit_space_group":
        hull = selected.properties.get('energy_hull')
        exp = selected.metadata.get('is_experimental', False)
        sg = selected.properties.get('space_group')
        return f"Selected by comprehensive policy: hull={hull}, experimental={exp}, space_group={sg}"
    elif policy == "prefer_low_hull_then_experimental":
        hull = selected.properties.get('energy_hull')
        exp = selected.metadata.get('is_experimental', False)
        return f"Selected for low energy hull ({hull}) and experimental status ({exp})"
    elif policy == "prefer_experimental":
        exp = selected.metadata.get('is_experimental', False)
        return f"Selected for experimental status ({exp})"
    elif policy == "prefer_completeness":
        return "Selected for highest data completeness"
    elif policy == "prefer_explicit_space_group":
        sg = selected.properties.get('space_group')
        return f"Selected for explicit space group ({sg})"
    else:
        return "Selected as first available option"

# ============================================================================
# Helper Functions
# ============================================================================

def _parse_cif_content(cif_content: str) -> Dict[str, Any]:
    """Parse CIF content and extract structure data"""
    lines = cif_content.split('\n')
    structure_data = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith('_cell_length_a'):
            structure_data['cell_a'] = float(line.split()[1])
        elif line.startswith('_cell_length_b'):
            structure_data['cell_b'] = float(line.split()[1])
        elif line.startswith('_cell_length_c'):
            structure_data['cell_c'] = float(line.split()[1])
        elif line.startswith('_cell_angle_alpha'):
            structure_data['alpha'] = float(line.split()[1])
        elif line.startswith('_cell_angle_beta'):
            structure_data['beta'] = float(line.split()[1])
        elif line.startswith('_cell_angle_gamma'):
            structure_data['gamma'] = float(line.split()[1])
        elif line.startswith('_space_group_name_H-M'):
            structure_data['space_group'] = line.split()[1]
        elif line.startswith('_cell_volume'):
            structure_data['volume'] = float(line.split()[1])
    
    # Create cell parameters
    if all(key in structure_data for key in ['cell_a', 'cell_b', 'cell_c', 'alpha', 'beta', 'gamma']):
        structure_data['cell_parameters'] = {
            'a': structure_data['cell_a'],
            'b': structure_data['cell_b'],
            'c': structure_data['cell_c'],
            'alpha': structure_data['alpha'],
            'beta': structure_data['beta'],
            'gamma': structure_data['gamma']
        }
    
    return structure_data

def _parse_poscar_content(poscar_content: str) -> Dict[str, Any]:
    """Parse POSCAR content and extract structure data"""
    lines = poscar_content.split('\n')
    structure_data = {}
    
    # POSCAR format parsing (simplified)
    if len(lines) >= 2:
        structure_data['scaling_factor'] = float(lines[1].strip())
    
    if len(lines) >= 5:
        # Cell vectors
        cell_vectors = []
        for i in range(3):
            if len(lines) > 5 + i:
                vector = [float(x) for x in lines[5 + i].split()]
                cell_vectors.append(vector)
        structure_data['cell_vectors'] = cell_vectors
    
    return structure_data

def _extract_cif_formula(cif_content: str) -> str:
    """Extract chemical formula from CIF content"""
    lines = cif_content.split('\n')
    for line in lines:
        if line.strip().startswith('_chemical_formula_sum'):
            return line.split()[1]
    return "Unknown"

def _extract_poscar_formula(poscar_content: str) -> str:
    """Extract chemical formula from POSCAR content"""
    lines = poscar_content.split('\n')
    if len(lines) >= 6:
        # Formula is typically on line 6
        return lines[5].strip()
    return "Unknown"

def _calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file for provenance"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return "unknown"

# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_phase1_interop():
    """Example of Phase 1 interop functionality"""
    
    try:
        # Test CIF parsing (if file exists)
        cif_file = "test_structure.cif"
        if os.path.exists(cif_file):
            print("Testing CIF parsing...")
            crystal = from_cif(cif_file)
            print(f"Parsed CIF: {crystal.formula}")
            print(f"Provenance: {crystal.provenance}")
        
        # Test POSCAR parsing (if file exists)
        poscar_file = "POSCAR"
        if os.path.exists(poscar_file):
            print("\nTesting POSCAR parsing...")
            crystal = from_poscar(poscar_file)
            print(f"Parsed POSCAR: {crystal.formula}")
            print(f"Provenance: {crystal.provenance}")
        
        # Test database matching with variant cards
        print("\nTesting database matching with variant cards...")
        test_crystal = Crystal(
            formula="Si",
            structure_data={},
            provenance={"source": "test"},
            metadata={},
            properties={}
        )
        
        match_report, variant_cards = match_database_with_provenance(
            test_crystal, sources=["MP", "COD"]
        )
        
        print(f"Found {len(variant_cards)} variant cards")
        for card in variant_cards[:3]:  # Show first 3
            print(f"  - {card.get_display_summary()}")
        
        # Test variant selection
        if variant_cards:
            print("\nTesting variant selection...")
            selected_card, report = select_variant_with_policy(
                variant_cards, "prefer_low_hull_then_experimental"
            )
            print(f"Selected: {selected_card.get_display_summary()}")
            print(f"Reason: {report.selection_reason}")
        
        # Test UI cards creation
        print("\nTesting UI cards creation...")
        ui_cards = create_variant_cards_ui(variant_cards)
        print(f"Created {len(ui_cards)} UI cards")
        for card in ui_cards[:2]:  # Show first 2
            print(f"  - {card['display_summary']}")
        
    except Exception as e:
        logger.error(f"Error in Phase 1 interop example: {e}")
        raise


def calculate_similarity_score(query_formula: str, match_formula: str, properties: Dict[str, Any]) -> float:
    """
    Calculate similarity score based on formula match and properties.
    
    Args:
        query_formula: Original formula being searched
        match_formula: Formula from database match
        properties: Properties dictionary from match
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    score = 0.0
    
    # Formula similarity (exact match = 1.0)
    if query_formula == match_formula:
        score += 0.6  # 60% weight for exact formula match
    
    # Property completeness bonus
    non_none_props = sum(1 for v in properties.values() if v is not None)
    total_props = len(properties)
    completeness = non_none_props / total_props if total_props > 0 else 0
    score += completeness * 0.3  # 30% weight for completeness
    
    # Stability bonus (lower energy hull is better)
    energy_hull = properties.get('energy_hull')
    if energy_hull is not None:
        # Normalize energy hull (lower is better, max 1.0 eV)
        stability_bonus = max(0, (1.0 - min(energy_hull, 1.0)) * 0.1)
        score += stability_bonus
    
    return min(score, 1.0)  # Cap at 1.0


if __name__ == "__main__":
    example_phase1_interop() 