#!/usr/bin/env python3
"""
AtomForge Database Connector - Phase 1 Implementation

This module implements the database matching functionality for Phase 1:
- Build from_cif / from_poscar and match_database (MP/ICSD/COD)
- Ship select_variant(policy) with preference policies
- Enhanced error handling and provenance tracking

Based on Phase 1 requirements from plan-v-2-0.tex
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Structures for Phase 1
# ============================================================================

@dataclass
class DatabaseMatch:
    """Represents a match from a database with provenance and similarity score"""
    database_name: str
    material_id: str
    formula: str
    similarity_score: float
    provenance: Dict[str, Any]
    properties: Dict[str, Any]
    metadata: Dict[str, Any]
    structure_data: Optional[Any] = None

@dataclass
class SelectionReport:
    """Report from variant selection process"""
    selected_variant: 'DatabaseMatch'
    ranking_criteria: List[str]
    policy_used: str
    alternatives_considered: List[DatabaseMatch]
    selection_reason: str
    timestamp: datetime

@dataclass
class MatchReport:
    """Report from database matching process"""
    query_formula: str
    matches_found: Dict[str, List[DatabaseMatch]]
    total_matches: int
    search_time: float
    databases_searched: List[str]
    errors: List[str]

# ============================================================================
# Enhanced Database Connectors (Fixed Version)
# ============================================================================

class MaterialsProjectConnector:
    """Materials Project API connector with Phase 1 enhancements"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.materialsproject.org"
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        })
        logger.info("Materials Project connector initialized")
    
    def search_by_formula(self, formula: str, limit: int = 10) -> List[DatabaseMatch]:
        """Search Materials Project by formula with enhanced matching"""
        try:
            import pymatgen.ext.matproj as mp
            with mp.MPRester(self.api_key) as m:
                # Fixed: Remove deprecated parameters
                docs = m.summary.search(formula=formula)
                
                matches = []
                for doc in docs[:limit]:
                    # Handle both dict and object responses
                    if isinstance(doc, dict):
                        material_id = doc.get('material_id', '')
                        formula_pretty = doc.get('formula_pretty', '')
                        energy_above_hull = doc.get('energy_above_hull', None)
                        formation_energy_per_atom = doc.get('formation_energy_per_atom', None)
                        band_gap = doc.get('band_gap', None)
                        density = doc.get('density', None)
                        volume = doc.get('volume', None)
                        is_stable = doc.get('is_stable', None)
                        is_metal = doc.get('is_metal', None)
                        is_magnetic = doc.get('is_magnetic', None)
                        is_experimental = doc.get('is_experimental', False)
                        theoretical = doc.get('theoretical', True)
                        nsites = doc.get('nsites', None)
                        symmetry = doc.get('symmetry', None)
                        structure = doc.get('structure', None)
                    else:
                        # Fallback for object attributes
                        material_id = getattr(doc, 'material_id', '')
                        formula_pretty = getattr(doc, 'formula_pretty', '')
                        energy_above_hull = getattr(doc, 'energy_above_hull', None)
                        formation_energy_per_atom = getattr(doc, 'formation_energy_per_atom', None)
                        band_gap = getattr(doc, 'band_gap', None)
                        density = getattr(doc, 'density', None)
                        volume = getattr(doc, 'volume', None)
                        is_stable = getattr(doc, 'is_stable', None)
                        is_metal = getattr(doc, 'is_metal', None)
                        is_magnetic = getattr(doc, 'is_magnetic', None)
                        is_experimental = getattr(doc, 'is_experimental', False)
                        theoretical = getattr(doc, 'theoretical', True)
                        nsites = getattr(doc, 'nsites', None)
                        symmetry = getattr(doc, 'symmetry', None)
                        structure = getattr(doc, 'structure', None)
                    
                    match = DatabaseMatch(
                        database_name="MP",
                        material_id=material_id,
                        formula=formula_pretty,
                        similarity_score=1.0,  # Exact formula match
                        provenance={
                            "database": "MP",
                            "id": material_id,
                            "retrieved_at": datetime.now().isoformat(),
                            "api_version": "v2"
                        },
                        properties={
                            "energy_hull": energy_above_hull,
                            "formation_energy": formation_energy_per_atom,
                            "band_gap": band_gap,
                            "density": density,
                            "volume": volume,
                            "is_stable": is_stable,
                            "is_metal": is_metal,
                            "is_magnetic": is_magnetic,
                            "space_group": symmetry.get("symbol", None) if symmetry else None,
                            "space_group_number": symmetry.get("number", None) if symmetry else None,
                            "crystal_system": symmetry.get("crystal_system", None) if symmetry else None
                        },
                        metadata={
                            "is_experimental": is_experimental,
                            "theoretical": theoretical,
                            "nsites": nsites,
                            "spacegroup": symmetry.get('symbol', None) if symmetry else None,
                            "crystal_system": symmetry.get('crystal_system', None) if symmetry else None,
                            "symmetry": symmetry
                        },
                        structure_data=structure
                    )
                    matches.append(match)
                
                logger.info(f"Found {len(matches)} matches in Materials Project")
                return matches
                
        except Exception as e:
            logger.error(f"Error searching Materials Project: {e}")
            return []

class CODConnector:
    """Crystallography Open Database connector with Phase 1 enhancements"""
    
    def __init__(self):
        self.base_url = "https://www.crystallography.net/cod"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AtomForge-MCP/1.0'
        })
        logger.info("COD connector initialized")
    
    def get_entry_by_id(self, cod_id: str) -> Optional[DatabaseMatch]:
        """Retrieve a specific COD entry by its 7-digit ID"""
        try:
            # Ensure COD ID is 7 digits
            cod_id = cod_id.zfill(7)
            cif_url = f"{self.base_url}/{cod_id}.cif"
            
            response = self.session.get(cif_url, timeout=30)
            response.raise_for_status()
            
            # Parse CIF content to extract metadata
            cif_content = response.text
            metadata = self._parse_cif_metadata(cif_content)
            
            match = DatabaseMatch(
                database_name="COD",
                material_id=cod_id,
                formula=metadata.get('formula', ''),
                similarity_score=1.0,  # Direct ID match
                provenance={
                    "database": "COD",
                    "id": cod_id,
                    "retrieved_at": datetime.now().isoformat(),
                    "api_version": "cif_direct"
                },
                properties={
                    "space_group": metadata.get('space_group'),
                    "space_group_number": metadata.get('space_group_number'),
                    "cell_volume": metadata.get('cell_volume'),
                    "z_value": metadata.get('z_value'),
                    "temperature": metadata.get('temperature'),
                    "pressure": metadata.get('pressure')
                },
                metadata={
                    "is_experimental": True,  # COD is primarily experimental
                    "theoretical": False,
                    "journal": metadata.get('journal'),
                    "year": metadata.get('year'),
                    "authors": metadata.get('authors'),
                    "title": metadata.get('title'),
                    "doi": metadata.get('doi'),
                    "cif_content": cif_content
                }
            )
            
            logger.info(f"Retrieved COD entry {cod_id}: {metadata.get('formula', 'Unknown')}")
            return match
            
        except Exception as e:
            logger.error(f"Error retrieving COD entry {cod_id}: {e}")
            return None
    
    def _parse_cif_metadata(self, cif_content: str) -> Dict[str, Any]:
        """Parse basic metadata from CIF content"""
        metadata = {}
        lines = cif_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('_chemical_formula_sum'):
                metadata['formula'] = line.split()[-1] if len(line.split()) > 1 else ''
            elif line.startswith('_space_group_name_H-M'):
                metadata['space_group'] = line.split()[-1] if len(line.split()) > 1 else ''
            elif line.startswith('_space_group_IT_number'):
                try:
                    metadata['space_group_number'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('_cell_volume'):
                try:
                    metadata['cell_volume'] = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('_cell_formula_units_Z'):
                try:
                    metadata['z_value'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('_exptl_crystal_description'):
                metadata['description'] = line.split(' ', 1)[-1] if len(line.split()) > 1 else ''
            elif line.startswith('_journal_name_full'):
                metadata['journal'] = line.split(' ', 1)[-1] if len(line.split()) > 1 else ''
            elif line.startswith('_journal_year'):
                try:
                    metadata['year'] = int(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            elif line.startswith('_publ_author_name'):
                metadata['authors'] = line.split(' ', 1)[-1] if len(line.split()) > 1 else ''
            elif line.startswith('_publ_section_title'):
                metadata['title'] = line.split(' ', 1)[-1] if len(line.split()) > 1 else ''
            elif line.startswith('_publ_section_doi'):
                metadata['doi'] = line.split()[-1] if len(line.split()) > 1 else ''
        
        return metadata
    
    def search_by_formula(self, formula: str, limit: int = 10) -> List[DatabaseMatch]:
        """Search COD database by formula - NOTE: COD doesn't provide formula search API"""
        logger.warning("COD does not provide a formula search API. Use get_entry_by_id() with known COD IDs instead.")
        logger.info("COD search requires known 7-digit COD IDs. Use: https://www.crystallography.net/cod/<COD-ID>.cif")
        return []

class ICSDConnector:
    """ICSD database connector with Phase 1 enhancements"""
    
    def __init__(self, api_key: str = None, username: str = None, password: str = None):
        self.api_key = api_key
        self.username = username
        self.password = password
        self.base_url = "https://icsd.products.fiz-karlsruhe.de"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AtomForge-MCP/1.0',
            'Content-Type': 'application/json'
        })
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}'
            })
        elif self.username and self.password:
            self.session.auth = (self.username, self.password)
        
        logger.info("ICSD connector initialized")
    
    def search_by_formula(self, formula: str, limit: int = 10) -> List[DatabaseMatch]:
        """Search ICSD database by formula with enhanced matching"""
        try:
            if not (self.api_key or (self.username and self.password)):
                logger.warning("ICSD authentication not configured")
                return []
            
            search_url = f"{self.base_url}/api/search"
            params = {
                'query': formula,
                'limit': limit,
                'format': 'json'
            }
            
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            matches = []
            
            for entry in data.get('results', []):
                match = DatabaseMatch(
                    database_name="ICSD",
                    material_id=str(entry.get('icsd_id', '')),
                    formula=entry.get('formula', ''),
                    similarity_score=1.0,  # Exact formula match
                    provenance={
                        "database": "ICSD",
                        "id": str(entry.get('icsd_id', '')),
                        "retrieved_at": datetime.now().isoformat(),
                        "api_version": "v1"
                    },
                    properties={
                        "space_group": entry.get('space_group'),
                        "space_group_number": entry.get('space_group_number'),
                        "volume": entry.get('volume'),
                        "density": entry.get('density'),
                        "temperature": entry.get('temperature'),
                        "pressure": entry.get('pressure')
                    },
                    metadata={
                        "is_experimental": True,  # ICSD is primarily experimental
                        "theoretical": False,
                        "journal": entry.get('journal'),
                        "year": entry.get('year'),
                        "authors": entry.get('authors'),
                        "title": entry.get('title'),
                        "doi": entry.get('doi')
                    }
                )
                matches.append(match)
            
            logger.info(f"Found {len(matches)} matches in ICSD")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching ICSD: {e}")
            return []

# ============================================================================
# Phase 1 Core Functions
# ============================================================================

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
                  policy: str = "prefer_low_hull_then_experimental") -> Tuple[DatabaseMatch, SelectionReport]:
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
    if policy == "prefer_low_hull_then_experimental":
        selected = _select_by_energy_hull_then_experimental(all_candidates)
    elif policy == "prefer_experimental":
        selected = _select_by_experimental(all_candidates)
    elif policy == "prefer_completeness":
        selected = _select_by_completeness(all_candidates)
    elif policy == "prefer_explicit_space_group":
        selected = _select_by_explicit_space_group(all_candidates)
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

def _select_by_explicit_space_group(candidates: List[DatabaseMatch]) -> DatabaseMatch:
    """Select by explicit space group information"""
    with_space_group = [c for c in candidates if c.properties.get('space_group') is not None]
    if with_space_group:
        return with_space_group[0]
    else:
        return candidates[0]

def _get_ranking_criteria(policy: str) -> List[str]:
    """Get ranking criteria for a given policy"""
    criteria_map = {
        "prefer_low_hull_then_experimental": ["energy_hull", "is_experimental"],
        "prefer_experimental": ["is_experimental"],
        "prefer_completeness": ["data_completeness"],
        "prefer_explicit_space_group": ["space_group"]
    }
    return criteria_map.get(policy, ["default"])

def _get_selection_reason(selected: DatabaseMatch, policy: str) -> str:
    """Generate human-readable selection reason"""
    if policy == "prefer_low_hull_then_experimental":
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
# Example Usage and Testing
# ============================================================================

def example_phase1_usage():
    """Example of Phase 1 database matching and variant selection"""
    
    try:
        # Test database matching
        print("Testing database matching...")
        match_report = match_database("Si", sources=["MP", "COD"])
        
        print(f"Found {match_report.total_matches} total matches")
        for db_name, matches in match_report.matches_found.items():
            print(f"  {db_name}: {len(matches)} matches")
            for match in matches[:2]:  # Show first 2
                print(f"    - {match.material_id}: {match.formula}")
        
        # Test variant selection
        print("\nTesting variant selection...")
        all_matches = []
        for matches in match_report.matches_found.values():
            all_matches.extend(matches)
        
        if all_matches:
            selected, report = select_variant(all_matches, "prefer_low_hull_then_experimental")
            print(f"Selected: {selected.database_name}:{selected.material_id}")
            print(f"Reason: {report.selection_reason}")
            print(f"Policy: {report.policy_used}")
        
    except Exception as e:
        logger.error(f"Error in Phase 1 example: {e}")
        raise

if __name__ == "__main__":
    example_phase1_usage() 