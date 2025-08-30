#!/usr/bin/env python3
"""
Database Connectors for AtomForge DSL

This module provides connectors for various materials science databases:
- Materials Project (MVP)
- COD (Crystallography Open Database) (MVP)
- AFLOW (High Priority)
- OQMD (High Priority)
- ICSD (High Priority)

Each connector follows the same interface pattern for consistency.
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime

# Import existing components
try:
    from pymatgen.ext.matproj import MPRester
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.cif import CifParser
    from pymatgen.core import Structure
except ImportError as e:
    print(f"Warning: pymatgen not available: {e}")

@dataclass
class DatabaseResult:
    """Standardized result format for all database connectors"""
    database_name: str
    material_id: str
    formula: str
    structure: Any
    properties: Dict[str, Any]
    metadata: Dict[str, Any]
    provenance: Dict[str, Any]

class BaseDatabaseConnector:
    """Base class for all database connectors"""
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
    def search(self, query: str, limit: int = 10) -> List[DatabaseResult]:
        """Search the database - to be implemented by subclasses"""
        raise NotImplementedError
        
    def get_material(self, material_id: str) -> Optional[DatabaseResult]:
        """Get specific material by ID - to be implemented by subclasses"""
        raise NotImplementedError
        
    def convert_to_atomforge(self, result: DatabaseResult) -> str:
        """Convert database result to AtomForge DSL - to be implemented by subclasses"""
        raise NotImplementedError
        
    def _cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
        
    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Get cached result"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None
        
    def _set_cached(self, cache_key: str, data: Dict):
        """Set cached result"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not cache result: {e}")

class MaterialsProjectConnector(BaseDatabaseConnector):
    """Enhanced Materials Project connector"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.materialsproject.org"
        
    def search(self, query: str, limit: int = 10) -> List[DatabaseResult]:
        """Search Materials Project by formula or elements"""
        cache_key = self._cache_key(f"mp_search_{query}_{limit}")
        cached = self._get_cached(cache_key)
        if cached:
            return [DatabaseResult(**item) for item in cached]
            
        try:
            with MPRester(self.api_key) as m:
                # Try different search strategies
                results = []
                
                # Strategy 1: Search by formula
                try:
                    docs = m.summary.search(formula=query, limit=limit)
                    results.extend(docs)
                except Exception:
                    pass
                
                # Strategy 2: Search by elements
                if len(results) < limit:
                    try:
                        # Extract elements from query
                        import re
                        elements = re.findall(r'[A-Z][a-z]?', query)
                        if elements:
                            docs = m.summary.search(elements=elements, limit=limit-len(results))
                            results.extend(docs)
                    except Exception:
                        pass
                
                # Strategy 3: Search by chemical system
                if len(results) < limit:
                    try:
                        if '-' in query:
                            docs = m.summary.search(chemsys=query, limit=limit-len(results))
                            results.extend(docs)
                    except Exception:
                        pass
                
                # Convert to DatabaseResult format
                db_results = []
                for doc in results[:limit]:
                    try:
                        structure = doc.get('structure')
                        if structure:
                            sga = SpacegroupAnalyzer(structure)
                            db_result = DatabaseResult(
                                database_name="materials_project",
                                material_id=doc['material_id'],
                                formula=doc['formula_pretty'],
                                structure=structure,
                                properties={
                                    'space_group': sga.get_space_group_symbol(),
                                    'crystal_system': sga.get_crystal_system(),
                                    'volume': doc.get('volume'),
                                    'density': doc.get('density'),
                                    'band_gap': doc.get('band_gap'),
                                    'formation_energy': doc.get('formation_energy_per_atom'),
                                },
                                metadata={
                                    'has_props': doc.get('has_props', {}),
                                    'nsites': doc.get('nsites'),
                                },
                                provenance={
                                    'source': 'Materials Project',
                                    'database_id': doc['material_id'],
                                    'retrieved_at': datetime.now().isoformat(),
                                }
                            )
                            db_results.append(db_result)
                    except Exception as e:
                        print(f"Error processing MP result: {e}")
                
                # Cache results
                self._set_cached(cache_key, [result.__dict__ for result in db_results])
                return db_results
                
        except Exception as e:
            print(f"Error searching Materials Project: {e}")
            return []
    
    def get_material(self, material_id: str) -> Optional[DatabaseResult]:
        """Get specific material by MP ID"""
        cache_key = self._cache_key(f"mp_material_{material_id}")
        cached = self._get_cached(cache_key)
        if cached:
            return DatabaseResult(**cached)
            
        try:
            with MPRester(self.api_key) as m:
                doc = m.summary.search(material_ids=[material_id])[0]
                structure = doc.get('structure')
                if structure:
                    sga = SpacegroupAnalyzer(structure)
                    result = DatabaseResult(
                        database_name="materials_project",
                        material_id=doc['material_id'],
                        formula=doc['formula_pretty'],
                        structure=structure,
                        properties={
                            'space_group': sga.get_space_group_symbol(),
                            'crystal_system': sga.get_crystal_system(),
                            'volume': doc.get('volume'),
                            'density': doc.get('density'),
                            'band_gap': doc.get('band_gap'),
                            'formation_energy': doc.get('formation_energy_per_atom'),
                        },
                        metadata={
                            'has_props': doc.get('has_props', {}),
                            'nsites': doc.get('nsites'),
                        },
                        provenance={
                            'source': 'Materials Project',
                            'database_id': doc['material_id'],
                            'retrieved_at': datetime.now().isoformat(),
                        }
                    )
                    self._set_cached(cache_key, result.__dict__)
                    return result
        except Exception as e:
            print(f"Error getting MP material {material_id}: {e}")
            return None
    
    def convert_to_atomforge(self, result: DatabaseResult) -> str:
        """Convert MP result to AtomForge DSL"""
        # Import the existing converter function
        from converter import convert_material
        return convert_material(result.material_id)

class CODConnector(BaseDatabaseConnector):
    """Crystallography Open Database connector"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.crystallography.net/cod/api"
        
    def search(self, query: str, limit: int = 10) -> List[DatabaseResult]:
        """Search COD by formula"""
        cache_key = self._cache_key(f"cod_search_{query}_{limit}")
        cached = self._get_cached(cache_key)
        if cached:
            return [DatabaseResult(**item) for item in cached]
            
        try:
            # COD API endpoint for search
            url = f"{self.base_url}/search"
            params = {
                "formula": query,
                "format": "json",
                "limit": limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            db_results = []
            for item in data.get('results', [])[:limit]:
                try:
                    # Get CIF data for this entry
                    cod_id = item.get('id')
                    if cod_id:
                        cif_data = self.get_cif_data(cod_id)
                        if cif_data:
                            structure = self._parse_cif(cif_data)
                            if structure:
                                db_result = DatabaseResult(
                                    database_name="cod",
                                    material_id=str(cod_id),
                                    formula=item.get('formula', ''),
                                    structure=structure,
                                    properties={
                                        'space_group': item.get('spacegroup'),
                                        'volume': item.get('volume'),
                                        'cell_length_a': item.get('cell_length_a'),
                                        'cell_length_b': item.get('cell_length_b'),
                                        'cell_length_c': item.get('cell_length_c'),
                                        'cell_angle_alpha': item.get('cell_angle_alpha'),
                                        'cell_angle_beta': item.get('cell_angle_beta'),
                                        'cell_angle_gamma': item.get('cell_angle_gamma'),
                                    },
                                    metadata={
                                        'title': item.get('title'),
                                        'journal': item.get('journal'),
                                        'year': item.get('year'),
                                    },
                                    provenance={
                                        'source': 'Crystallography Open Database',
                                        'database_id': str(cod_id),
                                        'retrieved_at': datetime.now().isoformat(),
                                    }
                                )
                                db_results.append(db_result)
                except Exception as e:
                    print(f"Error processing COD result: {e}")
            
            # Cache results
            self._set_cached(cache_key, [result.__dict__ for result in db_results])
            return db_results
            
        except Exception as e:
            print(f"Error searching COD: {e}")
            return []
    
    def get_cif_data(self, cod_id: str) -> Optional[str]:
        """Get CIF format data from COD"""
        try:
            url = f"{self.base_url}/cif/{cod_id}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error getting CIF data for COD ID {cod_id}: {e}")
            return None
    
    def _parse_cif(self, cif_data: str) -> Optional[Any]:
        """Parse CIF data to pymatgen Structure"""
        try:
            parser = CifParser.from_string(cif_data)
            return parser.get_structures()[0]
        except Exception as e:
            print(f"Error parsing CIF data: {e}")
            return None
    
    def convert_to_atomforge(self, result: DatabaseResult) -> str:
        """Convert COD result to AtomForge DSL"""
        # This would need to be implemented based on your existing converter
        # For now, return a basic template
        structure = result.structure
        if not structure:
            return "# Error: No structure data available"
            
        # Generate basic AtomForge DSL
        dsl = f"""#atomforge_version "1.0";

atom_spec {result.formula.replace(' ', '_')} {{
    header {{
        dsl_version = "1.0",
        title = "{result.formula}",
        created = {datetime.now().strftime('%Y-%m-%d')},
        uuid = "{result.material_id}"
    }}
    
    description = "Structure from COD database (ID: {result.material_id})",
    
    units {{
        system = "crystallographic_default",
        length = "angstrom",
        angle = "degree"
    }}
    
    lattice {{
        type = "{self._get_lattice_type(structure)}",
        a = {structure.lattice.a:.6f},
        b = {structure.lattice.b:.6f},
        c = {structure.lattice.c:.6f},
        alpha = {structure.lattice.alpha:.6f},
        beta = {structure.lattice.beta:.6f},
        gamma = {structure.lattice.gamma:.6f}
    }}
    
    symmetry {{
        space_group = "{result.properties.get('space_group', 'Unknown')}"
    }}
    
    basis {{
"""
        
        # Add sites
        for i, site in enumerate(structure):
            species_str = site.species_string
            coords = site.frac_coords
            dsl += f"""        site site_{i} {{
            wyckoff = "?",
            position = ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}),
            frame = fractional,
            species = ({{ element = "{species_str}", occupancy = 1.0 }})
        }}
    """
        
        dsl += """    }
    
    provenance {
        source = "Crystallography Open Database",
        method = "experimental",
        doi = ""
    }
    }"""
        
        return dsl
    
    def _get_lattice_type(self, structure: Any) -> str:
        """Get lattice type from structure"""
        try:
            sga = SpacegroupAnalyzer(structure)
            crystal_system = sga.get_crystal_system().lower()
            valid_types = ["cubic", "tetragonal", "orthorhombic", "hexagonal", 
                          "rhombohedral", "monoclinic", "triclinic"]
            return crystal_system if crystal_system in valid_types else "triclinic"
        except Exception:
            return "triclinic"

class AFLOWConnector(BaseDatabaseConnector):
    """AFLOW database connector"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://aflow.org/API/aflux"
        self.optimade_url = "https://aflow.org/API/optimade/v1"
        
    def search(self, query: str, limit: int = 10) -> List[DatabaseResult]:
        """Search AFLOW database"""
        cache_key = self._cache_key(f"aflow_search_{query}_{limit}")
        cached = self._get_cached(cache_key)
        if cached:
            return [DatabaseResult(**item) for item in cached]
            
        try:
            # Use AFLOW's aflux API
            url = f"{self.base_url}/search"
            params = {
                "q": query,
                "limit": limit,
                "format": "json"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            db_results = []
            for item in data.get('results', [])[:limit]:
                try:
                    db_result = DatabaseResult(
                        database_name="aflow",
                        material_id=item.get('auid', ''),
                        formula=item.get('composition', ''),
                        structure=None,  # Would need to parse structure data
                        properties={
                            'space_group': item.get('spacegroup_relax'),
                            'volume': item.get('volume_cell'),
                            'energy': item.get('energy_cell'),
                            'formation_energy': item.get('energy_atom'),
                        },
                        metadata={
                            'prototype': item.get('prototype'),
                            'auid': item.get('auid'),
                        },
                        provenance={
                            'source': 'AFLOW',
                            'database_id': item.get('auid', ''),
                            'retrieved_at': datetime.now().isoformat(),
                        }
                    )
                    db_results.append(db_result)
                except Exception as e:
                    print(f"Error processing AFLOW result: {e}")
            
            # Cache results
            self._set_cached(cache_key, [result.__dict__ for result in db_results])
            return db_results
            
        except Exception as e:
            print(f"Error searching AFLOW: {e}")
            return []
    
    def convert_to_atomforge(self, result: DatabaseResult) -> str:
        """Convert AFLOW result to AtomForge DSL"""
        # Basic template for AFLOW data
        dsl = f"""#atomforge_version "1.0";

atom_spec {result.formula.replace(' ', '_')} {{
    header {{
        dsl_version = "1.0",
        title = "{result.formula}",
        created = {datetime.now().strftime('%Y-%m-%d')},
        uuid = "{result.material_id}"
    }}
    
    description = "Structure from AFLOW database (AUID: {result.material_id})",
    
    units {{
        system = "crystallographic_default",
        length = "angstrom",
        angle = "degree"
    }}
    
    lattice {{
        type = "cubic",
        a = 1.0,
        b = 1.0,
        c = 1.0,
        alpha = 90.0,
        beta = 90.0,
        gamma = 90.0
    }}
    
    symmetry {{
        space_group = "{result.properties.get('space_group', 'Unknown')}"
    }}
    
    basis {{
        # Structure data would be populated here
    }}
    
    property_validation {{
        computational_backend: VASP {{
            functional: "PBE",
            energy_cutoff: 520.0,
            k_point_density: 1000.0
        }}
    }}
    
    provenance {{
        source = "AFLOW",
        method = "computational",
        doi = ""
    }}
}}"""
        
        return dsl

class OQMDConnector(BaseDatabaseConnector):
    """OQMD (Open Quantum Materials Database) connector"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://oqmd.org/API"
        
    def search(self, query: str, limit: int = 10) -> List[DatabaseResult]:
        """Search OQMD database"""
        cache_key = self._cache_key(f"oqmd_search_{query}_{limit}")
        cached = self._get_cached(cache_key)
        if cached:
            return [DatabaseResult(**item) for item in cached]
            
        try:
            # OQMD API endpoint
            url = f"{self.base_url}/search"
            params = {
                "q": query,
                "limit": limit,
                "format": "json"
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            db_results = []
            for item in data.get('results', [])[:limit]:
                try:
                    db_result = DatabaseResult(
                        database_name="oqmd",
                        material_id=item.get('id', ''),
                        formula=item.get('composition', ''),
                        structure=None,  # Would need to parse structure data
                        properties={
                            'formation_energy': item.get('formation_energy'),
                            'volume': item.get('volume'),
                            'space_group': item.get('space_group'),
                        },
                        metadata={
                            'entry_id': item.get('entry_id'),
                            'calculation_id': item.get('calculation_id'),
                        },
                        provenance={
                            'source': 'OQMD',
                            'database_id': item.get('id', ''),
                            'retrieved_at': datetime.now().isoformat(),
                        }
                    )
                    db_results.append(db_result)
                except Exception as e:
                    print(f"Error processing OQMD result: {e}")
            
            # Cache results
            self._set_cached(cache_key, [result.__dict__ for result in db_results])
            return db_results
            
        except Exception as e:
            print(f"Error searching OQMD: {e}")
            return []
    
    def convert_to_atomforge(self, result: DatabaseResult) -> str:
        """Convert OQMD result to AtomForge DSL"""
        # Basic template for OQMD data
        dsl = f"""#atomforge_version "1.0";

atom_spec {result.formula.replace(' ', '_')} {{
    header {{
        dsl_version = "1.0",
        title = "{result.formula}",
        created = {datetime.now().strftime('%Y-%m-%d')},
        uuid = "{result.material_id}"
    }}
    
    description = "Structure from OQMD database (ID: {result.material_id})",
    
    units {{
        system = "crystallographic_default",
        length = "angstrom",
        angle = "degree"
    }}
    
    lattice {{
        type = "cubic",
        a = 1.0,
        b = 1.0,
        c = 1.0,
        alpha = 90.0,
        beta = 90.0,
        gamma = 90.0
    }}
    
    symmetry {{
        space_group = "{result.properties.get('space_group', 'Unknown')}"
    }}
    
    basis {{
        # Structure data would be populated here
    }}
    
    property_validation {{
        computational_backend: VASP {{
            functional: "PBE",
            energy_cutoff: 520.0,
            k_point_density: 1000.0
        }}
    }}
    
    provenance {{
        source = "OQMD",
        method = "computational",
        doi = ""
    }}
}}"""
        
        return dsl

class ICSDConnector(BaseDatabaseConnector):
    """ICSD (Inorganic Crystal Structure Database) connector"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://icsd.fiz-karlsruhe.de/api/v1"
        
    def search(self, query: str, limit: int = 10) -> List[DatabaseResult]:
        """Search ICSD database"""
        cache_key = self._cache_key(f"icsd_search_{query}_{limit}")
        cached = self._get_cached(cache_key)
        if cached:
            return [DatabaseResult(**item) for item in cached]
            
        try:
            # ICSD API endpoint
            url = f"{self.base_url}/search"
            headers = {'Authorization': f'Bearer {self.api_key}'}
            params = {
                "q": query,
                "limit": limit,
                "format": "json"
            }
            
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            db_results = []
            for item in data.get('results', [])[:limit]:
                try:
                    db_result = DatabaseResult(
                        database_name="icsd",
                        material_id=item.get('icsd_id', ''),
                        formula=item.get('formula', ''),
                        structure=None,  # Would need to parse structure data
                        properties={
                            'space_group': item.get('space_group'),
                            'volume': item.get('volume'),
                            'cell_length_a': item.get('cell_length_a'),
                            'cell_length_b': item.get('cell_length_b'),
                            'cell_length_c': item.get('cell_length_c'),
                        },
                        metadata={
                            'title': item.get('title'),
                            'journal': item.get('journal'),
                            'year': item.get('year'),
                        },
                        provenance={
                            'source': 'ICSD',
                            'database_id': item.get('icsd_id', ''),
                            'retrieved_at': datetime.now().isoformat(),
                        }
                    )
                    db_results.append(db_result)
                except Exception as e:
                    print(f"Error processing ICSD result: {e}")
            
            # Cache results
            self._set_cached(cache_key, [result.__dict__ for result in db_results])
            return db_results
            
        except Exception as e:
            print(f"Error searching ICSD: {e}")
            return []
    
    def convert_to_atomforge(self, result: DatabaseResult) -> str:
        """Convert ICSD result to AtomForge DSL"""
        # Basic template for ICSD data
        dsl = f"""#atomforge_version "1.0";

atom_spec {result.formula.replace(' ', '_')} {{
    header {{
        dsl_version = "1.0",
        title = "{result.formula}",
        created = {datetime.now().strftime('%Y-%m-%d')},
        uuid = "{result.material_id}"
    }}
    
    description = "Structure from ICSD database (ID: {result.material_id})",
    
    units {{
        system = "crystallographic_default",
        length = "angstrom",
        angle = "degree"
    }}
    
    lattice {{
        type = "cubic",
        a = {result.properties.get('cell_length_a', 1.0)},
        b = {result.properties.get('cell_length_b', 1.0)},
        c = {result.properties.get('cell_length_c', 1.0)},
        alpha = 90.0,
        beta = 90.0,
        gamma = 90.0
    }}
    
    symmetry {{
        space_group = "{result.properties.get('space_group', 'Unknown')}"
    }}
    
    basis {{
        # Structure data would be populated here
    }}
    
    provenance {{
        source = "ICSD",
        method = "experimental",
        doi = ""
    }}
}}"""
        
        return dsl

class UnifiedMaterialsDatabase:
    """Unified interface for multiple materials databases"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.connectors = {
            'materials_project': MaterialsProjectConnector(
                self.config.get('mp_api_key', 'MP_API_KEY_REDACTED')
            ),
            'cod': CODConnector(),
            'aflow': AFLOWConnector(),
            'oqmd': OQMDConnector(),
            'icsd': ICSDConnector(self.config.get('icsd_api_key', ''))
        }
        
    def search_across_databases(self, query: str, databases: List[str] = None, 
                               limit_per_db: int = 5) -> Dict[str, List[DatabaseResult]]:
        """Search across multiple databases simultaneously"""
        results = {}
        databases = databases or list(self.connectors.keys())
        
        for db_name in databases:
            try:
                connector = self.connectors[db_name]
                results[db_name] = connector.search(query, limit=limit_per_db)
                print(f"Found {len(results[db_name])} results from {db_name}")
            except Exception as e:
                print(f"Error searching {db_name}: {e}")
                results[db_name] = []
        
        return results
    
    def get_material_from_database(self, database: str, material_id: str) -> Optional[DatabaseResult]:
        """Get specific material from a database"""
        connector = self.connectors.get(database)
        if connector:
            return connector.get_material(material_id)
        else:
            print(f"Unknown database: {database}")
            return None
    
    def convert_to_atomforge(self, database: str, result: DatabaseResult) -> str:
        """Convert database result to AtomForge DSL"""
        connector = self.connectors.get(database)
        if connector:
            return connector.convert_to_atomforge(result)
        else:
            raise ValueError(f"Unknown database: {database}")
    
    def validate_structure_consistency(self, material_id: str, databases: List[str] = None) -> Dict[str, Any]:
        """Validate structure data across databases"""
        validation_results = {}
        databases = databases or list(self.connectors.keys())
        
        for db_name in databases:
            try:
                connector = self.connectors[db_name]
                result = connector.get_material(material_id)
                if result:
                    validation_results[db_name] = {
                        'found': True,
                        'formula': result.formula,
                        'properties': result.properties
                    }
                else:
                    validation_results[db_name] = {'found': False}
            except Exception as e:
                validation_results[db_name] = {'error': str(e)}
        
        return validation_results

# Example usage
if __name__ == "__main__":
    # Initialize unified database
    config = {
        'mp_api_key': 'MP_API_KEY_REDACTED',
        'icsd_api_key': ''  # Add your ICSD API key if available
    }
    
    db = UnifiedMaterialsDatabase(config)
    
    # Search across all databases
    results = db.search_across_databases("Si", limit_per_db=3)
    
    # Print results
    for db_name, db_results in results.items():
        print(f"\n{db_name.upper()} Results:")
        for result in db_results:
            print(f"  - {result.formula} (ID: {result.material_id})")
    
    # Convert a result to AtomForge DSL
    if results['materials_project']:
        first_result = results['materials_project'][0]
        dsl = db.convert_to_atomforge('materials_project', first_result)
        print(f"\nAtomForge DSL for {first_result.formula}:")
        print(dsl) 