#!/usr/bin/env python3
"""
AtomForge DSL Converter

This module implements an intelligent approach for materials retrieval and 
AtomForge DSL generation using LLM reasoning and Materials Project integration.
"""

import os
import json
import re
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        pass

# Import existing components
try:
    from .converter import convert_material
    from ..parser.atomforge_parser import parse_atomforge_string
except ImportError:
    # Try alternative import paths
    try:
        from atomforge_mvp.src.converters.converter import convert_material
        from atomforge_mvp.src.parser.atomforge_parser import parse_atomforge_string
    except ImportError:
        # Fallback - define dummy functions
        def convert_material(*args, **kwargs):
            raise ImportError("Converter not available")
        
        def parse_atomforge_string(*args, **kwargs):
            raise ImportError("Parser not available")

@dataclass
class MaterialQuery:
    """Represents a material query with context and requirements."""
    query: str
    description: str
    expected_properties: List[str]
    confidence: float
    context: Dict[str, Any]

@dataclass
class MaterialData:
    """Represents retrieved material data from Materials Project."""
    material_id: str
    formula_pretty: str
    structure: Any
    properties: Dict[str, Any]
    computational_data: Dict[str, Any]
    provenance: Dict[str, Any]

class AtomForgeConverter:
    """
    AtomForge converter for intelligent materials retrieval and DSL generation.
    
    This converter implements a streamlined approach:
    1. Query Analysis - Understand the material request
    2. Intelligent Retrieval - Search Materials Project
    3. Data Enrichment - Gather additional context
    4. DSL Generation - Generate AtomForge DSL
    5. Validation - Ensure DSL correctness
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.mp_api_key = "MP_API_KEY_REDACTED"
        
        # Initialize LLM client if available
        self.llm_client = None
        if self.openai_api_key:
            try:
                import openai
                self.llm_client = openai.OpenAI(api_key=self.openai_api_key)
                print("OpenAI client initialized successfully")
            except ImportError:
                print("Warning: OpenAI library not available.")
        
    def analyze_query(self, user_input: str) -> MaterialQuery:
        """Analyze the user query to extract material information."""
        if not self.llm_client:
            return self._simple_query_analysis(user_input)
        
        analysis_prompt = f"""
        You are an expert materials scientist and AtomForge DSL specialist.
        
        Analyze the following material description and extract key information:
        
        User Input: {user_input}
        
        Please provide a JSON response with the following structure:
        {{
            "query": "The material to search for (formula, name, or MP ID)",
            "description": "A clear description of the material",
            "expected_properties": ["list", "of", "expected", "properties"],
            "confidence": 0.95,
            "context": {{
                "material_type": "crystal/molecule/amorphous",
                "application": "semiconductor/battery/catalyst/etc",
                "properties_of_interest": ["band_gap", "formation_energy", "elastic_constants"]
            }}
        }}
        
        Output ONLY the JSON object, no markdown formatting.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group(1))
                    return MaterialQuery(**analysis)
                except json.JSONDecodeError:
                    pass
            
            # Try to parse the entire content as JSON
            try:
                analysis = json.loads(content)
                return MaterialQuery(**analysis)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group())
                        return MaterialQuery(**analysis)
                    except json.JSONDecodeError:
                        pass
            
            return self._simple_query_analysis(user_input)
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._simple_query_analysis(user_input)
    
    def _simple_query_analysis(self, user_input: str) -> MaterialQuery:
        """Simple fallback analysis without LLM."""
        # Extract material query using regex
        formula_pattern = r'([A-Z][a-z]?\d*)'
        elements_found = re.findall(formula_pattern, user_input)
        
        # Validate elements - only allow real chemical elements
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
            'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
            'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
        }
        
        if elements_found:
            # Filter out invalid elements
            valid_elements_found = []
            for element in elements_found:
                element_symbol = re.sub(r'\d+', '', element)
                if element_symbol in valid_elements:
                    valid_elements_found.append(element)
            
            if valid_elements_found:
                query = ''.join(valid_elements_found)
            else:
                query = self._extract_reasonable_query(user_input)
        elif user_input.startswith('mp-'):
            query = user_input
        else:
            query = user_input.strip()
        
        return MaterialQuery(
            query=query,
            description=user_input,
            expected_properties=["structure", "properties"],
            confidence=0.7,
            context={
                "material_type": "crystal",
                "application": "general",
                "properties_of_interest": ["structure"]
            }
        )
    
    def _extract_reasonable_query(self, user_input: str) -> str:
        """Extract a reasonable query from user input when regex fails."""
        # Common material keywords to look for
        material_keywords = {
            'silicon', 'si', 'copper', 'cu', 'iron', 'fe', 'aluminum', 'al', 'gold', 'au',
            'lithium', 'li', 'sodium', 'na', 'potassium', 'k', 'calcium', 'ca',
            'titanium', 'ti', 'vanadium', 'v', 'chromium', 'cr', 'manganese', 'mn',
            'cobalt', 'co', 'nickel', 'ni', 'zinc', 'zn', 'gallium', 'ga', 'germanium', 'ge',
            'arsenic', 'as', 'selenium', 'se', 'bromine', 'br', 'rubidium', 'rb',
            'strontium', 'sr', 'yttrium', 'y', 'zirconium', 'zr', 'niobium', 'nb',
            'molybdenum', 'mo', 'technetium', 'tc', 'ruthenium', 'ru', 'rhodium', 'rh',
            'palladium', 'pd', 'silver', 'ag', 'cadmium', 'cd', 'indium', 'in',
            'tin', 'sn', 'antimony', 'sb', 'tellurium', 'te', 'iodine', 'i',
            'cesium', 'cs', 'barium', 'ba', 'lanthanum', 'la', 'cerium', 'ce',
            'praseodymium', 'pr', 'neodymium', 'nd', 'promethium', 'pm', 'samarium', 'sm',
            'europium', 'eu', 'gadolinium', 'gd', 'terbium', 'tb', 'dysprosium', 'dy',
            'holmium', 'ho', 'erbium', 'er', 'thulium', 'tm', 'ytterbium', 'yb',
            'lutetium', 'lu', 'hafnium', 'hf', 'tantalum', 'ta', 'tungsten', 'w',
            'rhenium', 're', 'osmium', 'os', 'iridium', 'ir', 'platinum', 'pt',
            'mercury', 'hg', 'thallium', 'tl', 'lead', 'pb', 'bismuth', 'bi'
        }
        
        # Look for material keywords in the input
        input_lower = user_input.lower()
        for keyword in material_keywords:
            if keyword in input_lower:
                return keyword.upper()
        
        return "Si"  # Default to silicon
    
    def retrieve_material_data(self, material_query: MaterialQuery) -> MaterialData:
        """Retrieve material data from Materials Project."""
        try:
            from pymatgen.ext.matproj import MPRester
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            
            with MPRester(self.mp_api_key) as m:
                # Use the query to search Materials Project
                if material_query.query.startswith('mp-'):
                    # Direct MP ID
                    docs = m.summary.search(material_ids=[material_query.query])
                else:
                    # Search by elements or formula
                    formula_pattern = r'([A-Z][a-z]?\d*)'
                    elements_found = re.findall(formula_pattern, material_query.query)
                    
                    if elements_found:
                        # Extract unique element symbols
                        element_symbols = []
                        for element in elements_found:
                            element_symbol = re.sub(r'\d+', '', element)
                            if element_symbol not in element_symbols:
                                element_symbols.append(element_symbol)
                        
                        # Search by elements
                        elements_str = ",".join(element_symbols)
                        try:
                            docs = m.summary.search(elements=elements_str)
                            
                            # Try to find exact match
                            for doc in docs:
                                if doc['formula_pretty'] == material_query.query:
                                    docs = [doc]
                                    break
                        except Exception:
                            docs = []
                        
                        # If no exact match found, try chemical system search
                        if not docs:
                            if len(element_symbols) == 2:
                                chemsys = f"{element_symbols[0]}-{element_symbols[1]}"
                            elif len(element_symbols) > 2:
                                chemsys = "-".join(element_symbols)
                            else:
                                chemsys = element_symbols[0]
                            
                            try:
                                docs = m.summary.search(chemsys=chemsys)
                            except Exception:
                                docs = []
                    else:
                        # Try as single element or chemical system
                        try:
                            docs = m.summary.search(elements=material_query.query)
                        except Exception:
                            docs = []
                        
                        if not docs:
                            try:
                                docs = m.summary.search(chemsys=material_query.query)
                            except Exception:
                                docs = []
                
                if not docs:
                    raise ValueError(f"No material found for '{material_query.query}'")
                
                # Get the best match
                doc = docs[0]
                structure = doc["structure"]
                sga = SpacegroupAnalyzer(structure)
                
                # Extract comprehensive data
                material_data = MaterialData(
                    material_id=doc["material_id"],
                    formula_pretty=doc["formula_pretty"],
                    structure=structure,
                    properties={
                        "space_group": sga.get_space_group_symbol(),
                        "crystal_system": sga.get_crystal_system(),
                        "volume": doc.get("volume"),
                        "density": doc.get("density"),
                        "band_gap": doc.get("band_gap"),
                        "formation_energy": doc.get("formation_energy_per_atom"),
                        "has_props": doc.get("has_props", {}),
                    },
                    computational_data={
                        "lattice": {
                            "a": structure.lattice.a,
                            "b": structure.lattice.b,
                            "c": structure.lattice.c,
                            "alpha": structure.lattice.alpha,
                            "beta": structure.lattice.beta,
                            "gamma": structure.lattice.gamma,
                        },
                        "atoms": len(structure),
                        "volume": structure.volume,
                    },
                    provenance={
                        "source": "Materials Project",
                        "database_IDs": doc.get("database_IDs", {}),
                        "has_props": doc.get("has_props", {}),
                    }
                )
                
                return material_data
                
        except Exception as e:
            print(f"Retrieval failed: {e}")
            raise
    
    def enrich_data(self, material_data: MaterialData, material_query: MaterialQuery) -> Dict[str, Any]:
        """Enrich material data with additional context."""
        if not self.llm_client:
            return self._simple_data_enrichment(material_data)
        
        enrichment_prompt = f"""
        You are an expert materials scientist. Analyze the following material data and enrich it with additional context.
        
        Material Query: {material_query.description}
        Material Data: {json.dumps(material_data.__dict__, default=str, indent=2)}
        
        Please provide a JSON response with enriched data:
        {{
            "computational_backend": {{
                "functional": "appropriate functional for this material",
                "energy_cutoff": "appropriate energy cutoff in eV",
                "k_point_density": "appropriate k-point density"
            }},
            "convergence_criteria": {{
                "energy_tolerance": "appropriate energy tolerance",
                "force_tolerance": "appropriate force tolerance",
                "stress_tolerance": "appropriate stress tolerance"
            }},
            "target_properties": {{
                "formation_energy": true/false,
                "band_gap": true/false,
                "elastic_constants": true/false
            }},
            "provenance": {{
                "method": "detailed method description",
                "doi": "DOI if available"
            }},
            "description": "enhanced material description"
        }}
        
        Output ONLY the JSON object, no markdown formatting.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": enrichment_prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            if not content:
                return self._simple_data_enrichment(material_data)
            
            # Try to extract JSON from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    enriched_data = json.loads(json_match.group(1))
                    return enriched_data
                except json.JSONDecodeError:
                    pass
            
            # Try to parse the entire content as JSON
            try:
                enriched_data = json.loads(content)
                return enriched_data
            except json.JSONDecodeError:
                return self._simple_data_enrichment(material_data)
            
        except Exception as e:
            print(f"LLM enrichment failed: {e}")
            return self._simple_data_enrichment(material_data)
    
    def _simple_data_enrichment(self, material_data: MaterialData) -> Dict[str, Any]:
        """Simple fallback data enrichment."""
        return {
            "computational_backend": {
                "functional": "PBE",
                "energy_cutoff": 520,
                "k_point_density": 1000.0
            },
            "convergence_criteria": {
                "energy_tolerance": 1e-5,
                "force_tolerance": 0.01,
                "stress_tolerance": 0.1
            },
            "target_properties": {
                "formation_energy": True,
                "band_gap": True,
                "elastic_constants": False
            },
            "provenance": {
                "method": "VASP DFT calculation",
                "doi": ""
            },
            "description": f"{material_data.formula_pretty} crystal structure from Materials Project"
        }
    
    def generate_dsl(self, material_data: MaterialData, enriched_data: Dict[str, Any], material_query: MaterialQuery) -> str:
        """Generate AtomForge DSL from material data."""
        if not self.llm_client:
            # Fallback to traditional converter
            return convert_material(
                material_data.material_id,
                description=enriched_data.get("description", material_query.description),
                computational_backend=enriched_data.get("computational_backend"),
                convergence_criteria=enriched_data.get("convergence_criteria"),
                target_properties=enriched_data.get("target_properties"),
                provenance_extra=enriched_data.get("provenance")
            )
        
        dsl_prompt = f"""
        You are an expert AtomForge DSL generator. Generate a complete AtomForge DSL specification
        for the following material based on the provided data.
        
        Material Query: {material_query.description}
        Material Data: {json.dumps(material_data.__dict__, default=str, indent=2)}
        Enriched Data: {json.dumps(enriched_data, indent=2)}
        
        Generate AtomForge DSL following this exact format:
        
        #atomforge_version "1.0";
        atom_spec [MATERIAL_NAME] {{
          header {{
            dsl_version = "1.0",
            title = "[FORMULA]",
            created = [YYYY-MM-DD],
            uuid = "[UUID]"
          }}
          description = "[DESCRIPTION]",
          units {{
            system = "crystallographic_default",
            length = angstrom,
            angle = degree
          }}
          lattice {{
            type = [crystal_system],
            a = [value],
            b = [value],
            c = [value],
            alpha = [value],
            beta = [value],
            gamma = [value]
          }}
          symmetry {{
            space_group = "[SPACE_GROUP]",
            origin_choice = 1
          }}
          basis {{
            site [ELEMENT1] {{
              wyckoff = "[MULTIPLICITY][LETTER]",
              position = ([x], [y], [z]),
              frame = fractional,
              species = ({{ element = "[ELEMENT]", occupancy = [OCCUPANCY] }})
            }}
          }}
          property_validation {{
            computational_backend: VASP {{
              functional: "[FUNCTIONAL]",
              energy_cutoff: [VALUE],
              k_point_density: [VALUE]
            }},
            convergence_criteria: {{
              energy_tolerance: [VALUE],
              force_tolerance: [VALUE],
              stress_tolerance: [VALUE]
            }},
            target_properties: {{
              formation_energy: [true/false],
              band_gap: [true/false],
              elastic_constants: [true/false]
            }}
          }}
          provenance {{
            source = "Materials Project",
            method = "[METHOD]",
            doi = "[DOI]"
          }}
        }}
        
        CRITICAL RULES:
        1. Use COMMAS (,) between fields within blocks, NOT semicolons (;)
        2. Use SEMICOLONS (;) only at the end of blocks and after the version declaration
        3. Use proper UUID format (e.g., "550e8400-e29b-41d4-a716-446655440000")
        4. Use proper date format (e.g., 2023-10-01 without quotes)
        5. Use proper boolean values (true/false without quotes)
        6. Use proper numeric values (without quotes)
        7. Use proper string values (with quotes)
        8. ALWAYS add commas after closing parentheses and braces within blocks
        9. Include ALL atomic sites - do NOT omit any sites for brevity
        
        Output ONLY the AtomForge DSL code - no markdown formatting, no explanations.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": dsl_prompt}],
                temperature=0.1
            )
            
            dsl_code = response.choices[0].message.content.strip()
            
            # Clean up markdown formatting if present
            dsl_code = self._clean_markdown_formatting(dsl_code)
            
            # Fix common syntax errors
            dsl_code = self._fix_common_syntax_errors(dsl_code)
            
            return dsl_code
            
        except Exception as e:
            print(f"LLM DSL generation failed: {e}")
            # Fallback to traditional converter
            return convert_material(
                material_data.material_id,
                description=enriched_data.get("description", material_query.description),
                computational_backend=enriched_data.get("computational_backend"),
                convergence_criteria=enriched_data.get("convergence_criteria"),
                target_properties=enriched_data.get("target_properties"),
                provenance_extra=enriched_data.get("provenance")
            )
    
    def _clean_markdown_formatting(self, dsl_code: str) -> str:
        """Remove markdown formatting from DSL code."""
        # Remove markdown code blocks
        dsl_code = re.sub(r'^```atomforge\s*', '', dsl_code, flags=re.MULTILINE)
        dsl_code = re.sub(r'^```plaintext\s*', '', dsl_code, flags=re.MULTILINE)
        dsl_code = re.sub(r'^```\s*', '', dsl_code, flags=re.MULTILINE)
        dsl_code = re.sub(r'\s*```$', '', dsl_code, flags=re.MULTILINE)
        
        # Remove any comments about omissions
        dsl_code = re.sub(r'\s*//\s*Additional sites omitted for brevity.*$', '', dsl_code, flags=re.MULTILINE)
        dsl_code = re.sub(r'\s*//\s*.*omitted.*$', '', dsl_code, flags=re.MULTILINE)
        
        # Remove any leading/trailing whitespace
        dsl_code = dsl_code.strip()
        
        return dsl_code
    
    def _fix_common_syntax_errors(self, dsl_code: str) -> str:
        """Fix common syntax errors in generated DSL."""
        # Fix missing commas after closing parentheses before closing braces
        dsl_code = re.sub(r'\)\s*(\})\s*$', r'),\1', dsl_code, flags=re.MULTILINE)
        
        # Fix missing commas after string values before closing braces
        dsl_code = re.sub(r'\"([^\"]*)\"\s*(\})\s*$', r'"\1",\2', dsl_code, flags=re.MULTILINE)
        
        # Fix missing commas after numeric values before closing braces
        dsl_code = re.sub(r'(\d+\.?\d*)\s*(\})\s*$', r'\1,\2', dsl_code, flags=re.MULTILINE)
        
        # Fix missing commas after boolean values before closing braces
        dsl_code = re.sub(r'(true|false)\s*(\})\s*$', r'\1,\2', dsl_code, flags=re.MULTILINE)
        
        return dsl_code
    
    def validate_dsl(self, dsl_code: str) -> Tuple[str, bool, str]:
        """Validate and refine the generated DSL."""
        # First, try to fix common syntax errors
        dsl_code = self._fix_common_syntax_errors(dsl_code)
        
        # Then try to parse the DSL
        try:
            parse_atomforge_string(dsl_code)
            is_valid = True
            validation_msg = "DSL is valid"
        except Exception as e:
            is_valid = False
            validation_msg = f"DSL validation error: {e}"
        
        # If invalid and LLM is available, try to fix it
        if not is_valid and self.llm_client:
            try:
                fix_prompt = f"""
                The following AtomForge DSL has validation errors:
                
                DSL Code:
                {dsl_code}
                
                Error: {validation_msg}
                
                Please fix the DSL code to follow the AtomForge grammar exactly.
                Generate only the corrected AtomForge DSL code.
                """
                
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": fix_prompt}],
                    temperature=0.1
                )
                
                fixed_dsl = response.choices[0].message.content.strip()
                
                # Validate the fixed DSL
                try:
                    parse_atomforge_string(fixed_dsl)
                    return fixed_dsl, True, "DSL fixed and validated"
                except Exception as e:
                    return fixed_dsl, False, f"DSL still has errors after fixing: {e}"
                    
            except Exception as e:
                return dsl_code, False, f"Could not fix DSL: {e}"
        
        return dsl_code, is_valid, validation_msg
    
    def convert(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main conversion pipeline.
        
        This implements the complete conversion approach:
        1. Query Analysis
        2. Intelligent Retrieval  
        3. Data Enrichment
        4. DSL Generation
        5. Validation
        """
        print("Step 1: Analyzing query...")
        material_query = self.analyze_query(user_input)
        
        print("Step 2: Retrieving material data...")
        material_data = self.retrieve_material_data(material_query)
        
        print("Step 3: Enriching data...")
        enriched_data = self.enrich_data(material_data, material_query)
        
        print("Step 4: Generating DSL...")
        dsl_code = self.generate_dsl(material_data, enriched_data, material_query)
        
        print("Step 5: Validating DSL...")
        final_dsl, is_valid, validation_msg = self.validate_dsl(dsl_code)
        
        # Prepare metadata
        metadata = {
            "material_query": material_query.__dict__,
            "material_data": {
                "material_id": material_data.material_id,
                "formula_pretty": material_data.formula_pretty,
                "properties": material_data.properties
            },
            "enriched_data": enriched_data,
            "validation": {
                "is_valid": is_valid,
                "message": validation_msg
            },
            "method": "AtomForge Converter"
        }
        
        return final_dsl, metadata

# Convenience function for backward compatibility
def convert_with_atomforge(user_input: str, openai_api_key: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Convert user input to AtomForge DSL using the AtomForge converter."""
    converter = AtomForgeConverter(openai_api_key)
    return converter.convert(user_input)

def main():
    """Test the AtomForge converter."""
    print("AtomForge DSL Converter")
    print("=" * 50)
    
    # Initialize converter
    converter = AtomForgeConverter()
    
    # Test cases
    test_cases = [
        "Silicon crystal structure for semiconductor applications",
        "LiFePO4 cathode material for lithium-ion batteries",
        "mp-149",
        "Iron metal with body-centered cubic structure"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case}")
        print("-" * 40)
        
        try:
            dsl, metadata = converter.convert(test_case)
            
            print(f"Success!")
            print(f"Material: {metadata['material_data']['formula_pretty']}")
            print(f"Validation: {metadata['validation']['message']}")
            
            # Show DSL preview
            lines = dsl.split('\n')[:10]
            print("DSL Preview:")
            for line in lines:
                print(f"   {line}")
            if len(dsl.split('\n')) > 10:
                print("   ...")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*50)
    
    print("AtomForge conversion testing complete!")

if __name__ == "__main__":
    main() 