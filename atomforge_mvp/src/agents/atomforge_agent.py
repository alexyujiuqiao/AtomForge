#!/usr/bin/env python3
"""
AtomForge Agent

A specialized agent for the AtomForge project that follows a 4-step workflow:
1. Query Analysis - Understand user intent
2. Data Query - Retrieve material data
3. DSL Converter - Generate AtomForge DSL
4. DSL Validator - Validate and fix DSL
"""

import os
import json
import re
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    def load_dotenv():
        pass

class AgentStep(Enum):
    """Enumeration of agent workflow steps."""
    QUERY_ANALYSIS = "query_analysis"
    DATA_QUERY = "data_query"
    DSL_CONVERTER = "dsl_converter"
    DSL_VALIDATOR = "dsl_validator"

class AgentStatus(Enum):
    """Enumeration of agent status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"

@dataclass
class AgentContext:
    """Context passed between agent steps."""
    user_input: str
    step_results: Dict[str, Any]
    metadata: Dict[str, Any]
    status: AgentStatus = AgentStatus.PENDING
    error_message: Optional[str] = None

@dataclass
class QueryAnalysisResult:
    """Result of query analysis step."""
    material_query: str
    material_type: str
    application: str
    expected_properties: List[str]
    confidence: float
    search_strategy: str
    context: Dict[str, Any]

@dataclass
class DataQueryResult:
    """Result of data query step."""
    material_id: str
    formula_pretty: str
    structure_data: Dict[str, Any]
    properties: Dict[str, Any]
    source: str
    confidence: float

@dataclass
class DSLConverterResult:
    """Result of DSL converter step."""
    dsl_code: str
    generation_method: str
    parameters_used: Dict[str, Any]
    confidence: float

@dataclass
class DSLValidatorResult:
    """Result of DSL validator step."""
    is_valid: bool
    validation_message: str
    fixed_dsl: Optional[str] = None
    errors_found: List[str] = None
    warnings: List[str] = None

class AtomForgeAgent:
    """
    Specialized AtomForge agent implementing the 4-step workflow.
    
    Workflow:
    1. Query Analysis - Understand user intent and extract material information
    2. Data Query - Retrieve material data from databases
    3. DSL Converter - Generate AtomForge DSL from material data
    4. DSL Validator - Validate and fix generated DSL
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the AtomForge agent."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.mp_api_key = os.getenv("MP_API_KEY", "LTL7OdZrpi4tRpD90gfdDmp7VjvZh0PE")
        
        # Initialize LLM client if available
        self.llm_client = None
        if self.openai_api_key:
            try:
                import openai
                self.llm_client = openai.OpenAI(api_key=self.openai_api_key)
                print("OpenAI client initialized successfully")
            except ImportError:
                print("Warning: OpenAI library not available.")
        
        # Initialize step handlers
        self.step_handlers = {
            AgentStep.QUERY_ANALYSIS: self._handle_query_analysis,
            AgentStep.DATA_QUERY: self._handle_data_query,
            AgentStep.DSL_CONVERTER: self._handle_dsl_converter,
            AgentStep.DSL_VALIDATOR: self._handle_dsl_validator
        }
    
    def run(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Run the complete AtomForge agent workflow.
        
        Args:
            user_input: Natural language description of the material
            
        Returns:
            Tuple of (final_dsl, metadata)
        """
        print("Starting AtomForge Agent Workflow")
        print("=" * 50)
        
        # Initialize context
        context = AgentContext(
            user_input=user_input,
            step_results={},
            metadata={
                "workflow_start": datetime.now().isoformat(),
                "agent_version": "1.0.0",
                "workflow_steps": []
            }
        )
        
        # Execute workflow steps
        workflow_steps = [
            AgentStep.QUERY_ANALYSIS,
            AgentStep.DATA_QUERY,
            AgentStep.DSL_CONVERTER,
            AgentStep.DSL_VALIDATOR
        ]
        
        for step in workflow_steps:
            print(f"\n📋 Step {step.value.upper()}")
            print("-" * 30)
            
            try:
                # Execute step
                result = self.step_handlers[step](context)
                context.step_results[step.value] = result
                context.metadata["workflow_steps"].append({
                    "step": step.value,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
                
                print(f"{step.value} completed successfully")
                
            except Exception as e:
                error_msg = f"Step {step.value} failed: {str(e)}"
                print(f"Error: {error_msg}")
                
                context.status = AgentStatus.FAILED
                context.error_message = error_msg
                context.metadata["workflow_steps"].append({
                    "step": step.value,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Return partial results if available
                if context.step_results:
                    context.status = AgentStatus.PARTIAL
                    return self._create_partial_response(context)
                else:
                    raise e
        
        # Workflow completed successfully
        context.status = AgentStatus.SUCCESS
        context.metadata["workflow_end"] = datetime.now().isoformat()
        
        print("\nAtomForge Agent Workflow Completed Successfully!")
        print("=" * 50)
        
        return self._create_final_response(context)
    
    def _handle_query_analysis(self, context: AgentContext) -> QueryAnalysisResult:
        """Step 1: Analyze user query to understand intent."""
        print("🔍 Analyzing user query...")
        
        if self.llm_client:
            return self._llm_query_analysis(context.user_input)
        else:
            return self._simple_query_analysis(context.user_input)
    
    def _llm_query_analysis(self, user_input: str) -> QueryAnalysisResult:
        """Use LLM for intelligent query analysis."""
        analysis_prompt = f"""
        You are an expert materials scientist and AtomForge DSL specialist.
        
        Analyze the following material description and extract key information:
        
        User Input: {user_input}
        
        Please provide a JSON response with the following structure:
        {{
            "material_query": "The material to search for (formula, name, or MP ID)",
            "material_type": "crystal/molecule/amorphous",
            "application": "semiconductor/battery/catalyst/etc",
            "expected_properties": ["list", "of", "expected", "properties"],
            "confidence": 0.95,
            "search_strategy": "direct_id/element_search/chemical_system",
            "context": {{
                "properties_of_interest": ["band_gap", "formation_energy", "elastic_constants"],
                "computational_focus": "electronic/structural/thermal"
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
            
            # Extract JSON from response
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(1))
            else:
                analysis = json.loads(content)
            
            return QueryAnalysisResult(
                material_query=analysis["material_query"],
                material_type=analysis["material_type"],
                application=analysis["application"],
                expected_properties=analysis["expected_properties"],
                confidence=analysis["confidence"],
                search_strategy=analysis["search_strategy"],
                context=analysis["context"]
            )
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._simple_query_analysis(user_input)
    
    def _simple_query_analysis(self, user_input: str) -> QueryAnalysisResult:
        """Simple fallback query analysis."""
        # Extract material query using regex
        formula_pattern = r'([A-Z][a-z]?\d*)'
        elements_found = re.findall(formula_pattern, user_input)
        
        # Validate elements
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'
        }
        
        if elements_found:
            # Filter valid elements
            valid_elements_found = []
            for element in elements_found:
                element_symbol = re.sub(r'\d+', '', element)
                if element_symbol in valid_elements:
                    valid_elements_found.append(element)
            
            if valid_elements_found:
                query = ''.join(valid_elements_found)
                search_strategy = "element_search"
            else:
                query = self._extract_reasonable_query(user_input)
                search_strategy = "chemical_system"
        elif user_input.startswith('mp-'):
            query = user_input
            search_strategy = "direct_id"
        else:
            query = self._extract_reasonable_query(user_input)
            search_strategy = "chemical_system"
        
        return QueryAnalysisResult(
            material_query=query,
            material_type="crystal",
            application="general",
            expected_properties=["structure", "properties"],
            confidence=0.7,
            search_strategy=search_strategy,
            context={
                "properties_of_interest": ["structure"],
                "computational_focus": "structural"
            }
        )
    
    def _extract_reasonable_query(self, user_input: str) -> str:
        """Extract a reasonable query from user input."""
        material_keywords = {
            'silicon', 'si', 'copper', 'cu', 'iron', 'fe', 'aluminum', 'al', 'gold', 'au',
            'lithium', 'li', 'sodium', 'na', 'potassium', 'k', 'calcium', 'ca'
        }
        
        input_lower = user_input.lower()
        for keyword in material_keywords:
            if keyword in input_lower:
                return keyword.upper()
        
        return "Si"  # Default to silicon
    
    def _handle_data_query(self, context: AgentContext) -> DataQueryResult:
        """Step 2: Query material data from databases."""
        print("🔍 Querying material data...")
        
        query_result = context.step_results[AgentStep.QUERY_ANALYSIS.value]
        
        try:
            from pymatgen.ext.matproj import MPRester
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            
            with MPRester(self.mp_api_key) as m:
                # Use search strategy from query analysis
                if query_result.search_strategy == "direct_id":
                    docs = m.summary.search(material_ids=[query_result.material_query])
                elif query_result.search_strategy == "element_search":
                    docs = m.summary.search(elements=query_result.material_query)
                else:  # chemical_system
                    docs = m.summary.search(chemsys=query_result.material_query)
                
                if not docs:
                    raise ValueError(f"No material found for '{query_result.material_query}'")
                
                # Get the best match
                doc = docs[0]
                structure = doc["structure"]
                sga = SpacegroupAnalyzer(structure)
                
                return DataQueryResult(
                    material_id=doc["material_id"],
                    formula_pretty=doc["formula_pretty"],
                    structure_data={
                        "space_group": sga.get_space_group_symbol(),
                        "crystal_system": sga.get_crystal_system(),
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
                    properties={
                        "volume": doc.get("volume"),
                        "density": doc.get("density"),
                        "band_gap": doc.get("band_gap"),
                        "formation_energy": doc.get("formation_energy_per_atom"),
                    },
                    source="Materials Project",
                    confidence=0.9
                )
                
        except Exception as e:
            print(f"Data query failed: {e}")
            raise
    
    def _handle_dsl_converter(self, context: AgentContext) -> DSLConverterResult:
        """Step 3: Convert material data to AtomForge DSL."""
        print("🔄 Converting to AtomForge DSL...")
        
        query_result = context.step_results[AgentStep.QUERY_ANALYSIS.value]
        data_result = context.step_results[AgentStep.DATA_QUERY.value]
        
        if self.llm_client:
            return self._llm_dsl_conversion(query_result, data_result)
        else:
            return self._template_dsl_conversion(query_result, data_result)
    
    def _llm_dsl_conversion(self, query_result: QueryAnalysisResult, data_result: DataQueryResult) -> DSLConverterResult:
        """Use LLM for DSL conversion."""
        dsl_prompt = f"""
        You are an expert AtomForge DSL generator. Generate a complete AtomForge DSL specification
        for the following material based on the provided data.
        
        Material Query: {query_result.material_query}
        Material Type: {query_result.material_type}
        Application: {query_result.application}
        Expected Properties: {query_result.expected_properties}
        
        Material Data:
        - Material ID: {data_result.material_id}
        - Formula: {data_result.formula_pretty}
        - Space Group: {data_result.structure_data['space_group']}
        - Crystal System: {data_result.structure_data['crystal_system']}
        - Lattice Parameters: {data_result.structure_data['lattice']}
        
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
              functional: "PBE",
              energy_cutoff: 520,
              k_point_density: 1000.0
            }},
            convergence_criteria: {{
              energy_tolerance: 1e-5,
              force_tolerance: 0.01,
              stress_tolerance: 0.1
            }},
            target_properties: {{
              formation_energy: true,
              band_gap: true,
              elastic_constants: false
            }}
          }}
          provenance {{
            source = "Materials Project",
            method = "VASP DFT calculation",
            doi = ""
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
            
            # Clean up formatting
            dsl_code = self._clean_markdown_formatting(dsl_code)
            
            return DSLConverterResult(
                dsl_code=dsl_code,
                generation_method="llm",
                parameters_used={
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "query_context": query_result.context
                },
                confidence=0.85
            )
            
        except Exception as e:
            print(f"LLM DSL conversion failed: {e}")
            return self._template_dsl_conversion(query_result, data_result)
    
    def _template_dsl_conversion(self, query_result: QueryAnalysisResult, data_result: DataQueryResult) -> DSLConverterResult:
        """Template-based DSL conversion fallback."""
        # Generate a simple template-based DSL
        dsl_code = f"""#atomforge_version "1.0";
atom_spec {data_result.formula_pretty.replace(' ', '_')} {{
  header {{
    dsl_version = "1.0",
    title = "{data_result.formula_pretty}",
    created = {datetime.now().strftime('%Y-%m-%d')},
    uuid = "{str(uuid.uuid4())}"
  }},
  description = "{query_result.material_query} {query_result.material_type} for {query_result.application}",
  units {{
    system = "crystallographic_default",
    length = angstrom,
    angle = degree
  }},
  lattice {{
    type = {data_result.structure_data['crystal_system']},
    a = {data_result.structure_data['lattice']['a']:.6f},
    b = {data_result.structure_data['lattice']['b']:.6f},
    c = {data_result.structure_data['lattice']['c']:.6f},
    alpha = {data_result.structure_data['lattice']['alpha']:.6f},
    beta = {data_result.structure_data['lattice']['beta']:.6f},
    gamma = {data_result.structure_data['lattice']['gamma']:.6f}
  }},
  symmetry {{
    space_group = "{data_result.structure_data['space_group']}",
    origin_choice = 1
  }},
  basis {{
    // Template-based conversion - atomic sites would be added here
  }},
  property_validation {{
    computational_backend: VASP {{
      functional: "PBE",
      energy_cutoff: 520,
      k_point_density: 1000.0
    }},
    convergence_criteria: {{
      energy_tolerance: 1e-5,
      force_tolerance: 0.01,
      stress_tolerance: 0.1
    }},
    target_properties: {{
      formation_energy: true,
      band_gap: true,
      elastic_constants: false
    }}
  }},
  provenance {{
    source = "Materials Project",
    method = "VASP DFT calculation",
    doi = ""
  }}
}};"""
        
        return DSLConverterResult(
            dsl_code=dsl_code,
            generation_method="template",
            parameters_used={
                "template": "basic_crystal",
                "data_source": "Materials Project"
            },
            confidence=0.6
        )
    
    def _handle_dsl_validator(self, context: AgentContext) -> DSLValidatorResult:
        """Step 4: Validate and fix generated DSL."""
        print("Validating DSL...")
        
        converter_result = context.step_results[AgentStep.DSL_CONVERTER.value]
        dsl_code = converter_result.dsl_code
        
        # Try to parse the DSL
        try:
            # Import parser if available
            try:
                from ..parser.atomforge_parser import parse_atomforge_string
                parse_atomforge_string(dsl_code)
                is_valid = True
                validation_message = "DSL is valid"
                errors_found = []
                warnings = []
            except ImportError:
                # Fallback validation
                is_valid = self._basic_dsl_validation(dsl_code)
                validation_message = "DSL passed basic validation" if is_valid else "DSL has syntax errors"
                errors_found = [] if is_valid else ["Parser not available for detailed validation"]
                warnings = []
            
            # If invalid and LLM is available, try to fix it
            if not is_valid and self.llm_client:
                try:
                    fixed_dsl = self._fix_dsl_with_llm(dsl_code, validation_message)
                    
                    # Validate the fixed DSL
                    try:
                        parse_atomforge_string(fixed_dsl)
                        return DSLValidatorResult(
                            is_valid=True,
                            validation_message="DSL fixed and validated",
                            fixed_dsl=fixed_dsl,
                            errors_found=[],
                            warnings=["DSL was automatically fixed"]
                        )
                    except Exception:
                        return DSLValidatorResult(
                            is_valid=False,
                            validation_message="DSL still has errors after fixing",
                            fixed_dsl=fixed_dsl,
                            errors_found=["Could not fix all syntax errors"],
                            warnings=[]
                        )
                        
                except Exception as e:
                    return DSLValidatorResult(
                        is_valid=False,
                        validation_message=f"Could not fix DSL: {e}",
                        fixed_dsl=None,
                        errors_found=[str(e)],
                        warnings=[]
                    )
            
            return DSLValidatorResult(
                is_valid=is_valid,
                validation_message=validation_message,
                fixed_dsl=None,
                errors_found=errors_found,
                warnings=warnings
            )
            
        except Exception as e:
            return DSLValidatorResult(
                is_valid=False,
                validation_message=f"Validation error: {e}",
                fixed_dsl=None,
                errors_found=[str(e)],
                warnings=[]
            )
    
    def _basic_dsl_validation(self, dsl_code: str) -> bool:
        """Basic DSL validation without parser."""
        # Check for basic syntax elements
        required_elements = [
            "#atomforge_version",
            "atom_spec",
            "header",
            "lattice",
            "symmetry",
            "basis",
            "property_validation",
            "provenance"
        ]
        
        for element in required_elements:
            if element not in dsl_code:
                return False
        
        # Check for balanced braces
        if dsl_code.count('{') != dsl_code.count('}'):
            return False
        
        return True
    
    def _fix_dsl_with_llm(self, dsl_code: str, error_message: str) -> str:
        """Use LLM to fix DSL syntax errors."""
        fix_prompt = f"""
        The following AtomForge DSL has validation errors:
        
        DSL Code:
        {dsl_code}
        
        Error: {error_message}
        
        Please fix the DSL code to follow the AtomForge grammar exactly.
        Generate only the corrected AtomForge DSL code.
        """
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
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
    
    def _create_final_response(self, context: AgentContext) -> Tuple[str, Dict[str, Any]]:
        """Create the final response from successful workflow."""
        validator_result = context.step_results[AgentStep.DSL_VALIDATOR.value]
        
        # Use fixed DSL if available, otherwise use original
        final_dsl = validator_result.fixed_dsl or context.step_results[AgentStep.DSL_CONVERTER.value].dsl_code
        
        metadata = {
            "workflow_status": context.status.value,
            "workflow_metadata": context.metadata,
            "step_results": {
                "query_analysis": context.step_results[AgentStep.QUERY_ANALYSIS.value].__dict__,
                "data_query": context.step_results[AgentStep.DATA_QUERY.value].__dict__,
                "dsl_converter": context.step_results[AgentStep.DSL_CONVERTER.value].__dict__,
                "dsl_validator": validator_result.__dict__
            },
            "final_result": {
                "dsl": final_dsl,
                "is_valid": validator_result.is_valid,
                "validation_message": validator_result.validation_message
            }
        }
        
        return final_dsl, metadata
    
    def _create_partial_response(self, context: AgentContext) -> Tuple[str, Dict[str, Any]]:
        """Create partial response when workflow fails."""
        partial_dsl = "#atomforge_version \"1.0\";\n// Partial DSL due to workflow failure\n"
        
        metadata = {
            "workflow_status": context.status.value,
            "error_message": context.error_message,
            "workflow_metadata": context.metadata,
            "step_results": context.step_results,
            "partial_result": {
                "dsl": partial_dsl,
                "is_valid": False,
                "validation_message": "Workflow failed - partial result"
            }
        }
        
        return partial_dsl, metadata

# Convenience function
def run_atomforge_agent(user_input: str, openai_api_key: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Run the AtomForge agent with the given input."""
    agent = AtomForgeAgent(openai_api_key)
    return agent.run(user_input) 