#!/usr/bin/env python3
"""
AtomForge Multi-Agent System Implementation

This module implements the multi-agent architecture for processing materials science data
and converting them into AtomForge DSL. Based on the architecture defined in the documentation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import os
import tempfile
import logging
import uuid

# Import AtomForge core modules
from crystal_v1_1 import Crystal, identity_hash, canonicalize, validate, from_cif, from_poscar
from atomforge_interop import from_cif as interop_from_cif, from_poscar as interop_from_poscar  # Alternative imports
from crystal_edit import substitute, vacancy, interstitial, make_supercell, to_poscar, to_cif, from_pymatgen
from crystal_calc import prepare_calc, estimate_kmesh
from atomforge_generator import generate_atomforge_program
from atomforge_parser import AtomForgeParser
from atomforge_compiler import AtomForgeCompiler
from atomforge_database_connector import (
    match_database,
    select_variant,
    DatabaseMatch,
    MatchReport,
    SelectionReport,
    MaterialsProjectConnector,
    CODConnector,
    ICSDConnector,
)
from atomforge_generator import generate_atomforge_program

try:
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    jsonschema = None
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# LLM SERVICE
# ============================================================================


class LLMServiceError(RuntimeError):
    """Raised when the LLM backend is unavailable or returns invalid data."""


class LLMService:
    """
    Wrapper around the configured LLM provider.

    Phase 4 requirement: eliminate deterministic defaults and route reasoning
    through an LLM with schema-constrained output.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("ATOMFORGE_REASONING_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.timeout = timeout

        if not self.api_key:
            raise LLMServiceError(
                "OPENAI_API_KEY must be set for MaterialsReasoningAgent. "
                "Set it in the environment or pass api_key explicitly."
            )

        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment specific
            raise LLMServiceError(
                "The 'openai' package is required for LLM interactions."
            ) from exc

        self._client = openai.OpenAI(api_key=self.api_key)

    def extract_material_query(self, user_input: str) -> Dict[str, Any]:
        """
        Transform raw user input into a structured material query using the LLM.

        Returns:
            Dict with keys: source_type, input_data, required_properties, constraints.
        """
        system_prompt = (
            "You are the Materials Reasoning Agent for AtomForge. "
            "Given raw user input about a material, respond with JSON matching:\n"
            "{\n"
            '  \"source_type\": \"natural_language\",\n'
            '  \"input_data\": string,\n'
            '  \"required_properties\": [string, ...],\n'
            '  \"constraints\": {\n'
            '    \"formula\": string,\n'
            '    \"elements\": [string, ...],\n'
            '    \"structure_type\": string,\n'
            '    \"processing_notes\": string\n'
            "  }\n"
            "}\n"
            "Do not add other keys. Use lowercase for structure_type. "
            "Provide a valid chemical formula recognizable by Materials Project; "
            "if uncertain, supply the closest plausible stoichiometric formula."
        )
        user_prompt = (
            "Material request:\n"
            f"{user_input.strip()}\n\n"
            "Respond ONLY with valid JSON."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                timeout=self.timeout,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=600,
            )
        except Exception as exc:  # pragma: no cover - network errors
            raise LLMServiceError(f"LLM request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMServiceError(
                f"Unexpected LLM response structure: {response}"
            ) from exc

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMServiceError(
                f"LLM response was not valid JSON: {content}"
            ) from exc

        required_keys = {
            "source_type",
            "input_data",
            "required_properties",
            "constraints",
        }
        missing = required_keys.difference(payload.keys())
        if missing:
            raise LLMServiceError(
                f"LLM response missing required keys: {', '.join(sorted(missing))}"
            )

        if not isinstance(payload.get("required_properties", []), list):
            raise LLMServiceError("required_properties must be a list.")

        constraints = payload.get("constraints", {})
        if not isinstance(constraints, dict):
            raise LLMServiceError("constraints must be a JSON object.")

        formula = constraints.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            raise LLMServiceError("constraints.formula must be a non-empty string.")

        if "elements" in constraints and not isinstance(constraints["elements"], list):
            raise LLMServiceError("constraints.elements must be an array.")

        return payload

    def generate_crystal_schema(self, context: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AtomForge crystal JSON adhering to the provided schema.
        """
        required_fields = schema.get("required", [])
        property_keys = list((schema.get("properties") or {}).keys())
        schema_summary = json.dumps(
            {
                "required": required_fields,
                "properties": property_keys,
            }
        )

        system_prompt = (
            "You are the AtomForge DSL Formatter Agent. "
            "Produce a JSON object describing a crystal that strictly conforms to the provided schema summary. "
            "Do not include any properties outside the schema."
        )
        user_prompt = (
            "Crystal context:\n"
            f"{context}\n\n"
            "Schema summary:\n"
            f"{schema_summary}\n\n"
            "Respond ONLY with valid JSON."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                timeout=self.timeout,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=900,
            )
        except Exception as exc:  # pragma: no cover - network errors
            raise LLMServiceError(f"LLM request failed: {exc}") from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMServiceError(
                f"Unexpected LLM response structure: {response}"
            ) from exc

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMServiceError(
                f"LLM response was not valid JSON: {content}"
            ) from exc

        return payload

# ============================================================================
# MESSAGE DATA STRUCTURES
# ============================================================================

@dataclass
class AgentMessage:
    """Base message structure for inter-agent communication"""
    message_id: str
    timestamp: str
    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialQuery:
    """Structured material query"""
    source_type: str  # "natural_language", "file", "database_id"
    input_data: str
    required_properties: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result from agent processing"""
    success: bool
    data: Any
    confidence: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# MATERIALS REASONING AGENT
# ============================================================================

class MaterialsReasoningAgent:
    """Agent for interpreting and extracting information from materials science inputs"""
    
    SUPPORTED_SOURCE_TYPES = {"natural_language", "file", "database_id"}

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.name = "MaterialsReasoningAgent"
        self.llm_service = llm_service or LLMService()
    
    def process(self, input_message: AgentMessage) -> AgentMessage:
        """
        Process raw user input and extract material information
        
        Args:
            input_message: AgentMessage containing raw user input
            
        Returns:
            AgentMessage with structured material query
        """
        raw_input = input_message.payload.get("input", "")
        input_type = input_message.payload.get("type", "natural_language")
        
        errors: List[str] = []
        warnings: List[str] = []

        try:
        # Extract information based on input type
        if input_type == "natural_language":
            material_query = self._parse_natural_language(raw_input)
        elif input_type == "cif":
                material_query = MaterialQuery(source_type="file", input_data=raw_input)
        elif input_type == "database_id":
            material_query = MaterialQuery(
                    source_type="database_id", input_data=raw_input
            )
        else:
                material_query = MaterialQuery(source_type=input_type, input_data=raw_input)
                warnings.append(f"Unrecognized input_type '{input_type}' passed through.")

            validation_warnings, validation_errors = self._validate_material_query(
                material_query
            )
            warnings.extend(validation_warnings)
            errors.extend(validation_errors)
        except LLMServiceError as exc:
            material_query = None
            errors.append(str(exc))

        success = not errors and material_query is not None
        confidence = 0.90 if success else 0.0

        result = ProcessingResult(
            success=success,
            data=material_query,
            confidence=confidence,
            warnings=warnings,
            errors=errors,
        )
        
        return AgentMessage(
            message_id=f"msg_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            sender=self.name,
            receiver="DatabaseCoordinatorAgent",
            message_type="material_query",
            payload={"result": result},
            metadata={"input_type": input_type}
        )
    
    def _parse_natural_language(self, text: str) -> MaterialQuery:
        """Parse natural language description using the LLM service."""
        llm_payload = self.llm_service.extract_material_query(text)

        source_type = llm_payload.get("source_type", "natural_language")
        input_data = llm_payload.get("input_data", text)
        required_properties = llm_payload.get("required_properties", [])
        constraints = llm_payload.get("constraints", {})
        
        return MaterialQuery(
            source_type=source_type,
            input_data=input_data,
            required_properties=required_properties,
            constraints=constraints,
        )

    def _validate_material_query(
        self, query: MaterialQuery
    ) -> Tuple[List[str], List[str]]:
        """
        Validate material query structure.

        Returns:
            Tuple[List[str], List[str]] -> (warnings, errors)
        """
        warnings: List[str] = []
        errors: List[str] = []

        if not query.input_data or not str(query.input_data).strip():
            errors.append("material query input_data is empty.")

        if query.source_type not in self.SUPPORTED_SOURCE_TYPES:
            warnings.append(
                f"source_type '{query.source_type}' is not in supported set "
                f"{sorted(self.SUPPORTED_SOURCE_TYPES)}."
            )

        elements = query.constraints.get("elements")
        if elements is not None and not isinstance(elements, list):
            errors.append("constraints.elements must be a list if provided.")

        formula = query.constraints.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            errors.append("constraints.formula must be a non-empty string.")

        structure_type = query.constraints.get("structure_type")
        if structure_type is None:
            warnings.append("constraints.structure_type missing.")
        elif not isinstance(structure_type, str) or not structure_type.strip():
            errors.append("constraints.structure_type must be a non-empty string.")

        if not query.required_properties:
            warnings.append("required_properties is empty; downstream agents may need defaults.")

        return warnings, errors


# ============================================================================
# DATABASE COORDINATOR AGENT
# ============================================================================

class DatabaseCoordinatorAgent:
    """Agent for orchestrating database queries and material data retrieval"""
    
    def __init__(self):
        self.name = "DatabaseCoordinatorAgent"
        requested_sources = os.getenv("ATOMFORGE_DATABASE_SOURCES")
        if requested_sources:
            self.databases = [src.strip() for src in requested_sources.split(",") if src.strip()]
        else:
            self.databases = ["MP"]
        self.logger = logging.getLogger(self.name)

        self.mp_api_key = os.getenv("MP_API_KEY", "MP_API_KEY_REDACTED")
        self.mp_connector: Optional[MaterialsProjectConnector] = None
        self.cod_connector: Optional[CODConnector] = None
        self.icsd_connector: Optional[ICSDConnector] = None

        try:
            if self.mp_api_key:
                self.mp_connector = MaterialsProjectConnector(self.mp_api_key)
        except Exception as exc:  # pragma: no cover - network/API init
            self.logger.warning(f"Failed to initialize MaterialsProjectConnector: {exc}")

        try:
            self.cod_connector = CODConnector()
        except Exception as exc:  # pragma: no cover - network/API init
            self.logger.warning(f"Failed to initialize CODConnector: {exc}")

        try:
            icsd_api_key = os.getenv("ICSD_API_KEY")
            icsd_username = os.getenv("ICSD_USERNAME")
            icsd_password = os.getenv("ICSD_PASSWORD")
            self.icsd_connector = ICSDConnector(
                api_key=icsd_api_key,
                username=icsd_username,
                password=icsd_password,
            )
        except Exception as exc:  # pragma: no cover - network/API init
            self.logger.warning(f"Failed to initialize ICSDConnector: {exc}")
    
    def process(self, input_message: AgentMessage) -> AgentMessage:
        """
        Query databases and retrieve material data
        
        Args:
            input_message: AgentMessage containing material query
            
        Returns:
            AgentMessage with retrieved material data
        """
        material_query: MaterialQuery = input_message.payload["result"].data
        
        warnings: List[str] = []
        errors: List[str] = []
        payload_data: Dict[str, Any] = {}

        source_type = material_query.source_type
        input_data = material_query.input_data
        
        if source_type == "file":
            crystal_data = self._read_file(input_data)
            if crystal_data is None:
                errors.append(f"Unsupported file format for path '{input_data}'.")
        else:
                payload_data = crystal_data
        elif source_type == "database_id":
            selected_match = self._fetch_by_id(input_data)
            if selected_match:
                payload_data = self._package_match_payload(selected_match)
                if payload_data.get("crystal") is None:
                    errors.append(
                        f"Could not convert database match {selected_match.material_id} into Crystal."
                    )
            else:
                errors.append(f"No database entry found for id '{input_data}'.")
        else:
            lookup_term = self._derive_lookup_term(material_query)

            try:
                match_report = match_database(lookup_term, sources=self.databases)
            except Exception as exc:  # pragma: no cover - network errors
                errors.append(f"Database search failed: {exc}")
                match_report = None

            if match_report:
                payload_data["match_report"] = match_report
                aggregated_matches = self._flatten_matches(match_report)

                if aggregated_matches:
                    selected_match, selection_report = self._select_variant(aggregated_matches)
                    payload_data["selected_match"] = selected_match
                    payload_data["selection_report"] = selection_report

                    crystal = self._convert_match_to_crystal(selected_match)
                    if crystal:
                        payload_data["crystal"] = crystal
                    else:
                        warnings.append(
                            f"Selected match {selected_match.material_id} lacks structure data."
                        )
                else:
                    errors.append("No matches returned from database search.")
            else:
                errors.append("Database match report was not generated.")

        success = not errors and bool(payload_data)

        result = ProcessingResult(
            success=success,
            data=payload_data if success else None,
            confidence=0.9 if success else 0.0,
            warnings=warnings,
            errors=errors,
        )
        
        return AgentMessage(
            message_id=f"msg_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            sender=self.name,
            receiver="StructureProcessorAgent",
            message_type="crystal_data",
            payload={"result": result}
        )
    
    def _fetch_by_id(self, db_id: str) -> Optional[DatabaseMatch]:
        """Fetch material by database ID using configured connectors."""
        try:
            if db_id.startswith("mp-"):
                return self._fetch_mp_entry(db_id)
            if db_id.lower().startswith("cod:") and self.cod_connector:
                cod_id = db_id.split(":", 1)[-1]
                return self.cod_connector.get_entry_by_id(cod_id)
            if db_id.lower().startswith("icsd:") and self.icsd_connector:
                formula = db_id.split(":", 1)[-1]
                matches = self.icsd_connector.search_by_formula(formula)
                if matches:
                    return matches[0]
        except Exception as exc:  # pragma: no cover - network errors
            self.logger.error(f"Database ID lookup failed for {db_id}: {exc}")
        return None
    
    def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Read material data from file"""
        path = Path(file_path)
        if path.suffix == ".cif":
            return {"format": "cif", "path": file_path}
        elif path.suffix in [".poscar", "POSCAR"]:
            return {"format": "poscar", "path": file_path}
        return None
    
    def _derive_lookup_term(self, query: MaterialQuery) -> str:
        """Derive lookup formula or term for database search."""
        constraints = query.constraints or {}
        if constraints.get("formula"):
            return constraints["formula"]
        elements = constraints.get("elements")
        if isinstance(elements, list) and elements:
            return "".join(elements)
        return query.input_data

    def _flatten_matches(self, report: MatchReport) -> List[DatabaseMatch]:
        """Flatten all matches from a MatchReport."""
        if not report or not report.matches_found:
            return []
        matches: List[DatabaseMatch] = []
        for match_list in report.matches_found.values():
            matches.extend(match_list)
        return matches

    def _select_variant(
        self, candidates: List[DatabaseMatch]
    ) -> Tuple[DatabaseMatch, SelectionReport]:
        """Select best variant using configured policy."""
        selected, selection_report = select_variant(
            candidates, policy="prefer_low_hull_then_experimental"
        )
        return selected, selection_report

    def _convert_match_to_crystal(self, match: DatabaseMatch) -> Optional[Crystal]:
        """Convert a DatabaseMatch into a Crystal object if possible."""
        if not match:
            return None

        structure = match.structure_data
        if structure is not None:
            try:
                return from_pymatgen(structure)
            except Exception as exc:
                self.logger.warning(
                    f"Failed to convert pymatgen structure for {match.material_id}: {exc}"
                )

        cif_content = match.metadata.get("cif_content") if match.metadata else None
        if cif_content:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w+", suffix=".cif", delete=False
                ) as tmp:
                    tmp.write(cif_content)
                    tmp.flush()
                    tmp_path = tmp.name
                if tmp_path:
                    crystal, _ = from_cif(tmp_path)
                    return crystal
            except Exception as exc:
                self.logger.warning(
                    f"Failed to parse CIF content for {match.material_id}: {exc}"
                )
            finally:
                if tmp_path:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass

        return None

    def _package_match_payload(self, match: DatabaseMatch) -> Dict[str, Any]:
        """Wrap a single match into the payload structure."""
        payload: Dict[str, Any] = {
            "selected_match": match,
            "selection_report": None,
        }
        crystal = self._convert_match_to_crystal(match)
        if crystal:
            payload["crystal"] = crystal
        return payload

    def _fetch_mp_entry(self, material_id: str) -> Optional[DatabaseMatch]:
        """Fetch Materials Project entry by material id."""
        if not self.mp_api_key:
            self.logger.warning("MP_API_KEY not configured; cannot fetch Materials Project entries.")
            return None

        try:
            import pymatgen.ext.matproj as mp  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            self.logger.error(
                "pymatgen is required for Materials Project lookups by id.", exc_info=exc
            )
            return None

        try:
            with mp.MPRester(self.mp_api_key) as m:
                docs = m.summary.search(material_ids=[material_id])
        except Exception as exc:  # pragma: no cover - API failure
            self.logger.error(f"Materials Project API lookup failed for {material_id}: {exc}")
            return None

        if not docs:
            return None

        doc = docs[0]
        return self._create_mp_match(doc)

    def _create_mp_match(self, doc: Any) -> DatabaseMatch:
        """Create DatabaseMatch from Materials Project summary document."""
        if isinstance(doc, dict):
            material_id = doc.get("material_id", "")
            formula_pretty = doc.get("formula_pretty", "")
            energy_above_hull = doc.get("energy_above_hull", None)
            formation_energy_per_atom = doc.get("formation_energy_per_atom", None)
            band_gap = doc.get("band_gap", None)
            density = doc.get("density", None)
            volume = doc.get("volume", None)
            is_stable = doc.get("is_stable", None)
            is_metal = doc.get("is_metal", None)
            is_magnetic = doc.get("is_magnetic", None)
            is_experimental = doc.get("is_experimental", False)
            theoretical = doc.get("theoretical", True)
            nsites = doc.get("nsites", None)
            symmetry = doc.get("symmetry", None)
            structure = doc.get("structure", None)
        else:
            material_id = getattr(doc, "material_id", "")
            formula_pretty = getattr(doc, "formula_pretty", "")
            energy_above_hull = getattr(doc, "energy_above_hull", None)
            formation_energy_per_atom = getattr(doc, "formation_energy_per_atom", None)
            band_gap = getattr(doc, "band_gap", None)
            density = getattr(doc, "density", None)
            volume = getattr(doc, "volume", None)
            is_stable = getattr(doc, "is_stable", None)
            is_metal = getattr(doc, "is_metal", None)
            is_magnetic = getattr(doc, "is_magnetic", None)
            is_experimental = getattr(doc, "is_experimental", False)
            theoretical = getattr(doc, "theoretical", True)
            nsites = getattr(doc, "nsites", None)
            symmetry = getattr(doc, "symmetry", None)
            structure = getattr(doc, "structure", None)

        return DatabaseMatch(
            database_name="MP",
            material_id=material_id,
            formula=formula_pretty,
            similarity_score=1.0,
            provenance={
                "database": "MP",
                "id": material_id,
                "retrieved_at": datetime.now().isoformat(),
                "api_version": "v2",
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
                "crystal_system": symmetry.get("crystal_system", None) if symmetry else None,
            },
            metadata={
                "is_experimental": is_experimental,
                "theoretical": theoretical,
                "nsites": nsites,
                "spacegroup": symmetry.get("symbol", None) if symmetry else None,
                "crystal_system": symmetry.get("crystal_system", None) if symmetry else None,
                "symmetry": symmetry,
            },
            structure_data=structure,
        )


# ============================================================================
# STRUCTURE PROCESSOR AGENT
# ============================================================================

class StructureProcessorAgent:
    """Agent for processing and standardizing crystal structures"""
    
    def __init__(self):
        self.name = "StructureProcessorAgent"
    
    def process(self, input_message: AgentMessage) -> AgentMessage:
        """
        Process raw structure data into canonicalized crystal
        
        Args:
            input_message: AgentMessage containing raw structure data
            
        Returns:
            AgentMessage with canonicalized crystal
        """
        crystal_data = input_message.payload["result"].data 
        
        # Load crystal from file or data
        try:
            if "path" in crystal_data:
                file_path = crystal_data["path"]
                fmt = crystal_data["format"]
                
                if fmt == "cif":
                    crystal, _ = from_cif(file_path)
                elif fmt == "poscar":
                    crystal, _ = from_poscar(file_path)
                else:
                    raise ValueError(f"Unsupported format: {fmt}")
            else:
                # Use provided crystal data
                crystal = crystal_data.get("crystal", None)
            
            # Canonicalize structure
            crystal_canon, canon_report = canonicalize(crystal)
            
            # Validate structure
            validation = validate(crystal_canon)
            
            # Create provenance
            provenance = {
                "source": crystal_data.get("source", "unknown"),
                "canonicalized_at": datetime.now().isoformat(),
                "validation": validation.ok,
                "identity_hash": crystal_canon.provenance.hash
            }
            
            result = ProcessingResult(
                success=validation.ok,
                data={
                    "crystal": crystal_canon,
                    "canon_report": canon_report,
                    "validation": validation,
                    "provenance": provenance
                },
                confidence=1.0 if validation.ok else 0.5,
                warnings=validation.warnings,
                errors=validation.errors
            )
            
        except Exception as e:
            result = ProcessingResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[str(e)]
            )
        
        return AgentMessage(
            message_id=f"msg_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            sender=self.name,
            receiver="VariantSelectorAgent",
            message_type="canonicalized_crystal",
            payload={"result": result}
        )
    
    def load_crystal(self, crystal_data: Dict[str, Any]) -> Optional[Crystal]:
        """Load crystal from data dictionary"""
        # This would parse crystal data into Crystal object
        return None


# ============================================================================
# VARIANT SELECTOR AGENT
# ============================================================================

class VariantSelectorAgent:
    """Agent for choosing the best structural variant from available options"""
    
    def __init__(self):
        self.name = "VariantSelectorAgent"
    
    def process(self, input_message: AgentMessage) -> AgentMessage:
        """
        Select best variant from available options
        
        Args:
            input_message: AgentMessage containing canonicalized crystal
            
        Returns:
            AgentMessage with selected variant
        """
        crystal_data = input_message.payload["result"].data
        crystal = crystal_data["crystal"]
        
        selection_report = crystal_data.get("selection_report")
        selection_rationale = "Best match from available variants"

        if isinstance(selection_report, SelectionReport):
            selection_rationale = selection_report.selection_reason
        else:
            # Apply variant selection policy if report not provided
            selected_match = crystal_data.get("selected_match")
            if selected_match:
                selection_rationale = (
                    f"Selected via default policy from {selected_match.database_name}"
                )
            else:
                crystal = self._select_best_variant(crystal)
        
        result = ProcessingResult(
            success=True,
            data={
                "crystal": crystal,
                "selection_rationale": selection_rationale,
                "selection_report": selection_report,
                "selected_match": crystal_data.get("selected_match"),
            },
            confidence=0.95
        )
        
        return AgentMessage(
            message_id=f"msg_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            sender=self.name,
            receiver="DSLFormatterAgent",
            message_type="selected_variant",
            payload={"result": result}
        )
    
    def _select_best_variant(self, crystal: Crystal) -> Crystal:
        """Select best variant based on policy"""
        # Simplified - return crystal as-is
        # In production, this would:
        # 1. Match against database
        # 2. Apply selection policy (energy hull, experimental, etc.)
        # 3. Return best variant
        return crystal


# ============================================================================
# DSL FORMATTER AGENT
# ============================================================================

class DSLFormatterAgent:
    """Agent for transforming crystal data into AtomForge DSL syntax"""
    
    def __init__(self):
        self.name = "DSLFormatterAgent"
        self.logger = logging.getLogger(self.name)
        try:
        self.parser = AtomForgeParser()
        except Exception as exc:
            self.logger.warning(
                "AtomForgeParser initialization failed (%s). "
                "Proceeding without parser; DSL validation will rely on downstream QA.",
                exc,
            )
            self.parser = None

        self.llm_service: Optional[LLMService] = None
        self.schema: Optional[Dict[str, Any]] = None
        schema_path = Path(__file__).resolve().parent.parent / "atomforge_crystal_schema.json"
        try:
            if schema_path.exists():
                self.schema = json.loads(schema_path.read_text())
            else:
                self.logger.warning(
                    "AtomForge crystal schema not found at %s; LLM fallback will be disabled.",
                    schema_path,
                )
        except Exception as exc:
            self.logger.warning(
                "Failed to load AtomForge crystal schema (%s); LLM fallback will be disabled.",
                exc,
            )
            self.schema = None
        if jsonschema is None:
            self.logger.warning(
                "jsonschema package not available; LLM fallback validation is disabled."
            )
    
    def process(self, input_message: AgentMessage) -> AgentMessage:
        """
        Generate AtomForge DSL from crystal structure
        
        Args:
            input_message: AgentMessage containing selected crystal variant
            
        Returns:
            AgentMessage with AtomForge DSL program
        """
        crystal_data = input_message.payload["result"].data
        crystal = crystal_data["crystal"]
        
        # Generate DSL program
        dsl_program = self._generate_dsl(crystal)
        
        result = ProcessingResult(
            success=dsl_program is not None,
            data={"dsl_program": dsl_program},
            confidence=0.9
        )
        
        if not dsl_program:
            result.errors.append("Failed to generate DSL program")
        
        return AgentMessage(
            message_id=f"msg_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            sender=self.name,
            receiver="QAVerifierAgent",
            message_type="dsl_program",
            payload={"result": result}
        )
    
    def _generate_dsl(self, crystal: Crystal) -> str:
        """
        Generate AtomForge DSL from crystal using rules-based generation.
        
        Phase 4: Uses crystal_to_dsl() function following FullLanguage.tex v2.1
        """
        try:
            dsl_program = generate_atomforge_program(
                crystal,
                material_name=getattr(crystal, "composition", "material"),
                description="Generated by AtomForge DSL Formatter Agent",
            )
            return dsl_program
        except Exception as exc:
            if not self.schema or jsonschema is None:
                raise ValueError(
                    f"Rules-based DSL generation failed ({exc}) and schema-based fallback is unavailable."
                ) from exc
            return self._generate_dsl_with_llm(crystal, str(exc))

    def _generate_dsl_with_llm(self, crystal: Crystal, root_error: str) -> str:
        """Fallback DSL generation via LLM with schema validation."""
        if self.llm_service is None:
            try:
                self.llm_service = LLMService()
            except LLMServiceError as exc:
                raise ValueError(
                    f"Rules-based DSL generation failed ({root_error}) and LLM fallback unavailable: {exc}"
                ) from exc

        context = self._build_crystal_context(crystal)
        schema_json = self.llm_service.generate_crystal_schema(context, self.schema)

        try:
            jsonschema.validate(schema_json, self.schema)  # type: ignore[arg-type]
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            raise ValueError(
                f"LLM produced JSON that failed schema validation: {exc.message}"
            ) from exc

        return self._schema_to_dsl(schema_json)

    def _build_crystal_context(self, crystal: Crystal) -> str:
        """Assemble a compact textual context describing the crystal."""
        lines = []
        composition = getattr(crystal, "composition", None)
        if composition is not None:
            lines.append(f"composition: {composition}")
        lattice = getattr(crystal, "lattice", None)
        if lattice is not None:
            lines.append(
                "lattice: "
                f"a={getattr(lattice, 'a', '?')}, "
                f"b={getattr(lattice, 'b', '?')}, "
                f"c={getattr(lattice, 'c', '?')}, "
                f"alpha={getattr(lattice, 'alpha', '?')}, "
                f"beta={getattr(lattice, 'beta', '?')}, "
                f"gamma={getattr(lattice, 'gamma', '?')}"
            )
        sites = getattr(crystal, "sites", None)
        if sites:
            site_summaries = []
            for site in sites[:8]:  # limit summary length
                species = getattr(site, "species", None)
                frac = getattr(site, "frac", None)
                site_summaries.append(f"species={species}, frac={frac}")
            lines.append("sites: " + "; ".join(site_summaries))
            if len(sites) > 8:
                lines.append(f"... (total sites: {len(sites)})")
        return "\n".join(lines)

    def _schema_to_dsl(self, schema_data: Dict[str, Any]) -> str:
        """Convert schema-compliant JSON into AtomForge DSL."""
        program_lines: List[str] = []
        composition_str = self._format_formula(schema_data.get("composition", {}))
        program_lines.append(f'atom_spec "{composition_str}" {{')
        program_lines.append("  header {")
        program_lines.append('    dsl_version = "2.1",')
        program_lines.append('    content_schema_version = "atomforge_crystal_v1.0",')
        program_lines.append(f'    uuid = "{uuid.uuid4()}",')
        program_lines.append(f'    title = "{composition_str}",')
        program_lines.append(f'    created = {datetime.now().date()}')
        program_lines.append("  },")
        program_lines.append('  description = "Generated via AtomForge LLM fallback",')
        program_lines.extend(self._schema_lattice_block(schema_data.get("lattice", {})))
        program_lines.extend(self._schema_symmetry_block(schema_data.get("symmetry", {})))
        program_lines.extend(self._schema_basis_block(schema_data.get("sites", [])))
        program_lines.extend(self._schema_provenance_block(schema_data.get("provenance", {})))
        program_lines.append("}")
        return "\n".join(program_lines)

    def _schema_lattice_block(self, lattice: Dict[str, Any]) -> List[str]:
        def _num(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return [
            "  lattice {",
            f'    type = triclinic,',
            f'    a = {_num(lattice.get("a"), 1.0):.6f},',
            f'    b = {_num(lattice.get("b"), 1.0):.6f},',
            f'    c = {_num(lattice.get("c"), 1.0):.6f},',
            f'    alpha = {_num(lattice.get("alpha"), 90.0):.1f},',
            f'    beta = {_num(lattice.get("beta"), 90.0):.1f},',
            f'    gamma = {_num(lattice.get("gamma"), 90.0):.1f}',
            "  },",
        ]

    def _schema_symmetry_block(self, symmetry: Dict[str, Any]) -> List[str]:
        space_group = symmetry.get("space_group", "P1")
        number = symmetry.get("number", 1)
        return [
            "  symmetry {",
            '    description = "Crystallographic symmetry",',
            f'    space_group = "{space_group}",',
            f'    origin_choice = 1',
            "  },",
        ]

    def _schema_basis_block(self, sites: List[Dict[str, Any]]) -> List[str]:
        lines = [
            "  basis {",
            '    description = "Atomic basis generated from schema-compliant JSON",',
        ]
        for idx, site in enumerate(sites, start=1):
            species = site.get("species", {})
            frac = site.get("frac", [0.0, 0.0, 0.0])
            wyckoff = site.get("wyckoff", "a")
            lines.append(f'    site "Site{idx}" {{')
            lines.append(f'      wyckoff = "{wyckoff}",')
            coords = []
            for value in frac[:3]:
                try:
                    coords.append(float(value))
                except (TypeError, ValueError):
                    coords.append(0.0)
            while len(coords) < 3:
                coords.append(0.0)
            lines.append(
                f'      position = ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}),'
            )
            lines.append("      frame = fractional,")
            species_entries = []
            for element, occupancy in species.items():
                species_entries.append(
                    f'{{ element = "{element}", occupancy = {occupancy:.6f} }}'
                )
            if not species_entries:
                species_entries.append('{ element = "X", occupancy = 1.000000 }')
            lines.append(f"      species = ({', '.join(species_entries)}),")
            lines.append('      adp_iso = 0.005,')
            lines.append(f'      label = "Site_{idx}"')
            lines.append("    },")
        lines.append("  },")
        return lines

    def _schema_provenance_block(self, provenance: Dict[str, Any]) -> List[str]:
        source = provenance.get("source", "LLM_fallback")
        method = provenance.get("method", "llm_schema_generation")
        return [
            "  provenance {",
            f'    source = "{source}",',
            f'    method = "{method}",',
            '    software = "AtomForge v2.1",',
            '    computational_details = "LLM fallback with schema validation",',
            '    doi = "",',
            '    url = "https://github.com/atomforge/atomforge"',
            "  }",
        ]

    def _format_formula(self, composition: Dict[str, Any]) -> str:
        reduced = composition.get("reduced", {})
        if not reduced:
            return "unknown_material"
        return "".join(f"{el}{count if count != 1 else ''}" for el, count in reduced.items())


# ============================================================================
# QA VERIFIER AGENT
# ============================================================================

class QAVerifierAgent:
    """Agent for performing quality assurance checks on generated DSL"""
    
    def __init__(self):
        self.name = "QAVerifierAgent"
    
    def process(self, input_message: AgentMessage) -> AgentMessage:
        """
        Perform QA checks on generated DSL
        
        Args:
            input_message: AgentMessage containing DSL program
            
        Returns:
            AgentMessage with validated DSL
        """
        dsl_data = input_message.payload["result"].data
        dsl_program = dsl_data["dsl_program"]
        
        # Validate DSL
        validation_results = self._validate_dsl(dsl_program)
        
        result = ProcessingResult(
            success=validation_results["valid"],
            data={
                "dsl_program": dsl_program,
                "validation": validation_results
            },
            confidence=validation_results.get("confidence", 0.85),
            warnings=validation_results.get("warnings", []),
            errors=validation_results.get("errors", [])
        )
        
        return AgentMessage(
            message_id=f"msg_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            sender=self.name,
            receiver="CompilerAgent",
            message_type="validated_dsl",
            payload={"result": result}
        )
    
    def _validate_dsl(self, dsl_program: str) -> Dict[str, Any]:
        """
        Validate DSL program with comprehensive checks.
        
        Phase 4: Enhanced validation following FullLanguage.tex v2.1
        """
        valid = True
        warnings = []
        errors = []
        
        # Check for required sections per FullLanguage.tex line 157-160
        required_sections = {
            "header": "atom_spec",
            "lattice": "lattice",
            "symmetry": "symmetry",
            "basis": "basis"
        }
        
        for section_name, keyword in required_sections.items():
            if keyword not in dsl_program:
                valid = False
                errors.append(f"Missing required section: {section_name}")
        
        # Check for proper program structure
        if not dsl_program.strip().startswith('atom_spec'):
            valid = False
            errors.append("DSL program must start with 'atom_spec'")
        
        if not dsl_program.strip().endswith('}'):
            warnings.append("DSL program may be incomplete (missing closing brace)")
        
        # Check for required header fields (per FullLanguage.tex line 206-208)
        if "dsl_version" not in dsl_program:
            valid = False
            errors.append("Missing required header field: dsl_version")
        if "title" not in dsl_program:
            valid = False
            errors.append("Missing required header field: title")
        if "created" not in dsl_program:
            valid = False
            errors.append("Missing required header field: created")
        
        # Phase 4: Compare against reference if available
        # This would be used when comparing generated DSL against expected output
        compare_reference = None  # Would be set if reference available
        
        return {
            "valid": valid,
            "warnings": warnings,
            "errors": errors,
            "confidence": 0.95 if valid and not errors else 0.5,
            "compare_reference": compare_reference
        }


# ============================================================================
# COMPILER AGENT
# ============================================================================

class CompilerAgent:
    """Agent for compiling validated DSL into output formats"""
    
    def __init__(self):
        self.name = "CompilerAgent"
        try:
            self.compiler = AtomForgeCompiler()
        except Exception as exc:
            logging.getLogger(self.name).warning(
                "AtomForgeCompiler initialization failed (%s). "
                "Compilation will return stub outputs.",
                exc,
            )
            self.compiler = None
    
    def process(self, input_message: AgentMessage) -> AgentMessage:
        """
        Compile validated DSL into output formats
        
        Args:
            input_message: AgentMessage containing validated DSL
            
        Returns:
            AgentMessage with compilation results
        """
        dsl_data = input_message.payload["result"].data
        dsl_program = dsl_data["dsl_program"]
        
        # Compile DSL
        try:
            if self.compiler:
                compiled_json = self.compiler.compile(dsl_program)
            outputs = {
                "dsl_program": dsl_program,
                    "compiled_output": compiled_json,
                    "formats": [self.compiler.output_format],
            }
            result = ProcessingResult(
                success=True,
                data=outputs,
                    confidence=0.9,
                )
            else:
                outputs = {
                    "dsl_program": dsl_program,
                    "compiled_output": None,
                    "formats": [],
                    "note": "Compiler unavailable; returning DSL only.",
                }
                result = ProcessingResult(
                    success=True,
                    data=outputs,
                    confidence=0.5,
                    warnings=["Compiler unavailable; returning DSL only."],
            )
            
        except Exception as e:
            result = ProcessingResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[str(e)]
            )
        
        return AgentMessage(
            message_id=f"msg_{datetime.now().isoformat()}",
            timestamp=datetime.now().isoformat(),
            sender=self.name,
            receiver="OutputGenerator",
            message_type="compilation_results",
            payload={"result": result}
        )


# ============================================================================
# MULTI-AGENT ORCHESTRATOR
# ============================================================================

class AtomForgeMultiAgentSystem:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self):
        llm_service = LLMService()
        self.reasoning_agent = MaterialsReasoningAgent(llm_service=llm_service)
        self.database_agent = DatabaseCoordinatorAgent()
        self.structure_agent = StructureProcessorAgent()
        self.variant_agent = VariantSelectorAgent()
        self.formatter_agent = DSLFormatterAgent()
        self.qa_agent = QAVerifierAgent()
        self.compiler_agent = CompilerAgent()
    
    def process(self, input_data: str, input_type: str = "natural_language") -> Dict[str, Any]:
        """
        Process materials science input through the multi-agent pipeline
        
        Args:
            input_data: Raw input (text, file path, database ID)
            input_type: Type of input ("natural_language", "file", "database_id")
            
        Returns:
            Dictionary containing all processing results
        """
        # Initialize message chain
        initial_message = AgentMessage(
            message_id="start",
            timestamp=datetime.now().isoformat(),
            sender="User",
            receiver="MaterialsReasoningAgent",
            message_type="raw_input",
            payload={"input": input_data, "type": input_type}
        )
        
        # Execute agent pipeline with stop-on-fail hardening
        messages: List[AgentMessage] = []
        current_message = initial_message
        agent_chain = [
            self.reasoning_agent,
            self.database_agent,
            self.structure_agent,
            self.variant_agent,
            self.formatter_agent,
            self.qa_agent,
            self.compiler_agent,
        ]

        final_result: Optional[ProcessingResult] = None

        for agent in agent_chain:
            current_message = agent.process(current_message)
        messages.append(current_message)
        
            result = current_message.payload.get("result")
            if isinstance(result, ProcessingResult):
                final_result = result
                if not result.success:
                    break
            else:
                # If a downstream agent returns an unexpected payload, halt pipeline.
                final_result = None
                break
        
        return {
            "success": final_result.success if final_result else False,
            "data": final_result.data if final_result else None,
            "confidence": final_result.confidence if final_result else 0.0,
            "messages": [self._message_to_dict(msg) for msg in messages],
            "processing_time": self._calculate_processing_time(messages)
        }
    
    def _message_to_dict(self, message: AgentMessage) -> Dict[str, Any]:
        """Convert AgentMessage to dictionary"""
        return {
            "message_id": message.message_id,
            "timestamp": message.timestamp,
            "sender": message.sender,
            "receiver": message.receiver,
            "message_type": message.message_type,
            "success": message.payload.get("result", ProcessingResult(False, None, 0.0)).success,
            "confidence": message.payload.get("result", ProcessingResult(False, None, 0.0)).confidence
        }
    
    def _calculate_processing_time(self, messages: List[AgentMessage]) -> float:
        """Calculate total processing time"""
        if len(messages) < 2:
            return 0.0
        
        start_time = datetime.fromisoformat(messages[0].timestamp)
        end_time = datetime.fromisoformat(messages[-1].timestamp)
        return (end_time - start_time).total_seconds()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Example usage of the multi-agent system"""
    system = AtomForgeMultiAgentSystem()
    
    # Example 1: Natural language input
    result = system.process("graphene monolayer", input_type="natural_language")
    print("Result:", json.dumps(result, indent=2, default=str))
    
    # Example 2: File input
    # result = system.process("path/to/structure.cif", input_type="file")
    
    # Example 3: Database ID
    # result = system.process("mp-149", input_type="database_id")


if __name__ == "__main__":
    main()
