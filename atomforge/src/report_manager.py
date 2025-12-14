#!/usr/bin/env python3
"""
AtomForge Report Manager
========================

This module provides comprehensive report storage functionality for AtomForge workflows.
It handles storing all types of reports (preprocessing, workflow steps, analysis results)
as organized files with proper timestamps and metadata.

Features:
- Structured report storage with timestamps
- Multiple report formats (text, JSON, structured data)
- Automatic file organization and naming
- Integration with existing workflow demos
- Support for different report types (preprocessing, analysis, comparison, etc.)
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import logging

@dataclass
class ReportMetadata:
    """Metadata for a report"""
    report_type: str
    workflow_phase: str
    timestamp: str
    material_name: str
    description: str
    file_paths: List[str]
    additional_info: Dict[str, Any] = None

class ReportManager:
    """
    Manages storage of all workflow reports as files.
    
    This class provides a centralized way to store different types of reports
    generated during AtomForge workflows, including preprocessing reports,
    analysis results, comparison summaries, and workflow outputs.
    """
    
    def __init__(self, base_output_dir: Optional[Path] = None):
        """
        Initialize the ReportManager.
        
        Args:
            base_output_dir: Base directory for storing reports. If None, uses default.
        """
        if base_output_dir is None:
            # Default to docs/demo_output
            self.base_output_dir = Path(__file__).resolve().parents[2] / "docs" / "demo_output"
        else:
            self.base_output_dir = Path(base_output_dir)
        
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different report types (lightweight, cacheable reports only)
        self.report_dirs = {
            'canon_reports': self.base_output_dir / "canon_reports",
            'validation_reports': self.base_output_dir / "validation_reports",
            'selection_reports': self.base_output_dir / "selection_reports",
            'supercell_maps': self.base_output_dir / "supercell_maps",
            'prep_reports': self.base_output_dir / "prep_reports",
            'compare_reports': self.base_output_dir / "compare_reports",
            'patch_records': self.base_output_dir / "patch_records",
            'atomforge_programs': self.base_output_dir / "atomforge_programs",
            'metadata': self.base_output_dir / "metadata"
        }
        
        for dir_path in self.report_dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Current session info
        self.session_timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        self.session_reports = []
        
    def create_timestamped_filename(self, prefix: str, extension: str = "txt") -> str:
        """Create a timestamped filename"""
        return f"{prefix}_{self.session_timestamp}.{extension}"
    
    def store_canon_report(self, 
                          material_name: str,
                          canon_report: Any,
                          description: str = "Canonicalization report") -> str:
        """
        Store CanonReport: actions taken, epsilon used, spglib settings.
        
        Args:
            material_name: Name of the material being processed
            canon_report: CanonReport object or dictionary
            description: Description of the canonicalization report
            
        Returns:
            Path to the stored report file
        """
        filename = self.create_timestamped_filename(f"{material_name}_canon", "json")
        file_path = self.report_dirs['canon_reports'] / filename
        
        # Convert report object to dictionary if needed
        if hasattr(canon_report, '__dict__'):
            report_data = canon_report.__dict__
        else:
            report_data = canon_report
        
        report_data = {
            "metadata": {
                "report_type": "canon_report",
                "material_name": material_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp
            },
            "canon_data": report_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("canon_reports", "Phase 0", material_name, description, [str(file_path)])
        
        return str(file_path)
    
    def store_validation_report(self,
                               material_name: str,
                               validation_report: Any,
                               description: str = "Validation report") -> str:
        """
        Store ValidationReport: pass/fail and messages per rule.
        
        Args:
            material_name: Name of the material being processed
            validation_report: ValidationReport object or dictionary
            description: Description of the validation report
            
        Returns:
            Path to the stored report file
        """
        filename = self.create_timestamped_filename(f"{material_name}_validation", "json")
        file_path = self.report_dirs['validation_reports'] / filename
        
        # Convert report object to dictionary if needed
        if hasattr(validation_report, '__dict__'):
            report_data = validation_report.__dict__
        else:
            report_data = validation_report
        
        report_data = {
            "metadata": {
                "report_type": "validation_report",
                "material_name": material_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp
            },
            "validation_data": report_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("validation_reports", "Phase 0", material_name, description, [str(file_path)])
        
        return str(file_path)
    
    def store_selection_report(self,
                              material_name: str,
                              selection_report: Any,
                              description: str = "Selection report") -> str:
        """
        Store SelectionReport: ranking criteria and chosen variant.
        
        Args:
            material_name: Name of the material being processed
            selection_report: SelectionReport object or dictionary
            description: Description of the selection report
            
        Returns:
            Path to the stored report file
        """
        filename = self.create_timestamped_filename(f"{material_name}_selection", "json")
        file_path = self.report_dirs['selection_reports'] / filename
        
        # Convert report object to dictionary if needed
        if hasattr(selection_report, '__dict__'):
            report_data = selection_report.__dict__
        else:
            report_data = selection_report
        
        report_data = {
            "metadata": {
                "report_type": "selection_report",
                "material_name": material_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp
            },
            "selection_data": report_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("selection_reports", "Phase 1", material_name, description, [str(file_path)])
        
        return str(file_path)
    
    def store_supercell_map(self,
                           material_name: str,
                           supercell_map: Any,
                           description: str = "Supercell mapping") -> str:
        """
        Store SupercellMap: mapping child-parent.
        
        Args:
            material_name: Name of the material being processed
            supercell_map: SupercellMap object or dictionary
            description: Description of the supercell mapping
            
        Returns:
            Path to the stored report file
        """
        filename = self.create_timestamped_filename(f"{material_name}_supercell_map", "json")
        file_path = self.report_dirs['supercell_maps'] / filename
        
        # Convert report object to dictionary if needed
        if hasattr(supercell_map, '__dict__'):
            report_data = supercell_map.__dict__
        else:
            report_data = supercell_map
        
        report_data = {
            "metadata": {
                "report_type": "supercell_map",
                "material_name": material_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp
            },
            "supercell_map_data": report_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("supercell_maps", "Phase 2", material_name, description, [str(file_path)])
        
        return str(file_path)
    
    def store_prep_report(self,
                         material_name: str,
                         prep_report: Any,
                         description: str = "Preparation report") -> str:
        """
        Store PrepReport: filled defaults, derived k-mesh, warnings.
        
        Args:
            material_name: Name of the material being processed
            prep_report: PrepReport object or dictionary
            description: Description of the preparation report
            
        Returns:
            Path to the stored report file
        """
        filename = self.create_timestamped_filename(f"{material_name}_prep", "json")
        file_path = self.report_dirs['prep_reports'] / filename
        
        # Convert report object to dictionary if needed
        if hasattr(prep_report, '__dict__'):
            report_data = prep_report.__dict__
        else:
            report_data = prep_report
        
        report_data = {
            "metadata": {
                "report_type": "prep_report",
                "material_name": material_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp
            },
            "prep_data": report_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("prep_reports", "Phase 3", material_name, description, [str(file_path)])
        
        return str(file_path)
    
    def store_compare_report(self,
                            material_name: str,
                            compare_report: Any,
                            description: str = "Comparison report") -> str:
        """
        Store CompareReport: equivalence flag, diffs summary.
        
        Args:
            material_name: Name of the material being processed
            compare_report: CompareReport object or dictionary
            description: Description of the comparison report
            
        Returns:
            Path to the stored report file
        """
        filename = self.create_timestamped_filename(f"{material_name}_compare", "json")
        file_path = self.report_dirs['compare_reports'] / filename
        
        # Convert report object to dictionary if needed
        if hasattr(compare_report, '__dict__'):
            report_data = compare_report.__dict__
        else:
            report_data = compare_report
        
        report_data = {
            "metadata": {
                "report_type": "compare_report",
                "material_name": material_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp
            },
            "compare_data": report_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("compare_reports", "Comparison", material_name, description, [str(file_path)])
        
        return str(file_path)
    
    def store_patch_record(self,
                          material_name: str,
                          patch_record: Any,
                          description: str = "Patch record") -> str:
        """
        Store PatchRecord: op, params, preconditions, result_hash, timestamp.
        
        Args:
            material_name: Name of the material being processed
            patch_record: PatchRecord object or dictionary
            description: Description of the patch record
            
        Returns:
            Path to the stored report file
        """
        filename = self.create_timestamped_filename(f"{material_name}_patch", "json")
        file_path = self.report_dirs['patch_records'] / filename
        
        # Convert report object to dictionary if needed
        if hasattr(patch_record, '__dict__'):
            report_data = patch_record.__dict__
        else:
            report_data = patch_record
        
        report_data = {
            "metadata": {
                "report_type": "patch_record",
                "material_name": material_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp
            },
            "patch_data": report_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("patch_records", "Patch", material_name, description, [str(file_path)])
        
        return str(file_path)
    
    def store_atomforge_program(self,
                               material_name: str,
                               program_content: str,
                               operation_name: str = "operation",
                               description: str = "AtomForge program") -> str:
        """
        Store AtomForge program after each operation.
        
        Args:
            material_name: Name of the material being processed
            program_content: Complete AtomForge DSL program as string
            operation_name: Name of the operation that generated this program
            description: Description of the AtomForge program
            
        Returns:
            Path to the stored program file
        """
        filename = self.create_timestamped_filename(f"{material_name}_{operation_name}", "atomforge")
        file_path = self.report_dirs['atomforge_programs'] / filename
        
        # Store the AtomForge program as a text file
        with open(file_path, 'w') as f:
            f.write(program_content)
        
        # Also store metadata about the program
        metadata_filename = self.create_timestamped_filename(f"{material_name}_{operation_name}_metadata", "json")
        metadata_path = self.report_dirs['atomforge_programs'] / metadata_filename
        
        metadata = {
            "metadata": {
                "report_type": "atomforge_program",
                "material_name": material_name,
                "operation_name": operation_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "session_timestamp": self.session_timestamp,
                "program_file": str(file_path)
            },
            "program_info": {
                "file_size": len(program_content),
                "line_count": len(program_content.split('\n')),
                "contains_lattice": "lattice" in program_content.lower(),
                "contains_basis": "basis" in program_content.lower(),
                "contains_symmetry": "symmetry" in program_content.lower(),
                "contains_tile": "tile" in program_content.lower(),
                "contains_patch": "patch" in program_content.lower()
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Store metadata
        self._store_report_metadata("atomforge_programs", operation_name, material_name, description, [str(file_path), str(metadata_path)])
        
        return str(file_path)
    
    def _store_report_metadata(self,
                              report_type: str,
                              phase: str,
                              name: str,
                              description: str,
                              file_paths: List[str]):
        """Store metadata for a report"""
        metadata = ReportMetadata(
            report_type=report_type,
            workflow_phase=phase,
            timestamp=datetime.now().isoformat(),
            material_name=name,
            description=description,
            file_paths=file_paths
        )
        
        self.session_reports.append(metadata)
    
    def generate_session_summary(self) -> str:
        """
        Generate a summary of all reports in the current session.
        
        Returns:
            Path to the session summary file
        """
        filename = self.create_timestamped_filename("session_summary", "json")
        file_path = self.report_dirs['metadata'] / filename
        
        summary_data = {
            "session_metadata": {
                "session_timestamp": self.session_timestamp,
                "total_reports": len(self.session_reports),
                "report_types": list(set(report.report_type for report in self.session_reports)),
                "generated_at": datetime.now().isoformat()
            },
            "reports": [asdict(report) for report in self.session_reports]
        }
        
        with open(file_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        return str(file_path)
    
    def get_session_reports(self) -> List[ReportMetadata]:
        """Get all reports from the current session"""
        return self.session_reports.copy()
    
    def list_all_reports(self) -> Dict[str, List[str]]:
        """
        List all report files in the output directory.
        
        Returns:
            Dictionary mapping report types to lists of file paths
        """
        all_reports = {}
        for report_type, dir_path in self.report_dirs.items():
            if dir_path.exists():
                files = [str(f) for f in dir_path.iterdir() if f.is_file()]
                all_reports[report_type] = files
            else:
                all_reports[report_type] = []
        
        return all_reports

# Convenience functions for backward compatibility
def create_report_manager(base_output_dir: Optional[Path] = None) -> ReportManager:
    """Create a new ReportManager instance"""
    return ReportManager(base_output_dir)

def store_canon_report(material_name: str,
                      canon_report: Any,
                      base_output_dir: Optional[Path] = None) -> str:
    """
    Convenience function to store canonicalization report.
    
    Args:
        material_name: Name of the material
        canon_report: CanonReport object or dictionary
        base_output_dir: Base output directory (optional)
        
    Returns:
        Path to the stored report file
    """
    manager = ReportManager(base_output_dir)
    return manager.store_canon_report(material_name, canon_report)

def store_validation_report(material_name: str,
                           validation_report: Any,
                           base_output_dir: Optional[Path] = None) -> str:
    """
    Convenience function to store validation report.
    
    Args:
        material_name: Name of the material
        validation_report: ValidationReport object or dictionary
        base_output_dir: Base output directory (optional)
        
    Returns:
        Path to the stored report file
    """
    manager = ReportManager(base_output_dir)
    return manager.store_validation_report(material_name, validation_report)

def store_patch_record(material_name: str,
                      patch_record: Any,
                      base_output_dir: Optional[Path] = None) -> str:
    """
    Convenience function to store patch record.
    
    Args:
        material_name: Name of the material
        patch_record: PatchRecord object or dictionary
        base_output_dir: Base output directory (optional)
        
    Returns:
        Path to the stored report file
    """
    manager = ReportManager(base_output_dir)
    return manager.store_patch_record(material_name, patch_record)

def store_atomforge_program(material_name: str,
                           program_content: str,
                           operation_name: str = "operation",
                           base_output_dir: Optional[Path] = None) -> str:
    """
    Convenience function to store AtomForge program.
    
    Args:
        material_name: Name of the material
        program_content: Complete AtomForge DSL program as string
        operation_name: Name of the operation that generated this program
        base_output_dir: Base output directory (optional)
        
    Returns:
        Path to the stored program file
    """
    manager = ReportManager(base_output_dir)
    return manager.store_atomforge_program(material_name, program_content, operation_name)

if __name__ == "__main__":
    # Example usage
    manager = ReportManager()
    
    # Store a sample canonicalization report
    canon_data = {
        "actions_taken": ["reduced_to_primitive", "converted_to_conventional"],
        "epsilon_used": 1e-8,
        "canonical_hash": "4e14561d22b57294...",
        "spglib_settings": {"symprec": 1e-5}
    }
    
    canon_path = manager.store_canon_report(
        "LLZO", 
        canon_data, 
        "LLZO canonicalization report"
    )
    
    print(f"Canonicalization report stored at: {canon_path}")
    
    # Store a sample validation report
    validation_data = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "rules_checked": ["occupancy_sum", "minimum_distance", "lattice_validity"]
    }
    
    validation_path = manager.store_validation_report(
        "LLZO",
        validation_data,
        "LLZO validation report"
    )
    
    print(f"Validation report stored at: {validation_path}")
    
    # Store a sample patch record
    patch_data = {
        "op": "vacancy",
        "params": {"site_sel": "Species:Li", "occupancy": 0.964},
        "preconditions": {"original_sites": 192},
        "result_hash": "a4e019bb2a5ab24a...",
        "timestamp": datetime.now().isoformat()
    }
    
    patch_path = manager.store_patch_record(
        "LLZO",
        patch_data,
        "LLZO Li vacancy creation patch"
    )
    
    print(f"Patch record stored at: {patch_path}")
    
    # Store a sample AtomForge program using the generator
    from atomforge_generator import generate_atomforge_program
    
    # Create a mock crystal object for testing
    class MockCrystal:
        def __init__(self):
            self.composition = "Li7La3Zr2O12"
            self.lattice = MockLattice()
            self.space_group = "I4_1/acd"
            self.sites = [MockSite()]
    
    class MockLattice:
        def __init__(self):
            self.a = 13.236
            self.b = 13.236
            self.c = 12.702
            self.alpha = 90.0
            self.beta = 90.0
            self.gamma = 90.0
    
    class MockSite:
        def __init__(self):
            self.wyckoff = "16f"
            self.frac = (0.0, 0.0, 0.0)
            self.species = {"Li": 0.964}
    
    mock_crystal = MockCrystal()
    
    atomforge_program = generate_atomforge_program(
        mock_crystal,
        material_name="LLZO_demo",
        description="LLZO garnet solid electrolyte with Li vacancies",
        operations=[
            {"operation": "vacancy", "params": {"site_sel": "Species:Li", "occupancy": 0.964}}
        ],
        supercell_info={"repeat": (2, 2, 2), "origin_shift": (0.0, 0.0, 0.0)}
    )
    
    program_path = manager.store_atomforge_program(
        "LLZO",
        atomforge_program,
        "li_vacancy_supercell",
        "LLZO with Li vacancies in 2x2x2 supercell"
    )
    
    print(f"AtomForge program stored at: {program_path}")
    
    # Generate session summary
    summary_path = manager.generate_session_summary()
    print(f"Session summary stored at: {summary_path}")
    
    # List all reports
    all_reports = manager.list_all_reports()
    print(f"All reports: {all_reports}")
