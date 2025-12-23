"""
Utility functions for unconditional AtomForge program generation.

This module provides helper functions for:
- Loading seed programs from the dataset
- Text canonicalization and fingerprinting
- Program saving and metadata management
- Logging and statistics aggregation
"""

import os
import re
import hashlib
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_seed_programs(data_dir: str, limit: int = 20) -> List[Tuple[str, str]]:
    """
    Load seed programs from the data directory.
    
    Args:
        data_dir: Path to the data directory containing .atomforge files
        limit: Maximum number of seed programs to load
        
    Returns:
        List of tuples (file_path, program_text)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Collect all .atomforge files from batch directories
    atomforge_files = []
    for batch_dir in sorted(data_path.glob("batch_*")):
        if batch_dir.is_dir():
            atomforge_files.extend(batch_dir.glob("*.atomforge"))
    
    if not atomforge_files:
        raise ValueError(f"No .atomforge files found in {data_dir}")
    
    # Randomly sample up to limit files
    selected_files = random.sample(atomforge_files, min(limit, len(atomforge_files)))
    
    seed_programs = []
    for file_path in selected_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            seed_programs.append((str(file_path), content))
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    logger.info(f"Loaded {len(seed_programs)} seed programs from {data_dir}")
    return seed_programs


def canonicalize_text(text: str) -> str:
    """
    Normalize whitespace and remove comments to create a canonical representation.
    
    Args:
        text: Raw AtomForge program text
        
    Returns:
        Canonicalized text with normalized whitespace and no comments
    """
    # Remove single-line comments
    text = re.sub(r'#.*?$', '', text, flags=re.MULTILINE)
    
    # Remove multi-line comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Normalize whitespace: collapse multiple spaces/tabs to single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize newlines: replace all newline variants with single newline
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split('\n')]
    
    # Remove empty lines
    lines = [line for line in lines if line.strip()]
    
    # Join and normalize spacing around braces and operators
    text = '\n'.join(lines)
    text = re.sub(r'\s*{\s*', ' { ', text)
    text = re.sub(r'\s*}\s*', ' } ', text)
    text = re.sub(r'\s*=\s*', ' = ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def program_fingerprint(text_or_program: Any) -> str:
    """
    Generate a stable hash fingerprint for a program.
    
    Args:
        text_or_program: Either a string (program text) or parsed AtomForgeProgram object
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(text_or_program, str):
        canonical = canonicalize_text(text_or_program)
    else:
        # If it's a parsed program object, convert to canonical string representation
        # For now, we'll use a simple approach: hash the identifier + lattice + symmetry + basis
        try:
            parts = []
            if hasattr(text_or_program, 'identifier'):
                parts.append(str(text_or_program.identifier))
            if hasattr(text_or_program, 'lattice') and text_or_program.lattice:
                if hasattr(text_or_program.lattice, 'bravais') and text_or_program.lattice.bravais:
                    bravais = text_or_program.lattice.bravais
                    # Handle Length objects (access .value) or direct floats
                    def get_value(obj):
                        if hasattr(obj, 'value'):
                            return obj.value
                        return float(obj)
                    a_val = get_value(bravais.a) if hasattr(bravais, 'a') else 0
                    b_val = get_value(bravais.b) if hasattr(bravais, 'b') else 0
                    c_val = get_value(bravais.c) if hasattr(bravais, 'c') else 0
                    parts.append(f"{bravais.type}_{a_val:.6f}_{b_val:.6f}_{c_val:.6f}")
            if hasattr(text_or_program, 'symmetry') and text_or_program.symmetry:
                parts.append(f"sg{text_or_program.symmetry.space_group}")
            if hasattr(text_or_program, 'basis') and text_or_program.basis:
                site_count = len(text_or_program.basis.sites) if hasattr(text_or_program.basis, 'sites') else 0
                parts.append(f"sites{site_count}")
            canonical = '|'.join(parts)
        except Exception:
            # Fallback: use string representation
            canonical = str(text_or_program)
    
    # Generate SHA256 hash
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


def save_program(
    out_dir: str,
    program_id: str,
    program_text: str,
    metadata: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Save a generated program to disk with metadata.
    
    Args:
        out_dir: Output directory path
        program_id: Unique identifier for the program (e.g., "AF_UNCOND_000001")
        program_text: The AtomForge program text
        metadata: Dictionary of metadata to save
        
    Returns:
        Tuple of (program_file_path, metadata_file_path)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create programs subdirectory
    programs_dir = out_path / "programs"
    programs_dir.mkdir(exist_ok=True)
    
    # Save program file
    program_file = programs_dir / f"{program_id}.atomforge"
    with open(program_file, 'w', encoding='utf-8') as f:
        f.write(program_text)
    
    # Save metadata as JSON (optional, can be part of metrics.jsonl)
    metadata_file = programs_dir / f"{program_id}.metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return str(program_file), str(metadata_file)


def log_metric(
    metrics_file: str,
    metric_data: Dict[str, Any]
) -> None:
    """
    Append a metric entry to the metrics JSONL file.
    
    Args:
        metrics_file: Path to the metrics.jsonl file
        metric_data: Dictionary of metric data to log
    """
    metrics_path = Path(metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'a', encoding='utf-8') as f:
        json.dump(metric_data, f, default=str)
        f.write('\n')


def aggregate_stats(metrics_file: str) -> Dict[str, Any]:
    """
    Aggregate statistics from the metrics JSONL file.
    
    Args:
        metrics_file: Path to the metrics.jsonl file
        
    Returns:
        Dictionary with aggregated statistics
    """
    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        return {}
    
    stats = {
        'total_attempts': 0,
        'parse_errors': 0,
        'validation_errors': 0,
        'duplicates': 0,
        'successful': 0,
        'total_tokens_in': 0,
        'total_tokens_out': 0,
        'unique_fingerprints': set(),
        'status_counts': {},
    }
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                stats['total_attempts'] += 1
                
                status = entry.get('status', 'unknown')
                stats['status_counts'][status] = stats['status_counts'].get(status, 0) + 1
                
                if status == 'ok':
                    stats['successful'] += 1
                elif status == 'parse_error':
                    stats['parse_errors'] += 1
                elif status == 'validation_error':
                    stats['validation_errors'] += 1
                elif status == 'duplicate':
                    stats['duplicates'] += 1
                
                if 'tokens_in' in entry:
                    stats['total_tokens_in'] += entry['tokens_in']
                if 'tokens_out' in entry:
                    stats['total_tokens_out'] += entry['tokens_out']
                
                if 'fingerprint' in entry:
                    stats['unique_fingerprints'].add(entry['fingerprint'])
            except json.JSONDecodeError:
                continue
    
    # Convert set to count
    stats['unique_count'] = len(stats['unique_fingerprints'])
    stats['unique_fingerprints'] = None  # Don't include in final output
    
    # Calculate rates
    if stats['total_attempts'] > 0:
        stats['validity_rate'] = stats['successful'] / stats['total_attempts']
        stats['unique_rate'] = stats['unique_count'] / max(stats['successful'], 1)
    else:
        stats['validity_rate'] = 0.0
        stats['unique_rate'] = 0.0
    
    return stats


def save_summary(out_dir: str, summary: Dict[str, Any]) -> str:
    """
    Save aggregated summary statistics to a JSON file.
    
    Args:
        out_dir: Output directory path
        summary: Dictionary of summary statistics
        
    Returns:
        Path to the saved summary file
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = out_path / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return str(summary_file)

