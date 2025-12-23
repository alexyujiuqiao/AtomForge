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


def program_fingerprint(obj: Any) -> str:
    """
    Generate a stable, structure-based hash fingerprint for an AtomForge program.

    The fingerprint is derived from the parsed IR and is **header-agnostic**:
    - It ignores identifier, uuid, title, created, and description.
    - It includes only structural information:
      - Bravais lattice type and (a, b, c, alpha, beta, gamma) rounded to 1e-4.
      - Symmetry space_group.
      - Basis sites as a sorted list of:
        - position (rounded to 1e-4),
        - frame ("fractional" or "cartesian"),
        - species list of (element, occupancy) pairs sorted by element then occupancy.

    If a parsed program object is not available (e.g., parsing failed), this function
    falls back to a canonicalized text-based fingerprint.

    Args:
        obj: A parsed AtomForge program object (preferred) or a raw text string.

    Returns:
        Hexadecimal SHA-256 hash string (first 16 hex characters).
    """
    # Structure-based path: any object that looks like an AtomForgeProgram
    if not isinstance(obj, str) and hasattr(obj, "basis") and hasattr(obj, "lattice"):
        try:
            prog = obj
            signature: Dict[str, Any] = {}

            # Bravais lattice
            lattice = getattr(prog, "lattice", None)
            bravais = getattr(lattice, "bravais", None)

            def get_val(x: Any, default: float) -> float:
                if x is None:
                    return float(default)
                if hasattr(x, "value"):
                    return float(getattr(x, "value"))
                return float(x)

            if bravais is not None:
                signature["bravais"] = {
                    "type": str(getattr(bravais, "type", "")),
                    "a": round(get_val(getattr(bravais, "a", 0.0), 0.0), 4),
                    "b": round(get_val(getattr(bravais, "b", 0.0), 0.0), 4),
                    "c": round(get_val(getattr(bravais, "c", 0.0), 0.0), 4),
                    "alpha": round(get_val(getattr(bravais, "alpha", 90.0), 90.0), 4),
                    "beta": round(get_val(getattr(bravais, "beta", 90.0), 90.0), 4),
                    "gamma": round(get_val(getattr(bravais, "gamma", 90.0), 90.0), 4),
                }

            # Symmetry
            symmetry = getattr(prog, "symmetry", None)
            if symmetry is not None and hasattr(symmetry, "space_group"):
                signature["space_group"] = symmetry.space_group

            # Basis / sites
            basis = getattr(prog, "basis", None)
            sites_sig: List[Dict[str, Any]] = []
            if basis is not None:
                for site in getattr(basis, "sites", []):
                    pos = getattr(site, "position", (0.0, 0.0, 0.0))
                    frame = str(getattr(site, "frame", "fractional"))
                    try:
                        x, y, z = pos
                    except Exception:
                        x, y, z = (0.0, 0.0, 0.0)
                    coords = (
                        round(float(x), 4),
                        round(float(y), 4),
                        round(float(z), 4),
                    )
                    species_entries: List[Tuple[str, float]] = []
                    for sp in getattr(site, "species", []):
                        element = getattr(sp, "element", None)
                        occ = getattr(sp, "occupancy", None)
                        if element is None or occ is None:
                            continue
                        try:
                            occ_val = float(occ)
                        except (TypeError, ValueError):
                            continue
                        species_entries.append((str(element), round(occ_val, 4)))
                    # Sort species within a site deterministically
                    species_entries.sort(key=lambda t: (t[0], t[1]))
                    sites_sig.append(
                        {
                            "position": coords,
                            "frame": frame,
                            "species": species_entries,
                        }
                    )
            # Sort sites deterministically so ordering does not affect the fingerprint
            sites_sig.sort(key=lambda s: (tuple(s["species"]), s["frame"], s["position"]))
            signature["sites"] = sites_sig

            canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        except Exception:
            # Fallback: use normalized text representation
            canonical = canonicalize_text(str(obj))
    else:
        # Textual fallback (e.g., when parsing failed)
        canonical = canonicalize_text(str(obj))

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

