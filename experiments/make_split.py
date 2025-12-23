#!/usr/bin/env python3
"""
Create Deterministic Train/Val/Test Splits for MP-20 Dataset

This script generates deterministic splits of the MP-20 minimal dataset for evaluation.
Splits are saved as JSON files mapping split names to file lists.

Usage:
    python -m experiments.make_split --data_dir data --out_dir data/splits --train_ratio 0.8 --val_ratio 0.1

Output:
    - data/splits/train.json (list of file paths)
    - data/splits/val.json
    - data/splits/test.json
    - data/splits/split_info.json (metadata about the split)
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_atomforge_files(data_dir: Path) -> List[Path]:
    """Collect all .atomforge files from batch directories."""
    atomforge_files: List[Path] = []
    
    for batch_dir in sorted(data_dir.glob("batch_*")):
        if batch_dir.is_dir():
            atomforge_files.extend(batch_dir.glob("*.atomforge"))
    
    return sorted(atomforge_files)


def create_splits(
    files: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create deterministic train/val/test splits.
    
    Args:
        files: List of file paths
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set (test gets remainder)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to lists of file paths (as strings)
    """
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle deterministically
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    
    # Compute split sizes
    n_total = len(files_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Split
    train_files = files_shuffled[:n_train]
    val_files = files_shuffled[n_train:n_train + n_val]
    test_files = files_shuffled[n_train + n_val:]
    
    return {
        'train': [str(f) for f in train_files],
        'val': [str(f) for f in val_files],
        'test': [str(f) for f in test_files],
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing batch_* subdirectories with .atomforge files'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default='data/splits',
        help='Output directory for split JSON files'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Fraction of data for training set'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='Fraction of data for validation set'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for deterministic splitting'
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    
    # Collect files
    logger.info(f"Collecting .atomforge files from {data_path}...")
    files = collect_atomforge_files(data_path)
    logger.info(f"Found {len(files)} files")
    
    if len(files) == 0:
        raise ValueError("No .atomforge files found")
    
    # Create splits
    logger.info(f"Creating splits (train={args.train_ratio:.1%}, val={args.val_ratio:.1%}, test={1-args.train_ratio-args.val_ratio:.1%})...")
    splits = create_splits(files, args.train_ratio, args.val_ratio, seed=args.seed)
    
    # Save individual split files
    for split_name, split_files in splits.items():
        split_file = out_path / f"{split_name}.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_files, f, indent=2)
        logger.info(f"  {split_name}: {len(split_files)} files -> {split_file}")
    
    # Save metadata
    info = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(data_path),
        'total_files': len(files),
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': 1.0 - args.train_ratio - args.val_ratio,
        'seed': args.seed,
        'split_sizes': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test']),
        }
    }
    
    info_file = out_path / "split_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"\nSplit metadata saved to: {info_file}")
    logger.info("="*60)
    logger.info("SPLIT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files: {info['total_files']}")
    logger.info(f"Train: {info['split_sizes']['train']} ({args.train_ratio:.1%})")
    logger.info(f"Val: {info['split_sizes']['val']} ({args.val_ratio:.1%})")
    logger.info(f"Test: {info['split_sizes']['test']} ({1-args.train_ratio-args.val_ratio:.1%})")
    logger.info("="*60)


if __name__ == '__main__':
    main()

