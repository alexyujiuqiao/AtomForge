#!/usr/bin/env python3
"""
Generative Metrics for AtomForge Evaluation

Implements comprehensive metrics for unconditional generation evaluation:
- Validity (structure and composition)
- Diversity (pairwise distances)
- Coverage (match to reference set)
- Novelty (distance from reference set)
- Distribution matching (Wasserstein distance)

Based on the structure from basic_eval.py and eval_util.py.
"""

import logging
import os
import errno
import signal
import functools
import itertools
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import pickle

import numpy as np

logger = logging.getLogger(__name__)

# Dependency-gated imports
try:
    from pymatgen.core import Structure, Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    import smact
    from smact.screening import pauling_test
    SMACT_AVAILABLE = True
except ImportError:
    SMACT_AVAILABLE = False

try:
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
    from matminer.featurizers.composition.composite import ElementProperty
    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False

try:
    from scipy.spatial.distance import pdist, cdist
    from scipy.stats import wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import StandardScaler and scaler constants
try:
    # Try to import from eval_util.py in root
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval_util import StandardScaler, CompScalerMeans, CompScalerStds, chemical_symbols
except ImportError:
    # Fallback: define here if not available
    StandardScaler = None
    CompScalerMeans = None
    CompScalerStds = None
    chemical_symbols = None

# Cutoffs (logger already defined above)
COV_CUTOFFS = {
    'mp20': {'struc': 0.4, 'comp': 10.0},
}

NOVELTY_CUTOFFS = {
    'mp20': {'struc': 0.1, 'comp': 2.0},
}

# Composition scaler (will be initialized if available)
CompScaler = None
if CompScalerMeans is not None and CompScalerStds is not None and StandardScaler is not None:
    CompScaler = StandardScaler(
        means=np.array(CompScalerMeans),
        stds=np.array(CompScalerStds),
        replace_nan_token=0.0
    )

# Fingerprint featurizers
CompFP = None
CrystalNNFP = None

if MATMINER_AVAILABLE:
    try:
        CompFP = ElementProperty.from_preset('magpie')
        CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
    except Exception as e:
        logger.warning(f"Could not initialize matminer featurizers: {e}")


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """Timeout decorator for fingerprint computation."""
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


@timeout(5)
def timeout_featurize(structure: Structure, site_idx: int) -> np.ndarray:
    """Featurize a single site with timeout."""
    if CrystalNNFP is None:
        raise RuntimeError("CrystalNNFingerprint not available")
    return CrystalNNFP.featurize(structure, site_idx)


def smact_validity(
    comp: Tuple[str, ...],
    count: Tuple[int, ...],
    use_pauling_test: bool = True,
    include_alloys: bool = True
) -> bool:
    """
    Check SMACT validity (charge balance feasibility).
    
    Ported from basic_eval.py smact_validity function.
    """
    if not SMACT_AVAILABLE:
        logger.warning("SMACT not available, skipping composition validity")
        return True  # Default to valid if SMACT unavailable
    
    if not chemical_symbols:
        logger.warning("chemical_symbols not available")
        return True
    
    elem_symbols = tuple([chemical_symbols[elem] if isinstance(elem, int) else elem for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    
    if len(set(elem_symbols)) == 1:
        return True
    
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def structure_validity(
    structure: Structure,
    min_dist_cutoff: float = 0.5,
    min_volume_cutoff: float = 0.1
) -> bool:
    """Check structure validity (min distance and volume)."""
    if not PYMATGEN_AVAILABLE:
        return False
    
    dist_mat = structure.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (min_dist_cutoff + 10.0))
    
    if dist_mat.min() < min_dist_cutoff or structure.volume < min_volume_cutoff:
        return False
    else:
        return True


class GenCrystal:
    """
    Wrapper class for pymatgen Structure with validity and fingerprint computation.
    
    Similar to Crystal class in basic_eval.py, but adapted for AtomForge pipeline.
    """
    
    def __init__(
        self,
        structure: Structure,
        min_dist_cutoff: float = 0.5,
        min_volume_cutoff: float = 0.1,
        include_alloys: bool = True
    ):
        """
        Initialize GenCrystal from pymatgen Structure.
        
        Args:
            structure: pymatgen Structure object
            min_dist_cutoff: minimum interatomic distance cutoff (default 0.5 Å)
            min_volume_cutoff: minimum volume cutoff (default 0.1 Å³)
            include_alloys: whether to allow alloys in SMACT validity (default True)
        """
        self.structure = structure
        self.constructed = structure is not None
        self.invalid_reason = None
        
        # Get composition
        self._get_composition()
        
        # Get validity
        self._get_validity(min_dist_cutoff, min_volume_cutoff, include_alloys)
        
        # Get fingerprints if valid
        if self.valid:
            self._get_fingerprints()
        else:
            self.comp_fp = None
            self.struct_fp = None
    
    def _get_composition(self):
        """Extract composition elements and counts."""
        if not self.constructed:
            self.elems = ()
            self.comps = ()
            return
        
        elem_counter = Counter(self.structure.species)
        # Convert to element symbols and counts
        composition = [(str(elem), count) for elem, count in elem_counter.items()]
        composition.sort(key=lambda x: x[0])  # Sort by element symbol
        
        elems, counts = list(zip(*composition)) if composition else ((), ())
        counts = np.array(counts)
        # Reduce to simplest integer formula
        counts = counts / np.gcd.reduce(counts.astype(int))
        self.elems = elems
        self.comps = tuple(counts.astype(int).tolist())
    
    def _get_validity(
        self,
        min_dist_cutoff: float,
        min_volume_cutoff: float,
        include_alloys: bool
    ):
        """Compute validity flags."""
        if not self.constructed:
            self.struct_valid = False
            self.comp_valid = False
            self.valid = False
            self.invalid_reason = "not_constructed"
            return
        
        # Structure validity
        self.struct_valid = structure_validity(
            self.structure,
            min_dist_cutoff=min_dist_cutoff,
            min_volume_cutoff=min_volume_cutoff
        )
        
        # Composition validity
        if self.elems and self.comps:
            self.comp_valid = smact_validity(
                self.elems,
                self.comps,
                include_alloys=include_alloys
            )
        else:
            self.comp_valid = False
        
        self.valid = self.constructed and self.struct_valid and self.comp_valid
        
        if not self.valid:
            if not self.struct_valid:
                self.invalid_reason = "struct_invalid"
            elif not self.comp_valid:
                self.invalid_reason = "comp_invalid"
            else:
                self.invalid_reason = "unknown"
    
    def _get_fingerprints(self):
        """Compute composition and structure fingerprints."""
        if not MATMINER_AVAILABLE:
            logger.warning("matminer not available, skipping fingerprints")
            self.comp_fp = None
            self.struct_fp = None
            self.valid = False
            self.invalid_reason = "fingerprint_failed"
            return
        
        try:
            # Composition fingerprint
            elem_counter = Counter(self.structure.species)
            comp = Composition(elem_counter)
            if CompFP is None:
                raise RuntimeError("CompFP not initialized")
            self.comp_fp = CompFP.featurize(comp)
            
            # Structure fingerprint (average over sites)
            if CrystalNNFP is None:
                raise RuntimeError("CrystalNNFP not initialized")
            
            site_fps = []
            for i in range(len(self.structure)):
                try:
                    site_fp = timeout_featurize(self.structure, i)
                    site_fps.append(site_fp)
                except Exception as e:
                    logger.warning(f"Failed to featurize site {i}: {e}")
                    # If one site fails, mark as invalid
                    self.valid = False
                    self.comp_fp = None
                    self.struct_fp = None
                    self.invalid_reason = "fingerprint_failed"
                    return
            
            if site_fps:
                self.struct_fp = np.array(site_fps).mean(axis=0)
            else:
                self.valid = False
                self.comp_fp = None
                self.struct_fp = None
                self.invalid_reason = "fingerprint_failed"
                
        except Exception as e:
            logger.warning(f"Fingerprint computation failed: {e}")
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            self.invalid_reason = "fingerprint_failed"


def compute_gen_metrics(
    pred: List[GenCrystal],
    gt_cov: List[GenCrystal],
    gt_nov: List[GenCrystal],
    eval_model_name: str = "mp20",
    n_samples: int = 1000
) -> Dict[str, Any]:
    """
    Compute generative metrics.
    
    Args:
        pred: List of GenCrystal objects for generated structures
        gt_cov: List of GenCrystal objects for coverage GT
        gt_nov: List of GenCrystal objects for novelty GT
        eval_model_name: Model name for cutoffs (default "mp20")
        n_samples: Number of valid samples for diversity/Wasserstein (default 1000)
    
    Returns:
        Dictionary with metrics
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for metrics computation. Install with: pip install scipy")
    
    metrics = {}
    
    # Validity over all predicted
    valid_pred = [c for c in pred if c.valid]
    total_pred = len(pred)
    
    struct_valid_count = sum(1 for c in pred if c.struct_valid)
    comp_valid_count = sum(1 for c in pred if c.comp_valid)
    valid_count = len(valid_pred)
    
    metrics["validity"] = {
        "struct_valid_rate": struct_valid_count / max(1, total_pred),
        "comp_valid_rate": comp_valid_count / max(1, total_pred),
        "valid_rate": valid_count / max(1, total_pred),
    }
    
    if not valid_pred:
        # If no valid predictions, return early with zeros
        metrics["diversity"] = {"comp_div": 0.0, "struct_div": 0.0}
        metrics["distribution"] = {
            "wdist_density": float('inf'),
            "wdist_num_elems": float('inf'),
        }
        metrics["coverage"] = {
            "cov_recall": 0.0,
            "cov_precision": 0.0,
            "amsd_recall": float('inf'),
            "amsd_precision": float('inf'),
            "amcd_recall": float('inf'),
            "amcd_precision": float('inf'),
        }
        metrics["novelty"] = {
            "struc_novelty_rate": 0.0,
            "comp_novelty_rate": 0.0,
            "novelty_rate": 0.0,
        }
        return metrics
    
    # Sample valid predictions for diversity/Wasserstein
    if len(valid_pred) > n_samples:
        import random
        valid_sample = random.sample(valid_pred, n_samples)
    else:
        valid_sample = valid_pred
    
    # Diversity (mean pairwise distance)
    comp_fps = [c.comp_fp for c in valid_sample if c.comp_fp is not None]
    struct_fps = [c.struct_fp for c in valid_sample if c.struct_fp is not None]
    
    if comp_fps and len(comp_fps) > 1:
        # Scale composition fingerprints
        if CompScaler is not None:
            comp_fps_scaled = CompScaler.transform(comp_fps)
        else:
            comp_fps_scaled = np.array(comp_fps)
        comp_div = float(pdist(comp_fps_scaled).mean())
    else:
        comp_div = 0.0
    
    if struct_fps and len(struct_fps) > 1:
        struct_div = float(pdist(struct_fps).mean())
    else:
        struct_div = 0.0
    
    metrics["diversity"] = {
        "comp_div": comp_div,
        "struct_div": struct_div,
    }
    
    # Distribution matching (Wasserstein)
    valid_densities = [float(c.structure.density) if hasattr(c.structure.density, '__float__') else c.structure.density for c in valid_sample]
    valid_nels = [len(c.structure.composition.elements) for c in valid_sample]
    
    gt_cov_densities = [float(c.structure.density) if hasattr(c.structure.density, '__float__') else c.structure.density for c in gt_cov if c.valid]
    gt_cov_nels = [len(c.structure.composition.elements) for c in gt_cov if c.valid]
    
    if valid_densities and gt_cov_densities:
        wdist_density = float(wasserstein_distance(valid_densities, gt_cov_densities))
    else:
        wdist_density = float('inf')
    
    if valid_nels and gt_cov_nels:
        wdist_num_elems = float(wasserstein_distance(valid_nels, gt_cov_nels))
    else:
        wdist_num_elems = float('inf')
    
    metrics["distribution"] = {
        "wdist_density": wdist_density,
        "wdist_num_elems": wdist_num_elems,
    }
    
    # Coverage
    valid_gt_cov = [c for c in gt_cov if c.valid]
    if not valid_gt_cov or not valid_pred:
        metrics["coverage"] = {
            "cov_recall": 0.0,
            "cov_precision": 0.0,
            "amsd_recall": float('inf'),
            "amsd_precision": float('inf'),
            "amcd_recall": float('inf'),
            "amcd_precision": float('inf'),
        }
    else:
        pred_struct_fps = [c.struct_fp for c in valid_pred if c.struct_fp is not None]
        pred_comp_fps = [c.comp_fp for c in valid_pred if c.comp_fp is not None]
        gt_struct_fps = [c.struct_fp for c in valid_gt_cov if c.struct_fp is not None]
        gt_comp_fps = [c.comp_fp for c in valid_gt_cov if c.comp_fp is not None]
        
        if CompScaler is not None and pred_comp_fps and gt_comp_fps:
            pred_comp_fps_scaled = CompScaler.transform(pred_comp_fps)
            gt_comp_fps_scaled = CompScaler.transform(gt_comp_fps)
        else:
            pred_comp_fps_scaled = np.array(pred_comp_fps) if pred_comp_fps else np.array([])
            gt_comp_fps_scaled = np.array(gt_comp_fps) if gt_comp_fps else np.array([])
        
        struc_cutoff = COV_CUTOFFS.get(eval_model_name, {}).get('struc', 0.4)
        comp_cutoff = COV_CUTOFFS.get(eval_model_name, {}).get('comp', 10.0)
        
        # Compute distance matrices
        if pred_struct_fps and gt_struct_fps:
            struc_dist_mat = cdist(pred_struct_fps, gt_struct_fps)
            struc_mins_pred_to_gt = struc_dist_mat.min(axis=1)
            struc_mins_gt_to_pred = struc_dist_mat.min(axis=0)
            
            cov_precision = float((struc_mins_pred_to_gt < struc_cutoff).sum() / max(1, len(pred_struct_fps)))
            cov_recall = float((struc_mins_gt_to_pred < struc_cutoff).sum() / max(1, len(gt_struct_fps)))
            
            amsd_precision = float(struc_mins_pred_to_gt.mean()) if len(struc_mins_pred_to_gt) > 0 else float('inf')
            amsd_recall = float(struc_mins_gt_to_pred.mean()) if len(struc_mins_gt_to_pred) > 0 else float('inf')
        else:
            cov_precision = 0.0
            cov_recall = 0.0
            amsd_precision = float('inf')
            amsd_recall = float('inf')
        
        if len(pred_comp_fps_scaled) > 0 and len(gt_comp_fps_scaled) > 0:
            comp_dist_mat = cdist(pred_comp_fps_scaled, gt_comp_fps_scaled)
            comp_mins_pred_to_gt = comp_dist_mat.min(axis=1)
            comp_mins_gt_to_pred = comp_dist_mat.min(axis=0)
            
            # Coverage requires BOTH structure and composition within cutoff
            both_covered_pred = (struc_mins_pred_to_gt < struc_cutoff) & (comp_mins_pred_to_gt < comp_cutoff)
            both_covered_gt = (struc_mins_gt_to_pred < struc_cutoff) & (comp_mins_gt_to_pred < comp_cutoff)
            
            cov_precision = float(both_covered_pred.sum() / max(1, len(pred_comp_fps)))
            cov_recall = float(both_covered_gt.sum() / max(1, len(gt_comp_fps)))
            
            amcd_precision = float(comp_mins_pred_to_gt.mean()) if len(comp_mins_pred_to_gt) > 0 else float('inf')
            amcd_recall = float(comp_mins_gt_to_pred.mean()) if len(comp_mins_gt_to_pred) > 0 else float('inf')
        else:
            amcd_precision = float('inf')
            amcd_recall = float('inf')
        
        metrics["coverage"] = {
            "cov_recall": cov_recall,
            "cov_precision": cov_precision,
            "amsd_recall": amsd_recall,
            "amsd_precision": amsd_precision,
            "amcd_recall": amcd_recall,
            "amcd_precision": amcd_precision,
        }
    
    # Novelty
    valid_gt_nov = [c for c in gt_nov if c.valid]
    if not valid_gt_nov or not valid_pred:
        metrics["novelty"] = {
            "struc_novelty_rate": 0.0,
            "comp_novelty_rate": 0.0,
            "novelty_rate": 0.0,
        }
    else:
        pred_struct_fps = [c.struct_fp for c in valid_pred if c.struct_fp is not None]
        pred_comp_fps = [c.comp_fp for c in valid_pred if c.comp_fp is not None]
        gt_struct_fps = [c.struct_fp for c in valid_gt_nov if c.struct_fp is not None]
        gt_comp_fps = [c.comp_fp for c in valid_gt_nov if c.comp_fp is not None]
        
        if CompScaler is not None and pred_comp_fps and gt_comp_fps:
            pred_comp_fps_scaled = CompScaler.transform(pred_comp_fps)
            gt_comp_fps_scaled = CompScaler.transform(gt_comp_fps)
        else:
            pred_comp_fps_scaled = np.array(pred_comp_fps) if pred_comp_fps else np.array([])
            gt_comp_fps_scaled = np.array(gt_comp_fps) if gt_comp_fps else np.array([])
        
        struc_cutoff = NOVELTY_CUTOFFS.get(eval_model_name, {}).get('struc', 0.1)
        comp_cutoff = NOVELTY_CUTOFFS.get(eval_model_name, {}).get('comp', 2.0)
        
        struc_mins = None
        comp_mins = None
        
        if pred_struct_fps and gt_struct_fps:
            struc_dist_mat = cdist(pred_struct_fps, gt_struct_fps)
            struc_mins = struc_dist_mat.min(axis=1)
            struc_novelty_rate = float((struc_mins > struc_cutoff).sum() / max(1, len(pred_struct_fps)))
        else:
            struc_novelty_rate = 0.0
        
        if len(pred_comp_fps_scaled) > 0 and len(gt_comp_fps_scaled) > 0:
            comp_dist_mat = cdist(pred_comp_fps_scaled, gt_comp_fps_scaled)
            comp_mins = comp_dist_mat.min(axis=1)
            comp_novelty_rate = float((comp_mins > comp_cutoff).sum() / max(1, len(pred_comp_fps)))
        else:
            comp_novelty_rate = 0.0
        
        # Novelty rate is OR of structure and composition novelty
        if struc_mins is not None and comp_mins is not None:
            novelty_rate = float(((struc_mins > struc_cutoff) | (comp_mins > comp_cutoff)).sum() / max(1, len(pred_struct_fps)))
        elif struc_mins is not None:
            novelty_rate = struc_novelty_rate
        elif comp_mins is not None:
            novelty_rate = comp_novelty_rate
        else:
            novelty_rate = 0.0
        
        metrics["novelty"] = {
            "struc_novelty_rate": struc_novelty_rate,
            "comp_novelty_rate": comp_novelty_rate,
            "novelty_rate": novelty_rate,
        }
    
    return metrics



def load_gt_crystals(
    ref_dir: Path,
    max_ref: Optional[int],
    min_dist_cutoff: float = 0.5,
    min_volume_cutoff: float = 0.1,
    cache_path: Optional[Path] = None
) -> List[GenCrystal]:
    """
    Load GT crystals from reference directory with optional caching.
    
    Args:
        ref_dir: Directory containing reference .atomforge files
        max_ref: Maximum number of reference files to load
        min_dist_cutoff: Minimum interatomic distance cutoff
        min_volume_cutoff: Minimum volume cutoff
        cache_path: Optional path to cache file (if provided, will try to load/save cache)
    
    Returns:
        List of GenCrystal objects
    """
    # Try to load from cache
    if cache_path and cache_path.exists():
        try:
            logger.info(f"Loading GT crystals from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # Verify cache is still valid (simple check)
                if isinstance(cached_data, list) and len(cached_data) > 0:
                    return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will recompute")
    
    # Load from files
    logger.info(f"Loading GT crystals from {ref_dir}")
    
    # Import parser and conversion function
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from experiments.eval.convert import parse_atomforge_file, atomforge_to_structure
    except ImportError:
        logger.error("Could not import atomforge_to_structure from experiments.eval.convert")
        return []
    
    files = sorted(ref_dir.glob("batch_*/*.atomforge"))
    if not files:
        logger.warning(f"No reference .atomforge files found under {ref_dir}")
        return []
    
    if max_ref:
        files = files[:max_ref]
    
    crystals = []
    for f in files:
        try:
            ok, program = parse_atomforge_file(f)
            if not ok:
                continue
            
            struct, _ = atomforge_to_structure(program, expand_symmetry=True, symprec=0.2, auto_detect_expanded=True)
            
            crystal = GenCrystal(
                struct,
                min_dist_cutoff=min_dist_cutoff,
                min_volume_cutoff=min_volume_cutoff
            )
            crystals.append(crystal)
        except Exception as e:
            logger.debug(f"Failed to load {f}: {e}")
            continue
    
    logger.info(f"Loaded {len(crystals)} GT crystals")
    
    # Save to cache
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(crystals, f)
            logger.info(f"Cached GT crystals to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    return crystals


def aggregate_results(
    results_csv: Path,
    model_name: str,
    gen_metrics: Dict[str, Any]
) -> None:
    """
    Append results to aggregation CSV.
    
    Args:
        results_csv: Path to results CSV file
        model_name: Model name (method column)
        gen_metrics: Dictionary of generative metrics
    """
    import csv
    
    # Flatten metrics dictionary for CSV
    row_data = {"method": model_name}
    
    def flatten_dict(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            else:
                row_data[key] = v
    
    flatten_dict(gen_metrics)
    
    # Check if file exists and if method already exists
    fieldnames = ["method"] + [k for k in row_data.keys() if k != "method"]
    
    if results_csv.exists():
        # Read existing rows
        with open(results_csv, 'r') as f:
            reader = csv.DictReader(f)
            existing_methods = {row["method"] for row in reader if "method" in row}
            fieldnames = list(reader.fieldnames) if reader.fieldnames else fieldnames
        
        # Check if method already exists
        if model_name in existing_methods:
            logger.info(f"Method {model_name} already exists in {results_csv}, skipping aggregation")
            return
        
        # Append row
        with open(results_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_data)
    else:
        # Create new file
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_data)
    
    logger.info(f"Appended results to {results_csv}")
