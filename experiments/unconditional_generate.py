#!/usr/bin/env python3
"""
Unconditional AtomForge Program Generation Pipeline

This script implements an end-to-end pipeline for generating unconditional AtomForge DSL
programs using a SOTA LLM (GPT-5.2). The pipeline:

1. Samples seed programs from the MP-20 minimal dataset
2. Calls the LLM to generate new unconditional AtomForge programs
3. Validates each generated program by parsing and calling validate()
4. Filters duplicates using canonical hash fingerprints
5. Saves accepted programs and logs detailed metrics
6. Optionally performs auto-repair on failed programs

Usage:
    python -m experiments.unconditional_generate --n_samples 500 --seed_k 20

Output:
    - outputs/uncond/programs/AF_UNCOND_000001.atomforge (generated programs)
    - outputs/uncond/metrics.jsonl (detailed metrics per attempt)
    - outputs/uncond/summary.json (aggregated statistics)
"""

import os
import sys
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Also try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import AtomForge parser and IR
try:
    from atomforge.src.atomforge_parser import AtomForgeParser
    from atomforge.src.atomforge_ir import AtomForgeProgram
except ImportError:
    # Try alternative import path
    sys.path.insert(0, str(project_root / "atomforge" / "src"))
    from atomforge_parser import AtomForgeParser
    from atomforge_ir import AtomForgeProgram

# Import local modules
from experiments.prompts import get_generation_prompt, get_repair_prompt, extract_program_from_response
from experiments.utils import (
    load_seed_programs,
    canonicalize_text,
    program_fingerprint,
    save_program,
    log_metric,
    aggregate_stats,
    save_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Abstract interface for LLM API clients.
    Supports OpenAI API and can be extended for other providers.
    """
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name (e.g., "gpt-5.1-thinking")
            api_key: API key (if None, reads from OPENAI_API_KEY env var)
            **kwargs: Additional model parameters (temperature, top_p, etc.)
        """
        self.model = model
        # Try to get API key from: 1) provided argument, 2) environment variable, 3) .env file (already loaded)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided. Please either:\n"
                "  1. Set OPENAI_API_KEY environment variable, or\n"
                "  2. Create a .env file in the project root with: OPENAI_API_KEY=sk-your-key-here\n"
                "  3. Install python-dotenv: pip install python-dotenv"
            )
        
        self.temperature = kwargs.get('temperature', 0.8)
        self.top_p = kwargs.get('top_p', 0.95)
        self.max_tokens = kwargs.get('max_tokens', 4000)
        
        # Initialize OpenAI client
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI Python SDK not installed. Install with: pip install openai")
    
    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from the LLM with exponential backoff retry.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            
        Returns:
            Tuple of (response_text, metadata_dict) where metadata includes tokens, etc.
        """
        for attempt in range(max_retries):
            try:
                # Newer OpenAI models (e.g. gpt-4.1, gpt-5.x) expect `max_completion_tokens`
                # instead of the legacy `max_tokens` parameter.
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in crystallography and materials science.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_completion_tokens=self.max_tokens,
                )
                
                response_text = response.choices[0].message.content
                metadata = {
                    'tokens_in': response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    'tokens_out': response.usage.completion_tokens if hasattr(response, 'usage') else None,
                    'total_tokens': response.usage.total_tokens if hasattr(response, 'usage') else None,
                }
                
                return response_text, metadata
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")
        
        raise RuntimeError("Should not reach here")


class ProgramValidator:
    """Validates AtomForge programs using the parser and IR validation."""
    
    def __init__(self):
        """Initialize the validator with an AtomForge parser."""
        self.parser = AtomForgeParser()
        self.seen_fingerprints = set()
    
    def validate(
        self,
        program_text: str,
        check_duplicates: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate an AtomForge program.
        
        Args:
            program_text: The AtomForge program text to validate
            check_duplicates: Whether to check for duplicate fingerprints
            
        Returns:
            Tuple of (is_valid, error_message, fingerprint)
            - is_valid: True if program is valid and unique
            - error_message: None if valid, error description if invalid
            - fingerprint: Program fingerprint hash
        """
        # Extract clean program text
        program_text = extract_program_from_response(program_text)
        
        # Check for ai_integration block (should not be present)
        if 'ai_integration' in program_text.lower():
            return False, "Program contains ai_integration block (not allowed)", None
        
        # Generate fingerprint
        fingerprint = program_fingerprint(program_text)
        
        # Check duplicates
        if check_duplicates and fingerprint in self.seen_fingerprints:
            return False, "Duplicate program (fingerprint match)", fingerprint
        
        # Try parsing
        try:
            program = self.parser.parse_and_transform(program_text)
        except Exception as e:
            return False, f"Parse error: {str(e)}", fingerprint
        
        # Try validation
        try:
            program.validate()
        except Exception as e:
            return False, f"Validation error: {str(e)}", fingerprint
        
        # Additional checks
        # Check that required blocks are present
        if not program.header:
            return False, "Missing required header block", fingerprint
        if not program.lattice:
            return False, "Missing required lattice block", fingerprint
        if not program.symmetry:
            return False, "Missing required symmetry block", fingerprint
        if not program.basis or not program.basis.sites:
            return False, "Missing required basis block with sites", fingerprint
        
        # Check atom count (roughly 2-20 sites)
        site_count = len(program.basis.sites)
        if site_count < 2 or site_count > 20:
            return False, f"Site count {site_count} outside reasonable range [2, 20]", fingerprint
        
        # Check lattice parameters
        if program.lattice.bravais:
            bravais = program.lattice.bravais
            # Handle Length and Angle objects (access .value) or direct floats
            def get_value(obj):
                if hasattr(obj, 'value'):
                    return obj.value
                return float(obj)
            
            a_val = get_value(bravais.a)
            b_val = get_value(bravais.b)
            c_val = get_value(bravais.c)
            alpha_val = get_value(bravais.alpha)
            beta_val = get_value(bravais.beta)
            gamma_val = get_value(bravais.gamma)
            
            if a_val <= 0 or b_val <= 0 or c_val <= 0:
                return False, "Lattice parameters must be positive", fingerprint
            # Check angles are reasonable (60-120 degrees typically)
            if not (60 <= alpha_val <= 120) or not (60 <= beta_val <= 120) or not (60 <= gamma_val <= 120):
                # Allow some flexibility, but warn if very extreme
                if alpha_val < 30 or alpha_val > 150:
                    return False, f"Lattice angle alpha={alpha_val} outside reasonable range", fingerprint
        
        # Mark as seen
        if check_duplicates:
            self.seen_fingerprints.add(fingerprint)
        
        return True, None, fingerprint


def generate_unconditional_programs(
    data_dir: str,
    out_dir: str,
    n_samples: int,
    seed_k: int,
    max_retries: int,
    model: str,
    temperature: float,
    top_p: float,
    dry_run: bool = False
) -> None:
    """
    Main pipeline for generating unconditional AtomForge programs.
    
    Args:
        data_dir: Directory containing seed programs
        out_dir: Output directory for generated programs
        n_samples: Number of programs to generate
        seed_k: Number of seed programs to include in few-shot context
        max_retries: Maximum repair attempts per program
        model: LLM model name
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        dry_run: If True, don't call API (just validate existing files)
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load seed programs
    logger.info(f"Loading {seed_k} seed programs from {data_dir}...")
    seed_programs = load_seed_programs(data_dir, limit=seed_k)
    
    # Initialize components
    if not dry_run:
        llm_client = LLMClient(
            model=model,
            temperature=temperature,
            top_p=top_p
        )
    else:
        llm_client = None
        logger.info("DRY RUN MODE: Will not call LLM API")
    
    validator = ProgramValidator()
    metrics_file = str(out_path / "metrics.jsonl")
    
    # Load existing fingerprints if metrics file exists (for resuming)
    if Path(metrics_file).exists():
        logger.info("Loading existing fingerprints from metrics file...")
        with open(metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if 'fingerprint' in entry and entry.get('status') == 'ok':
                            validator.seen_fingerprints.add(entry['fingerprint'])
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(validator.seen_fingerprints)} existing fingerprints")
    
    # Generation loop
    successful_count = 0
    attempt_count = 0
    
    logger.info(f"Starting generation of {n_samples} programs...")
    
    while successful_count < n_samples:
        attempt_count += 1
        program_id = f"AF_UNCOND_{attempt_count:06d}"
        
        logger.info(f"\n[{attempt_count}] Generating program {program_id}...")
        
        # Generate program
        if dry_run:
            # In dry run, skip generation
            logger.info("Skipping generation in dry run mode")
            break
        
        try:
            prompt = get_generation_prompt(seed_programs)
            response_text, llm_metadata = llm_client.generate(prompt)
            
            # Extract program from response
            program_text = extract_program_from_response(response_text)
            
            # Validate
            is_valid, error_msg, fingerprint = validator.validate(program_text, check_duplicates=True)
            
            # Repair loop if needed
            repair_attempts = 0
            while not is_valid and repair_attempts < max_retries:
                if error_msg and ("parse error" in error_msg.lower() or "validation error" in error_msg.lower()):
                    logger.info(f"  Attempting repair (attempt {repair_attempts + 1}/{max_retries})...")
                    repair_prompt = get_repair_prompt(program_text, error_msg)
                    repair_response, _ = llm_client.generate(repair_prompt)
                    program_text = extract_program_from_response(repair_response)
                    is_valid, error_msg, fingerprint = validator.validate(program_text, check_duplicates=True)
                    repair_attempts += 1
                else:
                    # Not a repairable error (duplicate, etc.)
                    break
            
            # Determine status
            if is_valid:
                status = "ok"
                successful_count += 1
                
                # Save program
                metadata = {
                    'program_id': program_id,
                    'generated_at': datetime.now().isoformat(),
                    'model': model,
                    'temperature': temperature,
                    'top_p': top_p,
                    'fingerprint': fingerprint,
                    'seed_source': seed_programs[0][0] if seed_programs else None,
                }
                save_program(out_dir, program_id, program_text, metadata)
                logger.info(f"  ✓ Success! Saved as {program_id}.atomforge")
            else:
                if "duplicate" in error_msg.lower():
                    status = "duplicate"
                elif "parse error" in error_msg.lower():
                    status = "parse_error"
                elif "validation error" in error_msg.lower():
                    status = "validation_error"
                else:
                    status = "error"
                logger.info(f"  ✗ Failed: {error_msg}")
            
            # Log metric
            metric_entry = {
                'id': program_id,
                'attempt': attempt_count,
                'status': status,
                'error_message': error_msg,
                'fingerprint': fingerprint,
                'seed_source': seed_programs[0][0] if seed_programs else None,
                'model': model,
                'temperature': temperature,
                'top_p': top_p,
                'tokens_in': llm_metadata.get('tokens_in'),
                'tokens_out': llm_metadata.get('tokens_out'),
                'repair_attempts': repair_attempts,
            }
            log_metric(metrics_file, metric_entry)
            
        except Exception as e:
            logger.error(f"  ✗ Exception during generation: {e}")
            metric_entry = {
                'id': program_id,
                'attempt': attempt_count,
                'status': 'exception',
                'error_message': str(e),
                'fingerprint': None,
            }
            log_metric(metrics_file, metric_entry)
        
        # Rate limiting: small delay between requests
        if not dry_run:
            time.sleep(0.5)
    
    # Generate summary
    logger.info("\nGenerating summary statistics...")
    stats = aggregate_stats(metrics_file)
    stats['config'] = {
        'data_dir': data_dir,
        'out_dir': out_dir,
        'n_samples': n_samples,
        'seed_k': seed_k,
        'max_retries': max_retries,
        'model': model,
        'temperature': temperature,
        'top_p': top_p,
        'dry_run': dry_run,
    }
    stats['timestamp'] = datetime.now().isoformat()
    
    summary_file = save_summary(out_dir, stats)
    logger.info(f"Summary saved to {summary_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("GENERATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total attempts: {stats['total_attempts']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Parse errors: {stats['parse_errors']}")
    logger.info(f"Validation errors: {stats['validation_errors']}")
    logger.info(f"Duplicates: {stats['duplicates']}")
    logger.info(f"Validity rate: {stats['validity_rate']:.2%}")
    logger.info(f"Unique rate: {stats['unique_rate']:.2%}")
    logger.info(f"Total tokens in: {stats['total_tokens_in']}")
    logger.info(f"Total tokens out: {stats['total_tokens_out']}")
    logger.info("="*60)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate unconditional AtomForge DSL programs using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing seed .atomforge files'
    )
    
    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs/uncond',
        help='Output directory for generated programs and metrics'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=500,
        help='Number of programs to generate'
    )
    
    parser.add_argument(
        '--seed_k',
        type=int,
        default=20,
        help='Number of seed programs to include in few-shot context'
    )
    
    parser.add_argument(
        '--max_retries',
        type=int,
        default=2,
        help='Maximum repair attempts per program'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5.1-thinking',
        help='LLM model name'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Top-p sampling parameter'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Dry run mode: do not call API, just validate existing files'
    )
    
    args = parser.parse_args()
    
    # Run generation pipeline
    generate_unconditional_programs(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        seed_k=args.seed_k,
        max_retries=args.max_retries,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()

