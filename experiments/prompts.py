"""
Prompt templates for unconditional AtomForge program generation.

This module contains:
- GENERATION prompt: For generating new unconditional AtomForge programs
- REPAIR prompt: For fixing parsing/validation errors in generated programs
"""

import re
from typing import List, Tuple


def get_generation_prompt(seed_programs: List[Tuple[str, str]]) -> str:
    """
    Generate the main prompt for unconditional AtomForge program generation.
    
    Args:
        seed_programs: List of (file_path, program_text) tuples for few-shot examples
        
    Returns:
        Complete generation prompt string
    """
    # Build few-shot examples section
    examples_section = ""
    if seed_programs:
        examples_section = "\n## Example AtomForge Programs:\n\n"
        for i, (file_path, program_text) in enumerate(seed_programs[:10], 1):  # Limit to 10 examples
            examples_section += f"### Example {i}:\n```\n{program_text}\n```\n\n"
    
    prompt = f"""You are an expert in crystallography and materials science. Your task is to generate a NEW, VALID AtomForge DSL v2.1 program that describes a crystal structure.

CRITICAL REQUIREMENTS:
1. Generate a COMPLETE, VALID AtomForge program following the exact grammar specification.
2. The program must include at minimum:
   - header block (with dsl_version, title, created date, etc.)
   - units block
   - lattice block (with Bravais lattice parameters)
   - symmetry block (with space_group and origin_choice)
   - basis block (with at least one atomic site)
3. DO NOT include any `ai_integration` block in your program.
4. Every basis site name (the identifier after `site`) MUST be unique within the program.
   - Required naming convention: ElementSymbol + index (e.g., Ba1, Ti1, O1, O2, O3, Cl1, Cl2, ...).
   - The alphabetic prefix MUST be a valid chemical element symbol (H, He, Li, ..., Og).
   - The numeric suffix MUST be a positive integer (1, 2, 3, ...).
   - Do NOT use bare element symbols as site IDs (for example, do not use `site "O"` or `site "Ti"`; use `O1`, `O2`, `Ti1`, etc.). 
5. Coordinates must be numeric, typically fractional in [0,1). If `frame = fractional`, each coordinate in `position = (x, y, z)` MUST satisfy 0 ≤ value < 1.
6. Lattice parameters (a, b, c) must be positive numbers > 0.
7. Angles (alpha, beta, gamma) must be in a reasonable range (typically 60-120 degrees).
8. Atom count should be between 2 and 20 sites (based on number of unique sites).
9. Encourage diversity: vary crystal systems (cubic, tetragonal, orthorhombic, hexagonal, etc.) and space groups.
10. Use valid chemical elements and reasonable Wyckoff positions.
11. Ensure all numeric values are physically reasonable.

OUTPUT FORMAT:
- Output ONLY the AtomForge program code.
- Do NOT include markdown code fences (```atomforge or ```).
- Do NOT include any explanatory text before or after the code.
- Start directly with `atom_spec` and end with the closing brace.
- The output should be valid AtomForge DSL that can be parsed directly.

{examples_section}

Now generate a NEW, UNIQUE AtomForge program for an unconditional crystal structure (no target properties, no user conditions). Be creative and generate a diverse structure that is different from the examples above.
"""
    return prompt


def get_repair_prompt(program_text: str, error_message: str) -> str:
    """
    Generate a repair prompt for fixing parsing/validation errors.
    
    Args:
        program_text: The AtomForge program text that failed
        error_message: The error message from the parser/validator
        
    Returns:
        Complete repair prompt string
    """
    prompt = f"""You are fixing an AtomForge DSL v2.1 program that failed to parse or validate.

The program that failed:
```
{program_text}
```

The error message:
```
{error_message}
```

TASK:
Fix ONLY the errors in the program. Return the CORRECTED AtomForge program code.

REQUIREMENTS:
1. Fix the specific errors mentioned in the error message.
2. DO NOT change parts of the program that are already correct.
3. DO NOT add any `ai_integration` block.
4. Ensure the program follows the AtomForge DSL v2.1 grammar exactly.
5. Maintain the same structure and intent as the original program.
6. Output ONLY the corrected AtomForge code - no markdown, no explanations, no code fences.
7. Start with `atom_spec` and end with the closing brace.

Return the fixed program:
"""
    return prompt


def extract_program_from_response(response: str) -> str:
    """
    Extract the AtomForge program code from an LLM response.
    
    This function handles cases where the model returns:
    - Code wrapped in markdown fences
    - Explanatory text before/after the code
    - Extra formatting
    
    Args:
        response: Raw response from the LLM
        
    Returns:
        Extracted AtomForge program text
    """
    text = response.strip()
    
    # Remove markdown code fences if present
    # Match ```atomforge, ```atomforge_dsl, ```, etc.
    code_fence_pattern = r'^```(?:atomforge|atomforge_dsl|dsl)?\s*\n'
    if re.match(code_fence_pattern, text, re.IGNORECASE):
        # Find the closing fence
        lines = text.split('\n')
        # Remove first line if it's a code fence
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        # Remove last line if it's a closing fence
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines)
    
    # Find the start of the program (atom_spec)
    atom_spec_start = text.find('atom_spec')
    if atom_spec_start >= 0:
        text = text[atom_spec_start:]
    
    # Find the last closing brace that matches the opening
    # Simple approach: count braces from the start
    brace_count = 0
    last_valid_pos = len(text)
    for i, char in enumerate(text):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                last_valid_pos = i + 1
                break
    
    text = text[:last_valid_pos].strip()
    
    # Remove any trailing markdown or explanatory text
    # Look for patterns like "```" or common explanatory phrases
    text = re.sub(r'\n\s*```\s*$', '', text)
    text = re.sub(r'\n\s*(Note|Note:|This|The program|Here|Output):.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text.strip()

