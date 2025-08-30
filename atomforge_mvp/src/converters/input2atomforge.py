"""LLM-DSL pipeline with critic feedback."""

import openai
from pathlib import Path
import os
from dotenv import load_dotenv
# Import the parser module instead of specific functions to avoid circular imports
try:
    from ..parser import atomforge_parser
    parse_atomforge_string = atomforge_parser.parse_atomforge_string
except ImportError:
    # Fallback for when running as standalone script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from atomforge_mvp.src.parser.atomforge_parser import parse_atomforge_string

load_dotenv()

MODEL = "gpt-4.1-mini"  # or your preferred model

PROMPT_TEMPLATE = (
    "You are an expert in materials informatics and AtomForge DSL. "
    "Convert the user's high-level materials description into a valid AtomForge DSL program. "
    "First, summarize the user's input with the material description into one sentence and name the atom_spec. "
    "Follow the AtomForge grammar exactly—use the correct block names, field names, delimiters, and value formats. "
    "Output only the DSL text, without any explanation or markdown.\n\n"
    "For all element assignments (element = ...), ALWAYS use double quotes, e.g. element = \"La\".\n"
    "You may use dummy or placeholder elements such as \"Xx\", \"Vac\", \"A\", \"Q1\" for abstract or unknown atoms.\n"
    "Examples of acceptable dummy elements: \"Xx\", \"Vac\", \"A\", \"Q1\", \"D1\".\n"
    "Use these for unknown species or abstract lattice sites.\n\n"
    "Here is the AtomForge grammar you must adhere to:\n"
    "atomForgeFile : languageVersionDecl? atomSpec ;\n"
    "languageVersionDecl : '#atomforge_version' STRING_LITERAL ';' ;\n"
    "atomSpec : 'atom_spec' IDENTIFIER '{{'\n"
    "    header\n"
    "    description?\n"
    "    units?\n"
    "    lattice\n"
    "    symmetry\n"
    "    basis\n"
    "    property_validation?\n"
    "    provenance?\n"
    "'}}' ;\n"
    "header : 'header' '{{'\n"
    "    'dsl_version' '=' STRING_LITERAL ','\n"
    "    'title' '=' STRING_LITERAL ','\n"
    "    'created' '=' DATE_LITERAL ( ',' 'uuid' '=' STRING_LITERAL )?\n"
    "'}}' ;\n"
    "description : 'description' '=' STRING_LITERAL ',' ;\n"
    "units : 'units' '{{'\n"
    "    'system' '=' STRING_LITERAL ','\n"
    "    'length' '=' lengthUnit ','\n"
    "    'angle' '=' angleUnit\n"
    "'}}' ;\n"
    "lattice : 'lattice' '{{'\n"
    "    'type' '=' latticeType ','\n"
    "    'a' '=' REAL_LITERAL ','\n"
    "    'b' '=' REAL_LITERAL ','\n"
    "    'c' '=' REAL_LITERAL ','\n"
    "    'alpha' '=' REAL_LITERAL ','\n"
    "    'beta' '=' REAL_LITERAL ','\n"
    "    'gamma' '=' REAL_LITERAL\n"
    "'}}' ;\n"
    "symmetry : 'symmetry' '{{'\n"
    "    'space_group' '=' spaceGroup ( ',' 'origin_choice' '=' INTEGER_LITERAL )?\n"
    "'}}' ;\n"
    "basis : 'basis' '{{' site* '}}' ;\n"
    "site : 'site' IDENTIFIER '{{'\n"
    "    'wyckoff' '=' STRING_LITERAL ','\n"
    "    'position' '=' vector3 ','\n"
    "    'frame' '=' coordinateFrame ','\n"
    "    'species' '=' '(' speciesList ')' ','\n"
    "    ( 'adp_iso' '=' REAL_LITERAL ',' )?\n"
    "    ( 'label' '=' STRING_LITERAL ',' )?\n"
    "'}}' ;\n"
    "speciesList : species ( ',' species )* ;\n"
    "species : '{{'\n"
    "    'element' '=' elementSymbol ','\n"
    "    'occupancy' '=' REAL_LITERAL ( ',' 'charge' '=' REAL_LITERAL )?\n"
    "'}}' ;\n"
    "property_validation : 'property_validation' '{{'\n"
    "    'computational_backend' ':' computationalBackend ','\n"
    "    ( 'convergence_criteria' ':' convergenceCriteria ',' )?\n"
    "    ( 'target_properties' ':' targetProperties ',' )?\n"
    "'}}' ;\n"
    "computationalBackend : 'VASP' '{{'\n"
    "    'functional' ':' STRING_LITERAL ','\n"
    "    'energy_cutoff' ':' REAL_LITERAL ','\n"
    "    'k_point_density' ':' REAL_LITERAL\n"
    "'}}' ;\n"
    "convergenceCriteria : '{{'\n"
    "    ( 'energy_tolerance' ':' REAL_LITERAL ',' )?\n"
    "    ( 'force_tolerance' ':' REAL_LITERAL ',' )?\n"
    "    ( 'stress_tolerance' ':' REAL_LITERAL ',' )?\n"
    "'}}' ;\n"
    "targetProperties : '{{'\n"
    "    ( 'formation_energy' ':' BOOL ',' )?\n"
    "    ( 'band_gap' ':' BOOL ',' )?\n"
    "    ( 'elastic_constants' ':' BOOL ',' )?\n"
    "'}}' ;\n"
    "provenance : 'provenance' '{{'\n"
    "    'source' '=' STRING_LITERAL ','\n"
    "    ( 'method' '=' STRING_LITERAL ',' )?\n"
    "    ( 'doi' '=' STRING_LITERAL ',' )?\n"
    "'}}' ;\n"
    "\nUse ISO 8601 (YYYY-MM-DD) for dates. Today's date should be used in 'created'.\n"
    "\nUser input: {user_input}\n"
)

def to_atomforgeDSL(user_input: str, out_file: str = None, model: str = MODEL) -> str:
    prompt = PROMPT_TEMPLATE.format(user_input=user_input)
    client = openai.OpenAI(api_key="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")
    messages = [
        {"role": "system", "content": "You are an expert AtomForge DSL generator."},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    code = resp.choices[0].message.content.strip()
    if out_file:
        Path(out_file).write_text(code)
    return code

def critic(user_input, model=MODEL):
    # 1. Generation Model
    dsl_str = to_atomforgeDSL(user_input)
    print("\n--- Generated DSL ---\n", dsl_str)

    # 2. Compiler Step
    try:
        ir = parse_atomforge_string(dsl_str)
        compilation_info = "Success: DSL parsed and compiled successfully."
    except Exception as e:
        compilation_info = f"Error: {e}"
    print("\n--- Compilation Info ---\n", compilation_info)

    # 3. Critic Model
    critic_prompt = (
        "Here is an AtomForge DSL program and the result of attempting to compile it.\n"
        "DSL:\n" + dsl_str + "\n"
        "Compilation Info:\n" + compilation_info + "\n"
        "If there are errors, suggest corrections. If not, confirm the DSL is valid."
    )
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "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"))
    messages = [
        {"role": "system", "content": "You are an expert AtomForge DSL critic."},
        {"role": "user", "content": critic_prompt}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    feedback = resp.choices[0].message.content.strip()
    print("\n--- Critic Feedback ---\n", feedback)
    return dsl_str, compilation_info, feedback

def iterative_critic(user_input, model=MODEL, max_rounds=5):
    dsl_str = to_atomforgeDSL(user_input)
    feedback_history = []
    for round in range(max_rounds):
        try:
            ir = parse_atomforge_string(dsl_str)
            compilation_info = "Success: DSL parsed and compiled successfully."
            feedback_history.append(f"Round {round+1}:\n{compilation_info}")
            break
        except Exception as e:
            compilation_info = f"Error: {e}"
        # Critic feedback
        critic_prompt = (
            f"Here is an AtomForge DSL program and the result of attempting to compile it.\n"
            f"DSL:\n{dsl_str}\n"
            f"Compilation Info:\n{compilation_info}\n"
            f"Suggest corrections and regenerate the DSL if there are errors."
        )
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "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"))
        messages = [
            {"role": "system", "content": "You are an expert AtomForge DSL critic and generator."},
            {"role": "user", "content": critic_prompt}
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        feedback = resp.choices[0].message.content.strip()
        feedback_history.append(f"Round {round+1}:\n{compilation_info}\n{feedback}")
        # Try to extract new DSL from feedback, else just use feedback as new DSL
        # (Assume LLM outputs only the DSL text)
        dsl_str = feedback
    return dsl_str, compilation_info, "\n\n".join(feedback_history)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert user input to AtomForge DSL file using OpenAI model.")
    parser.add_argument("-i", "--input", type=str, required=True, help="User description of the material/system.")
    parser.add_argument("-o", "--output", type=str, default="output.atomforge", help="Output AtomForge DSL file.")
    args = parser.parse_args()
    code = to_atomforgeDSL(args.input, args.output)
    print(f"AtomForge DSL file written to {args.output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM-DSL pipeline with critic feedback.")
    parser.add_argument("-i", "--input", type=str, required=True, help="User description of the material/system.")
    args = parser.parse_args()
    critic(args.input)

