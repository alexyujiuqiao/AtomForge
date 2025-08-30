from dash import Dash, html, Input, Output, State, dcc
from flask_caching import Cache
from pymatgen.io.vasp import Poscar
from crystal_toolkit.components.structure import StructureMoleculeComponent

# Import modules with fallback to avoid circular imports
try:
    # Try relative imports first (when running as part of package)
    from ..src.parser import atomforge_parser
    from ..src.compiler import code_generator
    from ..src.converters import converter
    from ..src.converters import input2atomforge
    
    parse_atomforge_string = atomforge_parser.parse_atomforge_string
    CodeGenerator = code_generator.CodeGenerator
    convert_material = converter.convert_material
    to_atomforgeDSL = input2atomforge.to_atomforgeDSL
    critic = input2atomforge.critic
    iterative_critic = input2atomforge.iterative_critic
    
except ImportError:
    # Fallback for when running as standalone script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.parser.atomforge_parser import parse_atomforge_string
    from src.compiler.code_generator import CodeGenerator
    from src.converters.converter import convert_material
    from src.converters.input2atomforge import to_atomforgeDSL
    from src.converters.input2atomforge import critic, iterative_critic

# Import AtomForge converter
try:
    from ..src.converters import atomforge_converter
    AtomForgeConverter = atomforge_converter.AtomForgeConverter
    ATOMFORGE_CONVERTER_AVAILABLE = True
except ImportError:
    try:
        from src.converters.atomforge_converter import AtomForgeConverter
        ATOMFORGE_CONVERTER_AVAILABLE = True
    except ImportError as e:
        ATOMFORGE_CONVERTER_AVAILABLE = False
        print(f"Warning: AtomForge converter not available: {e}. Using fallback methods.")

import tempfile
import os

# Initialize Dash
app = Dash(__name__)
server = app.server
cache = Cache(server, config={"CACHE_TYPE": "SimpleCache"})
app.title = "AtomForge DSL - VASP Crystal Visualizer"

# ---------------- Layout ----------------
app.layout = html.Div([
    html.H2("AtomForge DSL Crystal 3D Visualizer"),

    html.Div([
        html.H4("Search by Materials Project ID, Element Name, or Chemical Formula"),
        dcc.Input(
            id="material_input",
            placeholder="e.g. mp-149, Si, Fe, Cu, TaTe4, LiFePO4, or SiO2",
            debounce=True,
            style={"width": "400px"}
        ),
        html.Button("Generate", id="btn_mp", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.Div([
        html.H4("Describe the Material (AI-powered)"),
        dcc.Textarea(
            id="user_prompt_input",
            placeholder="e.g. a LiFePO4 cathode material with olivine structure for lithium-ion batteries",
            style={
                "width": "500px",
                "height": "120px",
                "fontSize": "14px",
                "fontFamily": "inherit",
                "resize": "vertical",
                "marginBottom": "10px"
            }
        ),
        html.Div([
            html.Button("Generate", id="btn_llamp", n_clicks=0),
            html.Span("AI Converter Available" if ATOMFORGE_CONVERTER_AVAILABLE else "AI Converter Not Available", 
                     style={"color": "green" if ATOMFORGE_CONVERTER_AVAILABLE else "orange", "fontSize": "12px", "marginLeft": "10px"})
        ]),
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.Div([
            html.H4("Generated AtomForge DSL (editable):"),
            dcc.Textarea(
                id="dsl_box",
                style={
                    "width": "100%",
                    "height": "600px",
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace",
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "backgroundColor": "#f9f9f9"
                }
            ),
            html.Button("Visualize Structure", id="btn_render", n_clicks=0, style={"marginTop": "10px"})
        ], style={"flex": "1", "marginRight": "20px"}),

        html.Div([
            html.H4("Critic Feedback:"),
            dcc.Textarea(
                id="feedback_box",
                style={
                    "width": "100%",
                    "height": "200px",
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace",
                    "border": "1px solid #ccc",
                    "padding": "10px",
                    "backgroundColor": "#f9f9f9"
                },
                readOnly=True
            ),
            html.H4("Crystal 3D Structure:"),
            html.Div(id="viewer", style={
                "flex": "1",
                "minWidth": "500px",
                "border": "1px solid #eee",
                "padding": "10px"
            })
        ], style={"flex": "1"}),
    ], style={
        "display": "flex",
        "flexDirection": "row",
        "alignItems": "flex-start",
        "gap": "20px"
    })
], style={"padding": "20px", "maxWidth": "1400px", "margin": "0 auto"})

# ---------------- DSL Generation Callback ----------------
@app.callback(
    Output("dsl_box", "value"),
    Output("feedback_box", "value"),
    Input("btn_mp", "n_clicks"),
    Input("btn_llamp", "n_clicks"),
    State("material_input", "value"),
    State("user_prompt_input", "value")
)
def generate_dsl(n_clicks_mp, n_clicks_llamp, material_id, user_prompt):
    from dash import ctx
    trigger_id = ctx.triggered_id

    if trigger_id == "btn_mp":
        if not material_id:
            return "Please enter a Materials Project ID, element name, or chemical formula.", ""
        try:
            return convert_material(material_id), ""
        except Exception as e:
            return f"Error retrieving material: {e}", ""

    elif trigger_id == "btn_llamp":
        if not user_prompt:
            return "Please enter your material description.", ""
        
        try:
            if ATOMFORGE_CONVERTER_AVAILABLE:
                # Use AtomForge converter
                converter = AtomForgeConverter()
                dsl_str, metadata = converter.convert(user_prompt)
                
                # Create informative feedback
                feedback_parts = []
                feedback_parts.append("AtomForge Conversion Successful!")
                feedback_parts.append(f"Material: {metadata['material_data']['formula_pretty']}")
                feedback_parts.append(f"Material ID: {metadata['material_data']['material_id']}")
                feedback_parts.append(f"Validation: {metadata['validation']['message']}")
                
                # Add enriched data info
                enriched = metadata['enriched_data']
                feedback_parts.append(f"Computational Backend: {enriched.get('computational_backend', {}).get('functional', 'N/A')}")
                feedback_parts.append(f"Target Properties: {list(enriched.get('target_properties', {}).keys())}")
                
                feedback = "\n".join(feedback_parts)
                return dsl_str, feedback
            else:
                # Fallback to traditional method
                dsl_str, compilation_info, feedback = iterative_critic(user_prompt)
                return dsl_str, f"{compilation_info}\n\n{feedback}"
                
        except Exception as e:
            return f"Error generating with AtomForge converter: {e}", ""

    return "", ""

# ---------------- DSL Rendering Callback ----------------
@app.callback(
    Output("viewer", "children"),
    Input("btn_render", "n_clicks"),
    State("dsl_box", "value")
)
def render_structure(n_clicks_render, dsl_str):
    if not dsl_str:
        return "No DSL provided to render."

    try:
        ir = parse_atomforge_string(dsl_str)
        gen = CodeGenerator(ir)
        tmpdir = tempfile.mkdtemp()
        # Write VASP input files
        with open(os.path.join(tmpdir, "POSCAR"), "w") as f:
            f.write(gen.generate_poscar())
        with open(os.path.join(tmpdir, "INCAR"), "w") as f:
            f.write(gen.generate_incar())
        with open(os.path.join(tmpdir, "KPOINTS"), "w") as f:
            f.write(gen.generate_kpoints())
        with open(os.path.join(tmpdir, "POTCAR"), "w") as f:
            f.write(gen.generate_potcar())
        structure = Poscar.from_file(os.path.join(tmpdir, "POSCAR")).structure

        viewer = StructureMoleculeComponent(structure)
        layout = viewer.layout()
        viewer.generate_callbacks(app, cache)

        return layout

    except Exception as e:
        return f"Error during rendering: {e}"

# ---------------- Launch ----------------
if __name__ == "__main__":
    app.run(debug=True)
