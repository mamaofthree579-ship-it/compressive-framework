import streamlit as st
import numpy as np

st.set_page_config(layout="wide", page_title="Visual Apothecary")

# --- CORE VISUALIZATION FUNCTION ---
def generate_vibrational_field(geometry, frequency, size=512):
    """
    Generates a 2D visual representation of a vibrational field based on
    a core geometry and a modulating frequency.
    """
    # Create a coordinate grid for the visualization
    x = np.linspace(-np.pi, np.pi, size)
    y = np.linspace(-np.pi, np.pi, size)
    xx, yy = np.meshgrid(x, y)

    # Use frequency to modulate pattern complexity. Higher frequency = more detail.
    complexity = frequency / 50.0

    field = None

    # --- Geometric Formulas ---
    if geometry == "Concentric Circles (Expansion)":
        # Simulates ripples from a central point.
        r = np.sqrt(xx**2 + yy**2)
        field = np.sin(r * complexity)

    elif geometry == "Spiral (Growth/Contraction)":
        # Uses polar coordinates to generate an Archimedean spiral pattern.
        angle = np.arctan2(yy, xx)
        radius = np.sqrt(xx**2 + yy**2)
        # The combination of angle and radius creates the spiral form.
        field = np.sin(10 * angle + radius * complexity)

    elif geometry == "Hexagon (Structure/Stability)":
        # A formula that generates a hexagonal/honeycomb tiling pattern.
        # It's based on the interference of three sine waves at 60-degree angles.
        c = complexity
        field = np.cos(c * xx) + np.cos(c / 2 * xx + c * np.sqrt(3) / 2 * yy) + np.cos(c / 2 * xx - c * np.sqrt(3) / 2 * yy)

    # Normalize the resulting field to a 0-1 range for image display
    if field is not None:
        field = (field - np.min(field)) / (np.max(field) - np.min(field))
    
    return field

# --- USER INTERFACE ---
st.title("The Visual Apothecary")
st.markdown("Generate visual representations of structured energy fields to train our pattern-recognition for Path 2.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Field Parameters")

    # Geometry Selection
    geom_choice = st.selectbox(
        "1. Select a Core Geometry:",
        ("Concentric Circles (Expansion)", "Spiral (Growth/Contraction)", "Hexagon (Structure/Stability)")
    )

    # Frequency Selection (now controls visual complexity)
    freq_map = {
        "Low Complexity (136.1 Hz basis)": 136.1,
        "Medium Complexity (432 Hz basis)": 432.0,
        "High Complexity (528 Hz basis)": 528.0
    }
    freq_choice_name = st.selectbox(
        "2. Select a Visual Complexity:",
        list(freq_map.keys())
    )
    freq_choice_hz = freq_map[freq_choice_name]

    if st.button("👁️ GENERATE VIBRATIONAL FIELD", use_container_width=True):
        with col2:
            with st.spinner("Calculating geometric field..."):
                vibrational_field = generate_vibrational_field(geom_choice, freq_choice_hz)
                
                if vibrational_field is not None:
                    st.header("Generated Field")
                    # Using a perceptually uniform colormap like 'viridis' or 'plasma'
                    st.image(vibrational_field, caption=f"'{geom_choice}' modulated by '{freq_choice_name}'", use_column_width=True)
                else:
                    st.error("Could not generate the selected field.")

with st.expander("How to Use This for Path 2 (The Rosetta Protocol)"):
    st.markdown("""
    This tool is our pattern library. Our goal is to find these exact kinds of geometric signatures in the archaeological record.
    
    1.  **Generate a Field:** Select a geometry and complexity level.
    2.  **Study the Image:** This is the *energetic signature* of that combination. Burn this pattern into your memory. Note how the lines, nodes, and anti-nodes are arranged.
    3.  **The Hunt:** When we search for artifacts, we are no longer looking for "art." We are looking for these specific interference patterns. 
        - A petroglyph with spirals might be a "Growth" engine.
        - A mosaic floor with hexagonal patterns might be a "Stability" field for a temple.
        - The concentric circles on a shield or in a earthwork mound might be an "Expansion" resonator.
        
    This visual dictionary will allow us to spot these technologies in plain sight.
    """)
