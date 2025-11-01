import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="CF-DPF Simulator", layout="wide")

st.title("🌌 Compressive Framework — Dynamic Particle Formation (CF-DPF)")
st.markdown(
    """
    This simulator models **field interactions** between the three primary waves:
    - **Graviton (γ)** — gravitational compression field  
    - **Chronon (χ)** — temporal coherence field  
    - **Cognon (κ)** — informational resonance field  

    Adjust the parameters below to explore wave coupling, curvature fields, and emergent domains.
    """
)

# Sidebar controls
st.sidebar.header("⚙️ Simulation Controls")
alpha = st.sidebar.slider("Graviton weight (α)", 0.0, 2.0, 0.7, 0.05)
beta = st.sidebar.slider("Chronon weight (β)", 0.0, 2.0, 0.5, 0.05)
gamma = st.sidebar.slider("Cognon weight (γ)", 0.0, 2.0, 0.6, 0.05)
phase_shift = st.sidebar.slider("Phase shift φ", 0.0, np.pi, 0.5, 0.05)
t = st.sidebar.slider("Time t", 0.0, 2*np.pi, 0.0, 0.05)

# Field grid
x = np.linspace(-5, 5, 400)
X = np.tile(x, (400, 1))
T = np.linspace(0, 2*np.pi, 400).reshape(-1, 1)

# Wave definitions
graviton = np.sin(alpha * X - t)
chronon  = np.sin(beta * X - phase_shift)
cognon   = np.sin(gamma * X - 2*t)

# Superposition
combined = graviton + chronon + cognon
curvature = np.gradient(np.gradient(combined, axis=1), axis=1)

# Plotly surface plot
fig = go.Figure(data=go.Surface(
    z=combined,
    x=x,
    y=np.linspace(0, 2*np.pi, 400),
    colorscale="Viridis",
    showscale=True
))
fig.update_layout(
    title="Wave-Field Superposition Ψ(x,t)",
    scene=dict(
        xaxis_title="Spatial Coordinate x",
        yaxis_title="Temporal Phase t",
        zaxis_title="Field Amplitude Ψ"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=True)

# Optional: curvature plot
st.subheader("Derived Curvature Field 𝒦(x,t)")
st.image(np.flipud(curvature), caption="Curvature Intensity Map", use_column_width=True)

st.markdown("🧩 *Tip:* Tune α, β, γ to explore coupling resonance and symmetry breaking.")
