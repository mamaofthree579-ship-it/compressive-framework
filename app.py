import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Quantum Field Simulator — Compressive Framework (CF-DPF)")
st.write("""
Explore the emergent dynamics of graviton, chronon, and cognon interactions.
Adjust field parameters to visualize compression, curvature, and harmonics.
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
alpha = st.sidebar.slider("α (Compression Strength)", 0.1, 1.0, 0.5, 0.05)
beta = st.sidebar.slider("β (Curvature Factor)", 0.1, 1.0, 0.7, 0.05)
omega = st.sidebar.slider("ω (Frequency)", 1.0, 10.0, 3.0, 0.5)

# Generate field data
x = np.linspace(-5, 5, 400)
t = np.linspace(0, 2*np.pi, 200)
X, T = np.meshgrid(x, t)

compression = np.sin(omega * X) * np.exp(-alpha * X**2)
curvature = beta * np.sin(compression**2 - T)

# ✅ Normalize for safe Streamlit display
curvature_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min())

# Display intensity map
st.subheader("Curvature Intensity Map")
st.image(
    np.flipud(curvature_norm),
    caption="Normalized curvature intensity (𝒦-field)",
    use_column_width=True
)

# Overlayed line plot
fig, ax = plt.subplots(figsize=(7,3))
ax.plot(x, compression[100, :], label="Compression Field ρc")
ax.plot(x, np.gradient(compression[100, :]), label="Curvature Nodes 𝒦'", color="orange")
ax.set_xlabel("Spatial Coordinate x")
ax.set_title("Local Curvature Nodes within Compression Field")
ax.legend()
st.pyplot(fig)

# Frequency response section
gamma_vals = np.linspace(-1, 1, 100)
chi_vals = np.linspace(-1, 1, 100)
G, C = np.meshgrid(gamma_vals, chi_vals)
omega_kappa = alpha * C + beta * G

fig2, ax2 = plt.subplots(figsize=(6,5))
contour = ax2.contourf(G, C, omega_kappa, 50, cmap="plasma")
ax2.set_xlabel("γ (Graviton influence)")
ax2.set_ylabel("χ (Chronon influence)")
ax2.set_title("Cognon Frequency Response ωκ(γ, χ)")
fig2.colorbar(contour, ax=ax2, label="Frequency (normalized)")
st.pyplot(fig2)
