import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- App Config ---
st.set_page_config(page_title="FGD Theory Tester", layout="wide")
st.title("🌌 Fractal Gravity Dynamics (FGD) Analyzer")
st.markdown("""
**Objective:** Test Hope Jones's $r^{-4}$ potential correction against General Relativity. 
Find the 'Falsification Point' by comparing the Fractal Core to M87* EHT data.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
M = st.sidebar.slider("Black Hole Mass (M_sun x 10^9)", 1.0, 10.0, 6.5) # M87* is ~6.5
L_f = st.sidebar.slider("Fractal Length (l_f)", 0.0, 5.0, 1.2, help="The Hope Jones correction factor")
r_range = st.sidebar.slider("Radius Range (GM/c^2)", 0.1, 15.0, (0.5, 10.0))

# --- Constants & Physics Logic ---
G_M = 1.0  # Normalized units
r = np.linspace(r_range[0], r_range[1], 1000)

# Potentials
v_gr = -G_M / r  # Simplified Newtonian/GR Limit
# FGD Potential: V = -GM/r + (GM * l_f^2) / (3 * r^3)
v_fgd = -G_M / r + (G_M * L_f**2) / (3 * r**3)

# Force Balance (The 'Repulsion' Check)
# Repulsion dominates where r < sqrt(3) * L_f
r_core = np.sqrt(3) * L_f

# --- Visualizations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Potential Well Analysis")
    fig, ax = plt.subplots()
    ax.plot(r, v_gr, 'k--', label="Standard GR (Singularity)")
    ax.plot(r, v_fgd, 'r-', linewidth=2, label="FGD (Fractal Core)")
    
    if L_f > 0:
        ax.axvspan(0, r_core, color='red', alpha=0.1, label="Repulsive Core Zone")
        ax.axvline(r_core, color='red', linestyle=':', alpha=0.5)
    
    ax.set_ylim(-3, 1)
    ax.set_xlabel("Radius (r)")
    ax.set_ylabel("Potential (V)")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Comparison to EHT M87* Shadow")
    st.write(f"**Calculated Core Radius:** {r_core:.2f} units")
    
    # EHT Shadow for M87* is approx 5.5 units (3*sqrt(3))
    eht_shadow_limit = 5.2 
    deviation = ((r_core / eht_shadow_limit) * 100) if L_f > 0 else 0
    
    st.metric("Shadow Deviation", f"{deviation:.2f}%", 
              delta="-Falsified" if deviation > 10 else "Potential Support",
              delta_color="inverse")

    st.info("""
    **Testing Logic:**
    1. If **Shadow Deviation > 10%**, the Fractal Core is too large and contradicts EHT's clean circular shadow.
    2. If **Deviation is < 2%**, the theory is 'Observationally Consistent' with current resolution limits.
    """)

# --- Temporal Scaling Section ---
st.divider()
st.subheader("Fractal Time Check (D_t ≈ 0.81)")
st.write("Hope Jones predicts light fluctuations follow a fractal power-law.")
# Simulated Light Curve
t = np.linspace(0, 100, 1000)
noise = np.random.normal(0, 1, 1000)
fractal_signal = np.cumsum(noise * (t**-0.19)) # Rough D_t scaling simulation

fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(t, fractal_signal, color='orange')
ax2.set_title("Predicted Fractal Light Curve (QPO Signature)")
st.pyplot(fig2)
