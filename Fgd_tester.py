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

# --- NEW: Galaxy Data Uploader Section ---
st.divider()
st.header("📊 Galaxy Rotation Curve Tester")
st.markdown("Upload a CSV with `radius` and `velocity` columns to test the Dark Matter vs. FGD claim.")

uploaded_file = st.file_uploader("Upload SPARC or custom CSV", type="csv")

if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    
    # Required columns check
    if 'radius' in df.columns and 'velocity' in df.columns:
        # Physics Calculation for Galaxy Scale
        # In FGD, the 'Extra' velocity comes from the r^-4 term integration
        r_gal = df['radius'].values
        v_obs = df['velocity'].values
        
        # Theoretical FGD Velocity (Baryonic + Fractal Correction)
        # Using the L_f from the sidebar scaled to galactic kpc
        v_fgd_gal = np.sqrt((G_M * M / r_gal) + (G_M * M * (L_f**2) / r_gal**3))
        
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.scatter(r_gal, v_obs, color='black', label='Observed Data (SPARC)', alpha=0.5)
        ax3.plot(r_gal, v_fgd_gal, color='blue', label='FGD Prediction (No Dark Matter)')
        
        ax3.set_xlabel("Radius (kpc)")
        ax3.set_ylabel("Velocity (km/s)")
        ax3.legend()
        st.pyplot(fig3)
        
        # Falsification Metric
        rmse = np.sqrt(np.mean((v_obs - v_fgd_gal)**2))
        st.metric("Model Fit Error (RMSE)", f"{rmse:.2f} km/s")
    else:
        st.error("CSV must contain 'radius' and 'velocity' columns.")
