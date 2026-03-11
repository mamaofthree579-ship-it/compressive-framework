import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- 1. SET UP PAGE & STYLING ---
st.set_page_config(page_title="FGD Theory Tester", layout="wide")
st.title("🌌 Fractal Gravity Dynamics (FGD) Research Dashboard")
st.markdown("""
**Research Partner:** Testing 'Mathematical Expansion & Simulation' by Hope Jones.
This tool evaluates the $r^{-4}$ correction against Black Hole (EHT) and Galactic (.DENS) data.
""")

# --- 2. SIDEBAR PARAMETERS ---
st.sidebar.header("Global Theory Constants")
L_f = st.sidebar.slider("Fractal Length Scale (ℓ_f)", 0.0, 10.0, 1.2, 
                       help="The characteristic scale where fractal repulsion begins.")
D_t = st.sidebar.slider("Temporal Dimension (D_t)", 0.5, 1.0, 0.81,
                       help="Hope Jones's proposed temporal fractal dimension.")

# --- 3. CORE PHYSICS FUNCTIONS ---
def calculate_fgd_potential(r, M, L_f):
    G_M = 1.0 # Normalized
    # V = -GM/r + (GM * L_f^2) / (3 * r^3)
    return -G_M / r + (G_M * L_f**2) / (3 * r**3)

def parse_dens_file(uploaded_file):
    # Reads [x y z rho] format common in .DENS files
    content = uploaded_file.getvalue().decode("utf-8")
    # Using sep=r'\s+' to handle spaces/tabs
    df = pd.read_csv(io.StringIO(content), sep=r'\s+', names=['x', 'y', 'z', 'rho'], comment='#')
    
    # Calculate radius and mass
    df['r'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df = df.sort_values('r')
    df['mass_enclosed'] = df['rho'].cumsum() # Cumulative mass profile
    
    # Binning for smoothness
    bins = np.linspace(df['r'].min(), df['r'].max(), 50)
    df['bin'] = pd.cut(df['r'], bins)
    binned = df.groupby('bin', observed=True).agg({'r': 'mean', 'mass_enclosed': 'max'}).dropna()
    return binned['r'].values, binned['mass_enclosed'].values

# --- 4. MODULE 1: BLACK HOLE SINGULARITY TESTER ---
tab1, tab2 = st.tabs(["🕳️ Black Hole Shadow (EHT)", "🌌 Galaxy Density (.DENS)"])

with tab1:
    st.subheader("Singularity Resolution & Shadow Diameter")
    col1_a, col1_b = st.columns([2, 1])
    
    r_bh = np.linspace(0.4, 10, 500)
    v_gr = -1.0 / r_bh
    v_fgd = calculate_fgd_potential(r_bh, 1.0, L_f)
    r_core = np.sqrt(3) * L_f
    
    with col1_a:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(r_bh, v_gr, 'k--', label="Standard GR (Singularity)")
        ax1.plot(r_bh, v_fgd, 'r-', label="FGD (Fractal Core)")
        ax1.axvspan(0, r_core, color='red', alpha=0.1, label="Repulsive Zone")
        ax1.set_ylim(-4, 1)
        ax1.set_ylabel("Potential (V)")
        ax1.legend()
        st.pyplot(fig1)
        
    with col1_b:
        eht_limit = 5.2 # M87* shadow size in normalized units
        deviation = (r_core / eht_limit) * 100
        st.metric("Shadow Deviation", f"{deviation:.2f}%", 
                  delta="Falsified" if deviation > 15 else "Consistent", 
                  delta_color="inverse")
        st.write(f"**Core Radius:** {r_core:.2f}")
        st.info("If Deviation > 15%, the fractal core would be visible to EHT, potentially falsifying the theory.")

# --- 5. MODULE 2: .DENS GALAXY TESTER ---
with tab2:
    st.subheader("Rotation Curve from Simulation Data")
    uploaded_file = st.file_uploader("Upload .DENS or .TXT file (Format: x y z rho)", type=["dens", "txt"])
    
    if uploaded_file:
        try:
            r_data, m_data = parse_dens_file(uploaded_file)
            
            # Velocity Calculations
            v_classic = np.sqrt(m_data / r_data)
            # FGD Velocity includes the r^-4 force correction integrated
            v_fgd_curve = np.sqrt((m_data / r_data) + (m_data * (L_f**2) / r_data**3))
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(r_data, v_classic, 'k--', label="Newtonian (No Dark Matter)")
            ax2.plot(r_data, v_fgd_curve, 'b-', label="FGD Prediction")
            ax2.set_xlabel("Radius (r)")
            ax2.set_ylabel("Velocity (v)")
            ax2.legend()
            st.pyplot(fig2)
            
            st.success("Analysis Complete: Check if the blue line stays 'flat' at high radii.")
        except Exception as e:
            st.error(f"Error: {e}. Ensure the file is space-separated columns of x, y, z, and density.")
    else:
        st.warning("Please upload a .DENS file from the research datasets to see results.")

# --- 6. TEMPORAL DYNAMICS ---
st.divider()
st.subheader("Fractal Temporal Flux (D_t Scaling)")
st.write(f"Simulating signal noise for D_t = {D_t}")
t = np.linspace(1, 100, 1000)
# Power law noise simulation
signal = np.cumsum(np.random.normal(0, 1, 1000) * (t**(D_t - 1)))
st.line_chart(signal)
