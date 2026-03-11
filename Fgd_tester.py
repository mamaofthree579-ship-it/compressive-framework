import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- CONFIG ---
st.set_page_config(page_title="FGD Falsifier", layout="wide")
st.title("🔬 FGD Theory Falsifier: SPARC Edition")
st.markdown("Testing Hope Jones's $r^{-4}$ model against empirical galaxy data.")

# --- SIDEBAR ---
st.sidebar.header("Model Parameters")
L_f = st.sidebar.slider("Fractal Scale (ℓ_f)", 0.0, 20.0, 5.0)
mass_scale = st.sidebar.slider("Baryonic Mass Scaling", 0.1, 2.0, 1.0)

# --- SPARC PARSER ---
def parse_sparc(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    # SPARC files usually have headers; we skip lines starting with #
    df = pd.read_csv(io.StringIO(content), sep=r'\s+', comment='#', header=None)
    
    # Standard SPARC columns: 
    # 1:Rad 2:Vobs 3:errV 4:Vgas 5:Vdisk 6:Vbulge
    df.columns = ['rad', 'v_obs', 'v_err', 'v_gas', 'v_disk', 'v_bulge'] + list(df.columns[6:])
    return df

# --- APP LOGIC ---
uploaded_file = st.file_uploader("Upload a SPARC .dat or .txt file", type=["dat", "txt"])

if uploaded_file:
    try:
        df = parse_sparc(uploaded_file)
        r = df['rad'].values
        v_obs = df['v_obs'].values
        v_err = df['v_err'].values
        
        # Calculate Baryonic Velocity (Sum of Gas, Disk, Bulge contributions)
        # v_baryonic^2 = v_gas^2 + v_disk^2 + v_bulge^2
        v_bary = np.sqrt(np.abs(df['v_gas']**2 + df['v_disk']**2 + df['v_bulge']**2)) * mass_scale
        
        # APPLY FGD CORRECTION: v_final^2 = v_bary^2 + (3 * G * M * Lf^2 / r^3)
        # We estimate G*M using the baryonic velocity at each point
        v_fgd = np.sqrt(v_bary**2 + (3 * (v_bary**2 * r) * (L_f**2) / r**3))
        
        # --- VISUALIZATION ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(r, v_obs, yerr=v_err, fmt='ko', label='Observed (SPARC)', alpha=0.6)
        ax.plot(r, v_bary, 'g--', label='Baryonic Only (Newtonian)')
        ax.plot(r, v_fgd, 'r-', linewidth=2, label='FGD Prediction')
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel("Velocity (km/s)")
        ax.legend()
        st.pyplot(fig)
        
        # --- THE FALSIFIER METRIC ---
        # Chi-squared: measures how far the red line is from the black dots
        chi_sq = np.sum(((v_obs - v_fgd) / v_err)**2) / len(r)
        
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Reduced Chi-Squared (χ²)", f"{chi_sq:.2f}")
        
        with col2:
            if chi_sq < 1.5:
                st.success("✅ THEORY SUPPORTED: Fits within observation error.")
            elif chi_sq < 5.0:
                st.warning("⚠️ WEAK FIT: Adjust ℓ_f or Mass Scaling.")
            else:
                st.error("❌ THEORY FALSIFIED: Model contradicts galactic data.")
                
    except Exception as e:
        st.error(f"Format Error: {e}. Ensure you are uploading a raw SPARC .dat file.")

st.info("**Instructions:** Go to [SPARC Database](https://astroweb.case.edu), pick a galaxy (e.g., NGC3198), and download the 'Rotcur' file.")
