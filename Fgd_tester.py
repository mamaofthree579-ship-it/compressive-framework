import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Universal FGD Tester", layout="wide")
st.title("🌌 Universal Fractal Gravity Research Dashboard")
st.markdown("Testing Hope Jones's **Particle-Time** dynamics against `.DAT` and `.DENS` datasets.")

# --- SIDEBAR ---
st.sidebar.header("Global Theory Constants")
L_f = st.sidebar.slider("Fractal Length (ℓ_f)", 0.0, 10.0, 1.2)
G_M = 1.0 # Normalized Gravity-Mass constant

# --- ROBUST PARSER ---
def universal_parser(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8")
    # Clean common scientific headers like '#' or '!'
    lines = [line for line in content.splitlines() if not line.strip().startswith(('#', '!', 'index'))]
    
    # Try reading with any whitespace separator
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=r'\s+', header=None)
    
    # AUTO-DETECT FORMAT
    cols = len(df.columns)
    
    if cols >= 4: # Likely 3D Density: x, y, z, rho
        st.info("Format Detected: 3D Density Map (x, y, z, ρ)")
        df.columns = ['x', 'y', 'z', 'rho'] + list(df.columns[4:])
        df['r'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        df = df.sort_values('r')
        df['mass_enclosed'] = df['rho'].cumsum()
        return df['r'].values, df['mass_enclosed'].values, "density"
    
    elif cols >= 2: # Likely 2D Rotation Curve: radius, velocity
        st.info("Format Detected: 2D Rotation Curve (r, v)")
        df.columns = ['r', 'v_obs'] + list(df.columns[2:])
        return df['r'].values, df['v_obs'].values, "curve"
    
    else:
        raise ValueError("Unknown file structure. Ensure at least 2 columns exist.")

# --- UI TABS ---
tab1, tab2 = st.tabs(["📊 Galaxy Data Analysis", "🕳️ Black Hole Core Theory"])

with tab1:
    uploaded_file = st.file_uploader("Upload .DAT or .DENS file", type=["dat", "dens", "txt"])
    
    if uploaded_file:
        try:
            r_raw, val_raw, mode = universal_parser(uploaded_file)
            
            # MATH: FGD Velocity Prediction
            if mode == "density":
                # v = sqrt( (GM/r) + (3GM*Lf^2 / r^3) )
                v_classic = np.sqrt(G_M * val_raw / r_raw)
                v_fgd = np.sqrt((G_M * val_raw / r_raw) + (3 * G_M * val_raw * (L_f**2) / r_raw**3))
                y_label = "Velocity (v)"
            else:
                # Compare observed curve to theoretical prediction
                v_classic = np.sqrt(G_M * 100 / r_raw) # 100 is dummy mass for curve test
                v_fgd = np.sqrt((G_M * 100 / r_raw) + (3 * G_M * 100 * (L_f**2) / r_raw**3))
                y_label = "Velocity (v)"

            fig, ax = plt.subplots(figsize=(10, 5))
            if mode == "curve":
                ax.scatter(r_raw, val_raw, color='gray', alpha=0.5, label="Observed Data (.DAT)")
            
            ax.plot(r_raw, v_classic, 'k--', label="Newtonian (Expected)")
            ax.plot(r_raw, v_fgd, 'b-', linewidth=2, label="FGD Prediction")
            ax.set_xlabel("Radius (r)")
            ax.set_ylabel(y_label)
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Parser Error: {e}")

with tab2:
    st.write(f"**Current Balance Point ($r_{{core}}$):** {np.sqrt(3)*L_f:.2f}")
    st.markdown("Check if $r_{core}$ overlaps with known EHT shadow data (~5.5 units).")
