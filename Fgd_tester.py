import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="FGD Cohesion Meter", layout="wide")
st.title("💎 The Jones Cohesion Meter: 97.9% Search")
st.markdown("""
**The Theoretical Goal:** Balance the 3D Material within the 4D Landscape. 
Jones suggests a 'Perfect Coherence' occurs when 4D Flux and Thermodynamic Micro-bursts align.
""")

# --- SIDEBAR: Tuning the Resonance ---
st.sidebar.header("Resonance Tuning")
psi = st.sidebar.slider("4D Flux Density (Ψ)", 0.5, 2.0, 1.25)
freq = st.sidebar.slider("Micro-burst Frequency (Hz)", 0.01, 0.1, 0.05, 
                         help="Jones's Temporal Fractal Dimension D_t ≈ 0.81")
mass_3d = st.sidebar.slider("3D Material Mass", 50, 200, 100)

# --- PHYSICS: Cohesion Calculation ---
r = np.linspace(1, 100, 500)

# 1. Potential Energy of the 3D Material
pe_3d = -mass_3d / r

# 2. Resonance Energy of the 4D Flux
# E_res = Ψ * mass * cos(freq * r)
e_res = psi * mass_3d * np.cos(freq * r / 5)

# 3. Calculate Cohesion (Cross-Correlation between Material and Field)
# A 'Perfect' match (1.0) means the 4D field perfectly supports the 3D mass.
correlation = np.abs(np.corrcoef(pe_3d, e_res)[0, 1])
cohesion_score = correlation * 100

# --- VISUALIZATION: The Cohesion Meter ---
st.subheader("System Stability Analysis")
col1, col2 = st.columns([1, 2])

with col1:
    # Gauge-style display
    st.metric("Cohesion Score", f"{cohesion_score:.2f}%", 
              delta=f"{cohesion_score - 97.9:.2f}% from Jones Constant")
    
    if 97.0 <= cohesion_score <= 98.5:
        st.balloons()
        st.success("🎯 HARMONIC LOCK: You've hit the 97.9% Cohesiveness!")
    elif cohesion_score > 98.5:
        st.warning("⚠️ OVER-COHERENT: System may be too rigid (Crystal State).")
    else:
        st.error("❌ DECAYING: Resonance is too low for 4D stability.")

with col2:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0e1117')
    ax.set_facecolor('#1e1e1e')
    
    # Plotting the 'Interference Pattern' of Space-Time
    ax.plot(r, pe_3d/10, color='gray', linestyle='--', label="3D Gravity Well")
    ax.plot(r, e_res/100, color='cyan', label="4D Flux Resonance")
    
    ax.fill_between(r, pe_3d/10, e_res/100, color='magenta', alpha=0.2, label="Energy Flux")
    
    ax.set_ylim(-3, 3)
    ax.legend(labelcolor='white')
    st.pyplot(fig)

# --- INTERPRETATION ---
st.divider()
st.info(f"""
**The Physics of the Result:**
- **The Cyan Wave:** This is the 'Pulsing' Dark Energy background.
- **The Grey Line:** This is the 'Static' 3D Matter.
- **The Gap:** Where the lines meet, the galaxy is 'Tethered' to the 4th dimension. 
- **The 97.9% Goal:** This represents the maximum thermodynamic efficiency of a galaxy before it becomes a 'Closed System' or collapses.
""")
