import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="FGD Nested Theory", layout="wide")
st.title("🛡️ The 'Nested' Structure Theory Tester")
st.markdown("""
**Concept:** Regular Matter (Core) creates a 'Compiled' outer layer (Dark Matter) 
and a 'Movement' boundary (Dark Energy). 
""")

# --- SIDEBAR: The Coupling Constants ---
st.sidebar.header("Fractal Coupling")
zeta = st.sidebar.slider("Coupling Constant (ζ)", 0.0, 1.0, 0.15, 
                        help="The ratio of Compiled Gravity (Dark Matter) to Regular Matter.")
ell_f = st.sidebar.slider("Fractal Scale (ℓ_f)", 0.0, 10.0, 2.5)
H_f = st.sidebar.slider("Expansion Movement (Dark Energy)", 0.0, 100.0, 67.0)

# --- PHYSICS LOGIC ---
r = np.linspace(1, 50, 500)
M_core = 100.0 # Our 'Regular' visible matter

# 1. Regular Gravity (Newtonian Core)
v_regular = np.sqrt(M_core / r)

# 2. 'Compiled' Outer Structure (Dark Matter Layer)
# This is derived from the core mass but 'compiled' via fractal interference
v_compiled = np.sqrt(zeta * (M_core / r) + (3 * M_core * ell_f**2 / r**3))

# 3. Total Gravitational Grip (Core + Compiled Layer)
v_total = np.sqrt(v_regular**2 + v_compiled**2)

# 4. 'Movement' Energy (Dark Energy Boundary)
# Represents the expansion pushing outward against the inward grip
v_expansion = (H_f / 1000) * r 

# --- VISUALIZATION ---
st.subheader("The Nested Galactic Profile")
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(r, v_regular, 'g--', label="Regular Matter (Visible Core)")
ax.fill_between(r, v_regular, v_total, color='blue', alpha=0.2, label="Compiled Structure (Dark Matter)")
ax.plot(r, v_total, 'b-', linewidth=2, label="Total Gravitational Grip")
ax.plot(r, v_expansion, 'r:', label="Expansion Movement (Dark Energy)")

ax.set_xlabel("Radius from Center (kpc)")
ax.set_ylabel("Velocity / Energy Flux (km/s)")
ax.set_title("Interaction of Visible Core and Fractal 'Dark' Boundaries")
ax.legend()
st.pyplot(fig)

# --- THE FALSIFICATION CHECK ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    # Check if the Expansion Energy overcomes Gravity (The Big Rip check)
    rip_point = r[np.where(v_expansion > v_total)[0][0]] if any(v_expansion > v_total) else None
    if rip_point:
        st.warning(f"⚠️ Boundary Breach: Expansion overcomes Gravity at {rip_point:.1f} kpc.")
    else:
        st.success("✅ Stable Structure: The Nested Gravity holds the system together.")

with col2:
    st.info(f"""
    **Current Analysis:**
    - The Blue Zone represents the 'Extra Grip' provided by her **Compiled Quantum Gravity**.
    - If **ζ (Zeta)** is constant for all galaxies, Dark Matter is proven to be a geometric shadow of regular matter.
    """)
