import streamlit as st

st.set_page_config(page_title="Feathered Serpent Propulsion", layout="wide")

st.title("🧶 Khipu Node v22: The Feathered Hull")
st.write("Simulating the Fluid Dynamics of the 'Serpent' Maritime Engine.")

# --- SIDEBAR: DESIGN CONTROLS ---
st.sidebar.header("🛶 Hull Engineering")
hull_flex = st.sidebar.slider("Hull Flexibility (Serpent Mode)", 0, 100, 85)
wind_capture = st.sidebar.slider("Sail Efficiency (Feather Mode)", 0, 100, 90)

# --- FLUID DYNAMICS LOGIC ---
# High flex + high wind = Peak Propulsion
propulsion_efficiency = (hull_flex + wind_capture) / 2
drag_reduction = hull_flex * 0.4 # More 'snake-like' = less resistance

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🐍 Serpentine Hydrodynamics")
    st.metric("Drag Reduction", f"{drag_reduction:.1f}%")
    st.write("**Mechanism:** Flex-hull technology dissipating wave torque.")
    st.progress(hull_flex / 100)

with col2:
    st.subheader("🪶 Feathered Propulsion")
    st.metric("Engine Efficiency", f"{propulsion_efficiency:.1f}%")
    if propulsion_efficiency >= 85:
        st.success("🌊 MARITIME LOCK: The ship is 'flying' on the Kelp Highway.")
    else:
        st.info("⚓ COASTAL MODE: Standard rowing/paddling propulsion.")

st.divider()
st.subheader("🧬 Result: The Living Engine")
st.write(f"The 'String' is vibrating in a **Propulsion Node** of {propulsion_efficiency}%.")
st.write("In String Theory terms, the Feathered Serpent is the **Fluid Brane** that allows for zero-resistance travel.")
