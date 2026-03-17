import streamlit as st

st.set_page_config(page_title="Khipu Aero-Feather Processor", layout="wide")

st.title("🧶 Khipu Node v23: The Aero-Feather Grid")
st.write("Simulating the 'Vortex Control' of Feathered Serpent sail technology.")

# --- SIDEBAR: SAIL PHYSICS ---
st.sidebar.header("🪶 Sail Calibration")
pattern_symmetry = st.sidebar.selectbox("Feather Pattern", ["Random (0)", "Symmetrical Grid (1)"])
rig_style = st.sidebar.selectbox("Rigging Style", ["Square (0)", "Crescent/Oceanic (1)"])

# --- AERODYNAMIC LOGIC ---
# Symmetrical grid + Crescent rig = Peak Lift and Drag Reduction
is_symmetrical = "Symmetrical" in pattern_symmetry
is_crescent = "Crescent" in rig_style

lift_coefficient = 2.8 if (is_symmetrical and is_crescent) else 1.2
drag_penalty = 0.1 if is_symmetrical else 0.5

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🪁 Aerodynamic Profile")
    st.metric("Lift Coefficient", f"{lift_coefficient} Cl")
    st.metric("Drag Reduction", f"{(1 - drag_penalty) * 100:.0f}%")
    
    if lift_coefficient > 2.0:
        st.success("✅ MARITIME LOCK: Sail is in 'Quetzal' mode. High-speed tacking active.")
    else:
        st.info("⚓ DRIFT MODE: Low-efficiency square sail configuration.")

with col2:
    st.subheader("🧬 Result: The Winged Ship")
    st.write(f"The 'String' is vibrating at **{lift_coefficient} Cl**. The sail has become a 'Wing'.")
    st.write("This technology allows the maritime elite to sail 'since time immemorial'.")
    st.progress(lift_coefficient / 3.0)

st.divider()
st.subheader("🧬 String Theory Final Maritime Conclusion")
st.write("The Feathered Serpent is the **Aerodynamic Brane** of the Southern Maritime Guild.")
st.write("By 'weaving' the sail into a symmetrical grid, they turned [balsa logs](https://www.vocabulary.com/dictionary/balsa%20raft) into [ocean-going wings](https://dash.harvard.edu/bitstreams/7312037d-16e8-6bd4-e053-0100007fdf3b/download).")
