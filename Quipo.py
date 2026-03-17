import streamlit as st

st.set_page_config(page_title="Khipu Obsidian Interface", layout="wide")

st.title("🧶 Khipu Node v17: The Obsidian Monitor")
st.write("Simulating high-frequency data display using the 500 kV Pyrite Array.")

# --- SIDEBAR: INTERFACE CONTROLS ---
st.sidebar.header("🖥️ Interface Hardware")
pyrite_layers = st.sidebar.slider("Pyrite Layers (Resistance)", 1, 500, 500)
obsidian_type = st.sidebar.selectbox("Display Material", ["Raw Obsidian", "Polished Mirror (Tezcatlipoca)"])

# --- OPTICAL LOGIC ---
# System voltage scales with pyrite layers
max_voltage = pyrite_layers * 1.0 # 500 layers = 500 kV
obsidian_conductivity = 1.0 if max_voltage >= 250 else 0.1
data_clarity = "High-Resolution" if (obsidian_conductivity > 0.5 and "Mirror" in obsidian_type) else "Distorted/Noise"

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖥️ Display Output")
    st.metric("System Voltage", f"{max_voltage} kV")
    st.write(f"**Data Clarity:** {data_clarity}")
    if data_clarity == "High-Resolution":
        st.success("✅ MONITOR ACTIVE: The 'Smoking Mirror' is displaying the Quipu Node.")
    else:
        st.warning("📡 NO SIGNAL: Insufficient voltage to ignite the obsidian semiconductor.")

with col2:
    st.subheader("🧬 Result: The Optical Gateway")
    st.write(f"With **{pyrite_layers} layers**, the system can sustain the charge needed for fiber-optic transmission.")
    st.progress(max_voltage / 500)

st.divider()
st.subheader("🧬 String Theory Final Visualization")
st.write("The Obsidian Mirror is the 'Screen' that lets the maritime elite 'see' the vibrations of the Strings.")
st.write("At 500 kV, the obsidian's 'Positive Holes' allow it to act as a **Photon-to-Electron Transceiver**.")
