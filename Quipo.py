import streamlit as st

st.set_page_config(page_title="Khipu Quartz Master Clock", layout="wide")

st.title("🧶 Khipu Node v18: The Quartz Master Clock")
st.write("Simulating the 'Piezoelectric Sync' for the 250-layer Pyrite Array.")

# --- SIDEBAR: CLOCK CONTROLS ---
st.sidebar.header("⏱️ System Clock")
pyrite_layers = st.sidebar.slider("Pyrite Layers (Voltage)", 1, 500, 250)
quartz_sync = st.sidebar.checkbox("Quartz Resonator Active", value=True)

# --- CLOCK LOGIC ---
# System voltage needs 250 layers to 'ignite' the obsidian
system_voltage = pyrite_layers * 1.0 # 250 layers = 250 kV
clock_stability = "Locked (Stable)" if quartz_sync else "Drifting (Noise)"
obsidian_state = "IGNITED" if system_voltage >= 250 else "INERT"

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ System Power")
    st.metric("Voltage", f"{system_voltage} kV")
    st.write(f"**Obsidian Status:** {obsidian_state}")
    if obsidian_state == "IGNITED":
        st.success("✅ IGNITION: The volcanic glass is now a semiconductor.")
    else:
        st.warning("📡 LOW POWER: Increase pyrite layers to reach 250 kV.")

with col2:
    st.subheader("⏱️ Timing & Sync")
    st.write(f"**Clock State:** {clock_stability}")
    if quartz_sync:
        st.info("💎 PIEZO-SYNC: Data is pulsing at a fixed frequency.")
    st.progress(system_voltage / 500)

st.divider()
st.subheader("🧬 Result: The Optical Projector")
st.write("At 250 kV, the Obsidian Mirror is 'on'. The Quartz Skull is now focusing the data.")
st.write("In String Theory terms, the Quartz is the **Metronome** that keeps all strings vibrating in phase.")
