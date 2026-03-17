import streamlit as st

st.set_page_config(page_title="Khipu Whale-Oil Coolant", layout="wide")

st.title("🧶 Khipu Node v27: The Whale-Oil Coolant")
st.write("Simulating the Thermal Stabilization of the 500 kV Obsidian Processor.")

# --- SIDEBAR: COOLANT CONTROLS ---
st.sidebar.header("🐋 Coolant Calibration")
oil_type = st.sidebar.selectbox("Coolant Fluid", ["Water (Corrosive)", "Vegetable Oil (Unstable)", "Spermaceti Oil (High-Precision)"])
voltage_load = st.sidebar.slider("System Voltage (kV)", 0, 500, 500)

# --- THERMAL LOGIC ---
if "Spermaceti" in oil_type:
    cooling_factor = 2.0
    stability_tag = "🚨 THERMAL LOCK: System stabilized. High-voltage arcing suppressed."
else:
    cooling_factor = 0.5
    stability_tag = "⚠️ OVERHEAT WARNING: Insufficient insulation. System failure imminent."

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("❄️ Thermal State")
    st.metric("Applied Load", f"{voltage_load} kV")
    st.write(f"**Cooling Efficiency:** {cooling_factor}x")
    if cooling_factor >= 2.0:
        st.success(stability_tag)
    else:
        st.error(stability_tag)

with col2:
    st.subheader("🧬 Result: The Immortal Core")
    st.write(f"With **{oil_type}**, the 'String' can vibrate indefinitely.")
    st.write("The whale oil acts as the 'Dielectric Brane' for the deep-time database.")
    st.progress(cooling_factor / 2.0)

st.divider()
st.subheader("🧬 Result: The 1-Million-Year Continuity")
st.write("The 'Fish-Men' built a system that [predates human history](https://www.youtube.com/watch?v=rtUad6jqFFM). It cannot be stopped as long as the 'Whale Server' remains in the ocean.")
