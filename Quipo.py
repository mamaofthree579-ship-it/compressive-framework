import streamlit as st

st.set_page_config(page_title="Khipu Pyrite Transistor", layout="wide")

st.title("🧶 Khipu Node v16: The Pyrite Transistor")
st.write("Simulating the 'Voltage-Gated' behavior of the Teotihuacan spheres.")

# --- SIDEBAR: TRANSISTOR CONTROLS ---
st.sidebar.header("🌑 Sphere Properties")
sphere_count = st.sidebar.slider("Number of Pyrite Spheres", 1, 500, 100)
system_voltage = st.sidebar.slider("System Voltage (kV)", 0, 500, 300)

# --- SEMICONDUCTOR LOGIC ---
# Pyrite conducts at high voltage. Threshold approx 250 kV.
conduction_threshold = 250
is_conducting = system_voltage >= conduction_threshold

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌑 Semiconductor State")
    st.metric("Applied Voltage", f"{system_voltage} kV")
    if is_conducting:
        st.success("⚡ CONDUCTING: The pyrite spheres are now active 'transistors'.")
    else:
        st.warning("🛡️ INSULATING: Voltage is below the semiconductor threshold.")

with col2:
    st.subheader("🧬 Result: The Solid-State Array")
    st.write(f"The **{sphere_count} pyrite spheres** are acting as 'Switches' in the database.")
    st.write("At high voltage, they allow the 'String' data to flow between the mercury pools.")
    st.progress(system_voltage / 500)

st.divider()
st.subheader("🧬 String Theory Final Synthesis")
st.write("The pyrite spheres are the 'Nodes' that can be switched on or off by the maritime elite.")
st.write("By combining **Mica Insulation**, **Mercury Capacitors**, and **Pyrite Transistors**, they built a **Solid-State Neural Computer** under the earth.")
