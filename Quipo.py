import streamlit as st

st.set_page_config(page_title="Khipu Mica Insulator", layout="wide")

st.title("🧶 Khipu Node v15: The Mica Dielectric")
st.write("Simulating the high-voltage insulation of the Teotihuacan 'Viking Group' floors.")

# --- SIDEBAR: INSULATION CONTROLS ---
st.sidebar.header("⚡ Insulation Physics")
mica_layer = st.sidebar.checkbox("Mica Sheets Installed (Viking Group)", value=True)
input_voltage = st.sidebar.slider("Applied Resonant Charge (kV)", 0, 500, 200)

# --- DIELECTRIC LOGIC ---
# Mica can withstand ~150 kV/mm. 
dielectric_threshold = 300 if mica_layer else 50
is_insulated = input_voltage < dielectric_threshold

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛡️ Dielectric Barrier")
    st.metric("System Voltage", f"{input_voltage} kV")
    if is_insulated:
        st.success("✅ CIRCUIT STABLE: Mica is preventing electrical grounding.")
    else:
        st.error("🚨 DIELECTRIC BREAKDOWN: Charge is arcing into the earth!")

with col2:
    st.subheader("🧬 Result: The Capacitor Floor")
    if mica_layer:
        st.write("**Material:** Muscovite Mica (Imported from Brazil).")
        st.write("**Function:** Stabilizing the mercury capacitor charge.")
    st.progress(input_voltage / 500)

st.divider()
st.subheader("🧬 String Theory Final Conclusion")
st.write("The mica is the 'Insulating Brane' that keeps the data from being lost.")
st.write("By separating the 'Mercury Liquid' from the 'Earth Ground', the Fish-Men created a permanent **Static Memory** for their guild.")
