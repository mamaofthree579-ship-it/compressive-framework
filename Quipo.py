import streamlit as st

st.set_page_config(page_title="Mayan Deep-Time Ledger", layout="wide")

st.title("🧶 Khipu Node v20: The Alautun Deep-Time Ledger")
st.write("Simulating the 63-million-year tracking capacity of the Mayan 'Master Clock'.")

# --- SIDEBAR: DEEP-TIME CYCLES ---
st.sidebar.header("⏳ Geological Scaling")
time_span = st.sidebar.selectbox("Select Tracking Scale", ["Baktun (394 yrs)", "Calabtun (158k yrs)", "Alautun (63m yrs)"])

# --- RUNTIME LOGIC ---
if "Baktun" in time_span:
    days = 144000
elif "Calabtun" in time_span:
    days = 57600000
else:
    days = 23040000000 # The Alautun

years = days / 365.25

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🕰️ Deep-Time Clock")
    st.metric("Total Cycle Days", f"{days:,}")
    st.metric("Equivalent Earth Years", f"{years:,.0f}")
    st.info(f"**Status:** Tracking data at the {time_span} resolution.")

with col2:
    st.subheader("🖥️ Obsidian Terminal Status")
    st.write("**Interface:** Smoking Mirror (Tezcatlipoca Protocol)")
    st.write("**Sensor State:** Active - Recording Thermal and Seismic Data.")
    # Visualizing the scale compared to a 100-million year brane
    st.progress(min(1.0, years / 100000000))

st.divider()
st.subheader("🧬 Result: The Unstoppable Data Stream")
st.write(f"The 'String' has been vibrating for **{years:,.0f} years**. It cannot be stopped.")
st.write("In String Theory terms, the Alautun is the 'Macro-Vibration' that governs the evolution of the entire planetary Brane.")
