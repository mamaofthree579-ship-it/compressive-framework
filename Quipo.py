import streamlit as st

st.set_page_config(page_title="Mayan System Runtime", layout="wide")

st.title("🧶 Khipu Node v19: The Long Count Runtime")
st.write("Simulating the 5,126-year cycle of the ancient Teotihuacán CPU.")

# --- SIDEBAR: SYSTEM CYCLES ---
st.sidebar.header("⏳ Runtime Status")
current_baktun = st.sidebar.slider("Current Baktun", 1, 13, 13)
system_reboot = st.sidebar.checkbox("Trigger 13th Baktun Reboot", value=True)

# --- RUNTIME LOGIC ---
days_per_baktun = 144000
total_runtime_days = current_baktun * days_per_baktun
system_stability = "Optimal" if not system_reboot else "New Cycle Initialized"

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🕰️ Long Count Clock")
    st.metric("Total System Runtime", f"{total_runtime_days:,} Days")
    st.write(f"**Baktun Cycle:** {current_baktun}/13")
    if current_baktun == 13 and system_reboot:
        st.success("✨ REBOOT COMPLETE: System initialized for the 14th Baktun.")

with col2:
    st.subheader("🧩 Processor Status")
    st.write(f"**Stability:** {system_stability}")
    st.info("The 250 kV Pyrite Array is holding. Quartz Sync is locked to the Venus Cycle.")
    st.progress(current_baktun / 13)

st.divider()
st.subheader("🧬 Result: The grand, ever-turning wheel.")
st.write("The Long Count isn't just time; it's the **Operation Manual** for the planet's data.")
st.write("In String Theory terms, the end of a Baktun is the moment the 'String' reaches a zero-point and begins a new vibration.")
