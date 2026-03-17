import streamlit as st

st.set_page_config(page_title="Khipu Spondylus Signal", layout="wide")

st.title("🧶 Khipu Node v25: The Spondylus Signal")
st.write("Simulating the 'Red Gold' Optical Barcode of the Pacific Guilds.")

# --- SIDEBAR: SIGNALING INPUT ---
st.sidebar.header("📡 Spondylus 'Ping'")
shell_polish = st.sidebar.slider("Inner Pearl Polish (%)", 0, 100, 95)
ambient_light = st.sidebar.selectbox("Signal Context", ["Sunlight (Day)", "Torchlight (Night)"])

# --- OPTICAL LOGIC ---
# Reflective surfaces can reach distances of several miles
reflection_distance = (shell_polish / 100) * (15 if "Sunlight" in ambient_light else 5)
signal_strength = "High Priority" if shell_polish >= 90 else "Routine"

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🐚 Sea-Mirror Reflection")
    st.metric("Signal Distance", f"{reflection_distance:.1f} Miles")
    st.write(f"**Priority Level:** {signal_strength}")
    if reflection_distance >= 10:
        st.success("✅ OPTICAL LOCK: Ship-to-ship connection established.")
    else:
        st.info("⚓ LOCAL PING: Communication limited to immediate fleet.")

with col2:
    st.subheader("🧬 Guild Status: Red Gold")
    if shell_polish >= 90:
        st.success("💎 CURRENCY-CODE: High-Value Elite Merchant Signature.")
    st.write("The Spondylus interior acts as the 'Mirror' to reflect the Guild's status.")
    st.progress(shell_polish / 100)

st.divider()
st.subheader("🧬 Result: The Luminous Ledger")
st.write(f"The 'String' is vibrating with a **Luminous Frequency** of {shell_polish}%.")
st.write("In String Theory terms, the Spondylus is the **Reflective Brane** that allows for light-speed data exchange.")
