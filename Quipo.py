import streamlit as st

st.set_page_config(page_title="Khipu Magnetic Beacon", layout="wide")

st.title("🧶 Khipu Node v13: The Magnetic Beacon")
st.write("Simulating the magnetic anomalies of Tiwanaku's 'H-Blocks'.")

# --- SIDEBAR: MAGNETIC CALIBRATION ---
st.sidebar.header("🧭 Navigation Probe")
stone_type = st.sidebar.selectbox("Material to Scan", ["Red Sandstone", "Grey Andesite"])
surface_finish = st.sidebar.slider("Surface Precision (Polish %)", 0, 100, 100)

# --- ANOMALY LOGIC ---
if stone_type == "Grey Andesite" and surface_finish > 80:
    anomaly_strength = 270 # Degrees of compass shift
    guild_tag = "🚨 BEACON ACTIVE: Magnetic 'Homing' Signature Detected."
elif stone_type == "Grey Andesite":
    anomaly_strength = 90
    guild_tag = "🌀 TRACE MAGNETIC: Unfinished stone signature."
else:
    anomaly_strength = 0
    guild_tag = "⚪ INERT: Standard geological baseline."

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧭 Compass Reaction")
    st.metric("Magnetic Deflection", f"{anomaly_strength}°")
    # A gauge showing the wildness of the needle
    st.progress(anomaly_strength / 360)

with col2:
    st.subheader("🧬 Population Y Protocol")
    st.info(guild_tag)
    st.write(f"The 'String' is experiencing a **magnetic pull** of {anomaly_strength} degrees.")
    st.write("This signature acts as a 'Homing Beacon' for maritime navigation.")

st.divider()
st.subheader("🧬 Result: The Magnetic Anchor")
st.write("The andesite isn't just stone; it's a 'Magnetic Battery' that stores the orientation of the guild.")
st.write("In String Theory terms, these stones are 'Magnetic Branes' that anchor the maritime database to the Earth's core.")
