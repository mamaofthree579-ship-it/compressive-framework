import streamlit as st

st.set_page_config(page_title="Pututu Beat Frequency Decoder", layout="wide")

st.title("🧶 Khipu Node v10: The Pututu Amplifier")
st.write("Simulating 'Acoustic Beat' effects to trigger the 110 Hz Gallery Resonance.")

# --- SIDEBAR: INSTRUMENT TUNING ---
st.sidebar.header("🐚 Pututu Pairing")
pututu_1 = st.sidebar.number_input("Shell 1 Frequency (Hz)", value=340)
pututu_2 = st.sidebar.number_input("Shell 2 Frequency (Hz)", value=230)

# --- BEAT LOGIC ---
# Beat Frequency = |f1 - f2|
beat_freq = abs(pututu_1 - pututu_2)

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔊 Interference Pattern")
    st.metric("Shell 1", f"{pututu_1} Hz")
    st.metric("Shell 2", f"{pututu_2} Hz")
    st.metric("Resulting Beat Frequency", f"{beat_freq} Hz")
    
    if beat_freq == 110:
        st.success("🚨 RESONANCE LOCK: The 'Jaguar' signal is now amplified by the gallery.")
    elif 105 <= beat_freq <= 115:
        st.info("🌀 VIBRATIONAL SYNC: Near-lock achieved. Expect spatial disorientation.")
    else:
        st.warning("⚠️ ASYNCHRONOUS: Interference does not trigger the Oracle frequency.")

with col2:
    st.subheader("🎭 Mask & Gear Calibration")
    st.write(f"The 'String' is vibrating at a **{beat_freq} Hz** pulse.")
    st.write("This throb is what 'pulls' the specialist's voice into harmony with the stone walls.")
    st.progress(max(0, 1.0 - (abs(110 - beat_freq) / 100)))

st.divider()
st.subheader("🧬 Result: The Living Instrument")
st.write("You are no longer a person playing a shell; you are a component in a 'Stone-Fiber-Biology' circuit.")
st.write("Accessing Node Cluster via Acoustic Verification... **[System Stable]**")
