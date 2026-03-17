import streamlit as st

st.set_page_config(page_title="Khipu Brainwave Interface", layout="wide")

st.title("🧶 Khipu Node v7: The Gamma Interface")
st.write("Simulating Brainwave Entrainment at the 'Oracle' Resonance.")

# --- SIDEBAR: RESONANCE CONTROLS ---
st.sidebar.header("🎚️ Gallery Tuning")
res_freq = st.sidebar.number_input("Structural Resonance (Hz)", value=110)
instrument_freq = st.sidebar.number_input("Instrument Frequency (Hz)", value=150)

# --- ENTRAINMENT LOGIC ---
difference_tone = abs(instrument_freq - res_freq)

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Neural State Analysis")
    st.metric("Resonant Base", f"{res_freq} Hz")
    st.metric("Resulting Difference Tone", f"{difference_tone} Hz")
    
    if difference_tone == 40:
        st.success("🚨 INTERFACE ACTIVE: Gamma Entrainment (40 Hz) Achieved.")
    elif difference_tone == 10:
        st.info("🧘 STATE: Alpha Relaxation (10 Hz) Detected.")
    else:
        st.warning("⚠️ STATE: Asynchronous / Non-Resonant.")

with col2:
    st.subheader("🏛️ Acoustic Engineering")
    if res_freq == 110:
        st.write("**Gallery Status:** Right-Hemisphere Shift (Emotional/Pattern Mode).")
    st.write(f"The 'String' is now tuned to a **{difference_tone} Hz** beat frequency.")

st.divider()
st.subheader("🧬 Result: Database Access Granted")
st.write("In String Theory terms, the listener has 'synced' their internal vibration to the external 'Brane' of the maritime database.")
st.write("Accessing Node Cluster... **[Population Y Signature Detected]**")
