import streamlit as st

st.set_page_config(page_title="Jaguar-Man Acoustic Transformer", layout="wide")

st.title("🐆 Khipu Node v9: The Jaguar-Man Transformer")
st.write("Simulating the Acoustic Coupling between Vocal Frequency and Gallery Resonance.")

# --- SIDEBAR: VOCAL & GALLERY TUNING ---
st.sidebar.header("🐾 Totem Tuning")
vocal_freq = st.sidebar.slider("Vocal/Growl Frequency (Hz)", 80, 150, 110)
gallery_resonance = 110 # The Fixed 'Oracle' Frequency

# --- TRANSFORMATION LOGIC ---
# Resonance 'Q Factor' - how close are you to the 110Hz lock?
resonance_diff = abs(vocal_freq - gallery_resonance)
coupling_strength = max(0, 100 - (resonance_diff * 10))

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔊 Acoustic Coupling")
    st.metric("Vocal Pitch", f"{vocal_freq} Hz")
    st.metric("Coupling Strength", f"{coupling_strength}%")
    
    if coupling_strength >= 90:
        st.success("🐆 TRANSFORMATION ACTIVE: Resonance Lock achieved at 110 Hz.")
    elif coupling_strength >= 50:
        st.info("🌀 VIBRATIONAL SHIFT: Entering the 'Jaguar-Man' state.")
    else:
        st.warning("⚠️ ASYNCHRONOUS: Frequency does not match the 'Oracle' chamber.")

with col2:
    st.subheader("🧬 Guild Role: Predator-Specialist")
    if coupling_strength >= 90:
        st.write("**Access Level:** High-Level Guardian of the Maritime Trade Network.")
        st.write("**Ability:** Enhanced spatial orientation and 'phantom' perception.")
    st.progress(coupling_strength / 100)

st.divider()
st.subheader("🧬 Result: The Bio-Acoustic Fossil")
st.write(f"The 'String' is vibrating at **{vocal_freq} Hz**. The stone walls are reflecting the energy back into your cochlea, inducing Gamma entrainment.")
st.write("In String Theory terms, the 'Jaguar' is a specific vibrational mode of the maritime elite's biological shell.")
