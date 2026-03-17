import streamlit as st

st.set_page_config(page_title="Khipu Planetary Synchronizer", layout="wide")

st.title("🧶 Khipu Node v11: The Planetary Pulse")
st.write("Simulating the harmony between 'Stamping Floors' and the 7.83 Hz Schumann Resonance.")

# --- SIDEBAR: RHYTHMIC INPUT ---
st.sidebar.header("🦶 Stamping Rhythm")
steps_per_minute = st.sidebar.slider("Steps per Minute", 60, 600, 470)
stamping_freq = steps_per_minute / 60 # Convert BPM to Hz

earth_heartbeat = 7.83 # Schumann Resonance fundamental

# --- HARMONIC LOGIC ---
resonance_match = 100 - (abs(stamping_freq - earth_heartbeat) * 10)

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🥁 Rhythmic Frequency")
    st.metric("Stamping Speed", f"{stamping_freq:.2f} Hz")
    st.metric("Resonance Match", f"{max(0, resonance_match):.1f}%")
    
    if 95 <= resonance_match <= 105:
        st.success("🌍 EARTH SYNC: The city is vibrating in harmony with the Schumann Resonance.")
    elif resonance_match > 70:
        st.info("🌀 VIBRATIONAL BRIDGE: Standing waves are forming in the foundations.")
    else:
        st.warning("⚠️ ASYNCHRONOUS: The 'String' is out of tune with the planetary core.")

with col2:
    st.subheader("🏗️ Shicra Stability")
    st.write(f"At {stamping_freq:.2f} Hz, the stone-filled fiber bags are dissipating energy.")
    st.write("This 'tuning' prevents the stone joints from locking, allowing the city to 'ride' the seismic waves.")
    st.progress(max(0, min(1.0, resonance_match / 100)))

st.divider()
st.subheader("🧬 Result: The Terrestrial Map")
st.write("The city isn't just sitting on the ground; it is 'plugged into' the planet.")
st.write("In String Theory terms, the city is a 'Macro-Node' vibrating on the Earth's primary Brane.")
