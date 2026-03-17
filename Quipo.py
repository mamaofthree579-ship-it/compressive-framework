import streamlit as st

st.set_page_config(page_title="Khipu Equinox Synchronizer", layout="wide")

st.title("🧶 Khipu Node v12: The Equinox Aperture")
st.write("Simulating the Peak Electromagnetic Season at the Chankillo Towers.")

# --- SIDEBAR: SEASONAL CONTROLS ---
st.sidebar.header("🗓️ Solar Alignment")
is_equinox = st.sidebar.checkbox("Equinox Alignment (Towers 6 & 7)", value=True)
solar_intensity = st.sidebar.slider("Solar Energy Balance (%)", 0, 100, 100 if is_equinox else 50)

schumann_base = 7.83 # The Earth's Heartbeat

# --- RESONANCE LOGIC ---
# During equinox, the 'Q Factor' (amplification) increases due to solar balance
amplification = 2.5 if is_equinox else 1.0
effective_pulse = schumann_base * (solar_intensity / 100) * amplification

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("☀️ Solar-Acoustic Gate")
    st.metric("Tower Alignment", "Active (6/7)" if is_equinox else "Passive")
    st.metric("Effective Pulse Strength", f"{effective_pulse:.2f} Hz")
    
    if is_equinox and effective_pulse > 15:
        st.success("🔥 APERTURE OPEN: Maximum Electromagnetic Coupling achieved.")
    else:
        st.info("🌑 SEEDING MODE: Accumulating energy for the next seasonal shift.")

with col2:
    st.subheader("🏔️ Akapana Hydraulic Sync")
    if is_equinox:
        st.write("**Drainage Status:** Water thundering through the pyramid tiers.")
    st.progress(min(1.0, effective_pulse / 20))
    st.write("The stone foundation is 'riding' the planetary wave.")

st.divider()
st.subheader("🧬 Result: The Global Node")
st.write(f"The 'String' is vibrating at its peak semiannual amplitude.")
st.write("In String Theory terms, the Equinox is the moment when the 'Beringian' and 'Amazonian' Branes reach a point of perfect interference.")
