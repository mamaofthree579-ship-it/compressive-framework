import streamlit as st

st.set_page_config(page_title="Khipu Mercury Capacitor", layout="wide")

st.title("🧶 Khipu Node v14: The Mercury Capacitor")
st.write("Simulating the 'Liquid Conductor' synergy beneath the Andesite hubs.")

# --- SIDEBAR: CAPACITOR CONTROLS ---
st.sidebar.header("⚗️ Sub-Basement Analysis")
mercury_present = st.sidebar.checkbox("Liquid Mercury Layer Detected", value=True)
stone_polish = st.sidebar.slider("Andesite Precision (%)", 0, 100, 65)

# --- CIRCUIT LOGIC ---
base_deflection = 90 if stone_polish >= 65 else 0
# Mercury acts as a 'Multiplier' (Q-Factor)
multiplier = 2.5 if mercury_present else 1.0
final_signal = min(360, base_deflection * multiplier)

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Signal Amplification")
    st.metric("Base Stone Signal", f"{base_deflection}°")
    st.metric("Final Beacon Power", f"{final_signal:.0f}°", delta="Mercury Boosted" if mercury_present else None)
    
    if final_signal >= 225:
        st.success("🚨 SIGNAL LOCK: 'Fish-Man' Homing Beacon is at Max Power.")
    else:
        st.info("📡 TRACE SIGNAL: Beacon is in 'Low-Power' standby mode.")

with col2:
    st.subheader("🧬 Result: The Liquid Circuit")
    st.write(f"With the **Mercury Layer**, your {stone_polish}% polish has been amplified.")
    st.write("The mercury acts as the 'Ground' for the magnetic brane, stabilizing the data.")
    st.progress(final_signal / 360)

st.divider()
st.subheader("🧬 String Theory Conclusion")
st.write("The mercury isn't just 'liquid'; it's a **Vibrational Fluid Brane**.")
st.write("By vibrating the stone above the liquid metal, they created a 'Trans-Oceanic Wi-Fi' for the maritime specialists.")
