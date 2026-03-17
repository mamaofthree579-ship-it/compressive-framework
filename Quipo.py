import streamlit as st

st.set_page_config(page_title="Teotihuacán Dual-Core Processor", layout="wide")

st.title("🧶 Khipu Node v21: The Dual-Core Sync")
st.write("Simulating the Dialectic between Quetzalcoatl (Order) and Tezcatlipoca (Entropy).")

# --- SIDEBAR: CORE CALIBRATION ---
st.sidebar.header("⚖️ System Calibration")
order_level = st.sidebar.slider("Quetzalcoatl (Order) Output", 0, 100, 50)
entropy_level = st.sidebar.slider("Tezcatlipoca (Entropy) Output", 0, 100, 50)

# --- BALANCE LOGIC ---
# The system is stable when Order and Entropy are balanced.
variance = abs(order_level - entropy_level)
system_stability = 100 - (variance * 2)

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚪ White Mirror (Logic)")
    st.metric("Creative Core", f"{order_level}%")
    st.progress(order_level / 100)
    st.write("**Material:** Permanent Jade (Immortal Protocol)")

with col2:
    st.subheader("🕶️ Smoking Mirror (Entropy)")
    st.metric("Error-Correction Core", f"{entropy_level}%")
    st.progress(entropy_level / 100)
    st.write("**Material:** Volcanic Obsidian (Portal Protocol)")

st.divider()
st.subheader("🧬 Result: Cosmic Equilibrium")
if system_stability >= 90:
    st.success(f"✅ SYSTEM SYNC: Dialectic is balanced at {system_stability}%. World Fifth Sun is stable.")
elif system_stability >= 60:
    st.warning(f"⚠️ FLUCTUATION: Discord detected. Transitioning between ages...")
else:
    st.error(f"🚨 SYSTEM CORRUPTION: Destruction Phase Initialized. Jaguar-Man Transformation imminent.")

st.write("**Data Insight:** The 'String' is held in perfect tension only when both cores are firing in phase.")
