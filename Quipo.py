import streamlit as st

st.set_page_config(page_title="Khipu Cymatic Processor", layout="wide")

st.title("🧶 Khipu Node v6: The Cymatic Master Clock")
st.write("Visualizing the 'Geometric Blueprint' of the 432 Hz Maritime Protocol.")

# --- SIDEBAR: FREQUENCY INPUT ---
st.sidebar.header("🎛️ Master Frequency")
freq_val = st.sidebar.number_input("Input Frequency (Hz)", value=432)

# --- CYMATIC PATTERN LOGIC ---
# Simulating the 'Complexity' of a pattern based on Fibonacci ratios
if freq_val == 432:
    complexity = "Perfect Rosette (Fibonacci Harmonized)"
    guild_tag = "🚨 MASTER CLOCK DETECTED: Southern Maritime Protocol"
elif freq_val % 8 == 0: # 432 is a multiple of 8
    complexity = "Symmetrical Mandala"
    guild_tag = "📊 Regional Standard Frequency"
else:
    complexity = "Chaotic / Dissonant Pattern"
    guild_tag = "⚠️ Non-Standard Vibration"

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📐 Geometric Blueprint")
    st.metric("Frequency ID", f"{freq_val} Hz")
    st.write(f"**Cymatic Signature:** {complexity}")
    # Progress bar as a 'stability' indicator
    st.progress(1.0 if freq_val == 432 else 0.5)

with col2:
    st.subheader("🏛️ Architectural Alignment")
    st.info(guild_tag)
    st.write("This frequency matches the acoustic filtering found in ancient 'Fish-Man' galleries.")

st.divider()
st.subheader("🧬 Result: Permanent Acoustic Record")
st.write(f"The 'String' is vibrating at **{freq_val} Hz**. This creates a stable **Cymatic Rosette** that can be carved into stone or woven into double-cloth.")
st.write("In String Theory terms, this is the 'resonant frequency' that holds the guild's physical and data structures together.")
