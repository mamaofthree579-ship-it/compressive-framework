import streamlit as st

st.set_page_config(page_title="Khipu Whale Packet Decoder", layout="wide")

st.title("🧶 Khipu Node v28: The Whale Song Packet")
st.write("Decoding the 'Zipfian' Data Packets of the South Pacific Whale Server.")

# --- SIDEBAR: PACKET PARAMETERS ---
st.sidebar.header("📡 Packet Configuration")
song_length = st.sidebar.slider("Song Duration (min)", 1, 30, 20)
complexity = st.sidebar.selectbox("Packet Logic", ["Simple Repetition", "Zipfian/Linguistic (1)"])

# --- DATA LOGIC ---
# Zipfian distribution allows for human-language levels of information density
if "Zipfian" in complexity:
    data_density = 1000 # High-density bits per song
    st_tag = "🚨 DATA LOCK: Song matches Human-Language statistical properties."
else:
    data_density = 10
    st_tag = "⚠️ LOW SIGNAL: Simple repetition provides minimal data transfer."

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎵 Acoustic Bitrate")
    st.metric("Song Length", f"{song_length} min")
    st.metric("Calculated Information Density", f"{data_density} pts")
    if data_density >= 1000:
        st.success(st_tag)
    else:
        st.warning(st_tag)

with col2:
    st.subheader("🧬 Guild Status: The Deep-Sea Archivist")
    st.write(f"The 'String' is vibrating with a **Linguistic Symmetry** of {data_density/1000:.1f}.")
    st.info("Transmission distance estimated at 8,000 km across the South Pacific.")
    st.progress(data_density / 1000)

st.divider()
st.subheader("🧬 Result: The Unbroken Ledger")
st.write(f"The 'Whale Server' has transmitted **{song_length * data_density:,} bits** in this session.")
st.write("This is the 'Operating System' of the ocean. It has been running for 1 million years.")
