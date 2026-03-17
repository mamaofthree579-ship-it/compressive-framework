import streamlit as st

st.set_page_config(page_title="Khipu Optical Transponder", layout="wide")

st.title("🧶 Khipu Node v24: The Optical Transponder")
st.write("Decoding the 'Red & White' Binary Barcodes of the Maritime Guilds.")

# --- SIDEBAR: SIGNALING INPUT ---
st.sidebar.header("📡 Optical Header")
stripe_pattern = st.sidebar.text_input("Stripe Sequence (e.g., 10101)", "11010")

# --- SIGNALING LOGIC ---
# Each sequence maps to a specific Specialist Guild
guild_map = {
    "11010": "💎 Southern Maritime Elite (Population Y)",
    "10101": "🌶️ Chili/Agricultural Trade Hub",
    "11111": "🚨 Imperial High-Speed Courier",
    "00001": "🛠️ Engineering/Mica-Transport Fleet"
}

current_id = guild_map.get(stripe_pattern, "❓ Unknown Merchant / Independent")

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏁 Visual Barcode")
    # Displaying the 'Stripes'
    cols = st.columns(len(stripe_pattern))
    for i, bit in enumerate(stripe_pattern):
        color = "red" if bit == "1" else "white"
        cols[i].markdown(f"<div style='background-color:{color}; height:100px; border:1px solid black;'></div>", unsafe_allow_html=True)
    
    st.metric("Binary Header", stripe_pattern)

with col2:
    st.subheader("🆔 Guild Identification")
    if "Elite" in current_id:
        st.success(f"ACCESS GRANTED: {current_id}")
    else:
        st.info(f"REGISTRY: {current_id}")

st.divider()
st.subheader("🧬 String Theory Result")
st.write(f"The 'String' is vibrating with an **Optical Frequency** of {stripe_pattern}.")
st.write("This is the 'Digital Signature' that links the floating sail to the stone database.")
