import streamlit as st

st.set_page_config(page_title="Khipu Binary Decoder", layout="wide")

st.title("🧶 Khipu Node: Algorithmic Metadata Decoder")
st.write("Interpret the 'String Theory' of the Southern Maritime Guild.")

# Sidebar for Knot Parameters (The 7-Bit Array)
st.sidebar.header("Knot Bit-Mapping")
material = st.sidebar.selectbox("Material (Bit 1)", ["Cotton (0)", "Wool (1)"])
color_class = st.sidebar.selectbox("Color Group (Bit 2)", ["Primary/Natural (0)", "Dye-Guild Purple/Red (1)"])
spin = st.sidebar.selectbox("Spin Direction (Bit 3)", ["S-Twist (0)", "Z-Twist (1)"])
ply = st.sidebar.selectbox("Ply Direction (Bit 4)", ["S-Ply (0)", "Z-Ply (1)"])
attachment = st.sidebar.selectbox("Attachment (Bit 5)", ["Recto/Front (0)", "Verso/Back (1)"])
knot_type = st.sidebar.selectbox("Knot Style (Bit 6)", ["Single/Decimal (0)", "Long/Narrative (1)"])
knot_direction = st.sidebar.selectbox("Knot Slant (Bit 7)", ["S-Slant (0)", "Z-Slant (1)"])

# Convert selections to bits
bits = [
    material[8], color_class[16] if "Dye" in color_class else color_class[16], 
    spin[8], ply[8], attachment[6] if "Recto" in attachment else attachment[6],
    knot_type[7] if "Single" in knot_type else knot_type[5],
    knot_direction[8]
]

binary_string = "".join([b for b in bits if b.isdigit()])
decimal_val = int(binary_string, 2)

# Display the Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Signature")
    st.metric("Binary ID", binary_string)
    st.metric("Decimal Node Index", decimal_val)
    
with col2:
    st.subheader("Guild Interpretation")
    if binary_string.startswith("11"):
        st.success("🚨 SIGNATURE DETECTED: Southern Maritime Elite (Purple Dye/Wool)")
    elif "1" in binary_string[2:4]:
        st.info("🛠️ SIGNATURE DETECTED: Technical/Engineering Guild (S/Z Plying)")
    else:
        st.warning("📊 SIGNATURE DETECTED: Agricultural/Accounting Data")

st.divider()
st.subheader("String Theory Visualization")
st.write(f"This knot represents **Node {decimal_val}** in a multi-dimensional database.")
st.progress(decimal_val / 127)

# Theoretical Syllabic Match (Sabine Hyland's Theory)
st.info(f"**Potential Phonetic Tag:** Based on Sabine Hyland's research, this specific bit-cluster could represent a clan identifier or a 'Fish-Man' maritime rank.")
