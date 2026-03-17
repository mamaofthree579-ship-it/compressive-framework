import streamlit as st

st.set_page_config(page_title="Khipu Binary Decoder", layout="wide")

st.title("🧶 Khipu Node: Algorithmic Metadata Decoder")
st.write("Interpret the 'String Theory' of the Southern Maritime Guild.")

# Sidebar for Knot Parameters (The 7-Bit Array)
st.sidebar.header("Knot Bit-Mapping")

# Defining options with (0) or (1) clearly for extraction
opt_material = st.sidebar.selectbox("Material", ["Cotton (0)", "Wool (1)"])
opt_color = st.sidebar.selectbox("Color Group", ["Natural (0)", "Dye-Guild Purple (1)"])
opt_spin = st.sidebar.selectbox("Spin Direction", ["S-Twist (0)", "Z-Twist (1)"])
opt_ply = st.sidebar.selectbox("Ply Direction", ["S-Ply (0)", "Z-Ply (1)"])
opt_attach = st.sidebar.selectbox("Attachment", ["Recto (0)", "Verso (1)"])
opt_style = st.sidebar.selectbox("Knot Style", ["Single (0)", "Long/Narrative (1)"])
opt_slant = st.sidebar.selectbox("Knot Slant", ["S-Slant (0)", "Z-Slant (1)"])

# Robust Bit Extraction: Split by '(' and take the char before ')'
def get_bit(s):
    return s.split('(')[1].split(')')[0]

# Generate the 7-bit string
binary_string = "".join([
    get_bit(opt_material),
    get_bit(opt_color),
    get_bit(opt_spin),
    get_bit(opt_ply),
    get_bit(opt_attach),
    get_bit(opt_style),
    get_bit(opt_slant)
])

decimal_val = int(binary_string, 2)

# Display the Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Signature")
    st.metric("Binary ID", binary_string)
    st.metric("Decimal Node Index", decimal_val)
    
with col2:
    st.subheader("Guild Interpretation")
    # Using your "Population Y" parameters for triggers
    if binary_string.startswith("11"):
        st.success("🚨 SIGNATURE: Southern Maritime Elite (Purple Dye/Wool)")
    elif binary_string[2] == "1" or binary_string[3] == "1":
        st.info("🛠️ SIGNATURE: Technical/Engineering Guild (Z-Spin/Ply)")
    else:
        st.warning("📊 SIGNATURE: Agricultural/Accounting Data")

st.divider()
st.subheader("String Theory Visualization")
st.write(f"This knot represents **Node {decimal_val}** in the multi-dimensional database.")
st.progress(decimal_val / 127)
