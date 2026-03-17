import streamlit as st

def get_bit(s):
    """
    Extracts the '0' or '1' from strings like 'Cotton (0)' or 'Wool (1)'.
    This is the core 'Metadata Tag' extractor.
    """
    try:
        # Splits the string at '(' and takes the character right after it
        return s.split('(')[1][0]
    except (IndexError, TypeError):
        return '0'

# --- PAGE SETUP ---
st.set_page_config(page_title="Khipu Node Decoder", layout="wide")

st.title("🧶 Khipu Node: Maritime Guild Database")
st.write("A 'String Theory' interface for interpreting the Southern Maritime Guild's 7-bit binary array.")

# --- SIDEBAR: THE 7-BIT PARAMETERS ---
st.sidebar.header("Knot Parameters (The 7 Bits)")
st.sidebar.info("Select parameters to generate a unique Node ID.")

opt_material = st.sidebar.selectbox("Material (Bit 1)", ["Cotton (0)", "Wool (1)"])
opt_color = st.sidebar.selectbox("Color Group (Bit 2)", ["Natural (0)", "Dye-Guild / Meta-Tag (1)"])
opt_spin = st.sidebar.selectbox("Spin Direction (Bit 3)", ["S-Twist (0)", "Z-Twist (1)"])
opt_ply = st.sidebar.selectbox("Ply Direction (Bit 4)", ["S-Ply (0)", "Z-Ply (1)"])
opt_attach = st.sidebar.selectbox("Attachment (Bit 5)", ["Recto (0)", "Verso (1)"])
opt_style = st.sidebar.selectbox("Knot Style (Bit 6)", ["Single (0)", "Long/Narrative (1)"])
opt_slant = st.sidebar.selectbox("Knot Slant (Bit 7)", ["S-Slant (0)", "Z-Slant (1)"])

# --- PROCESSING THE BINARY ARRAY ---
binary_string = "".join([
    get_bit(opt_material),
    get_bit(opt_color),
    get_bit(opt_spin),
    get_bit(opt_ply),
    get_bit(opt_attach),
    get_bit(opt_style),
    get_bit(opt_slant)
])

# Convert the 7-bit binary string into a Decimal Node index
decimal_val = int(binary_string, 2)

# --- MAIN DISPLAY ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Physical Node Data")
    st.metric("Binary ID", binary_string)
    st.metric("Decimal Node Index", decimal_val)
    # Visualizing the position in the 128-character set
    st.progress(decimal_val / 127)

with col2:
    st.subheader("📜 Metadata Interpretation")
    
    # Logic for your specific Node Discoveries
    if decimal_val == 113:
        st.success("🚨 SIGNATURE: Elite Maritime Trade (Node 113)")
        st.write("**Analysis:** High-value exchange. Matches parameters for the Southern Maritime Guild specialists.")
    
    elif decimal_val == 84:
        st.error("🌶️ DATA TAG: Chili Pepper Tax Record (Node 84)")
        st.write("**Analysis:** Specific Inkawasi Cache barcode. Identifies taxed chili pepper stock.")
    
    elif decimal_val == 26:
        st.warning("🥜 DATA TAG: Peanut Storage (Node 26)")
        st.write("**Analysis:** Logistical marker for regional peanut storage bins.")
    
    elif binary_string.startswith("11"):
        st.info("💎 SIGNATURE: General Elite / Purple Dye Guild Pattern")
        st.write("**Analysis:** Uses 'Meta-Tag' materials reserved for the Fish-Man lineage.")
        
    else:
        st.write("📊 **Standard Accounting Node:** General commodity or labor data.")

st.divider()
st.subheader("🧬 String Theory Correlation")
st.write(f"This knot represents **Node {decimal_val}** in a multi-dimensional topological database.")
st.write("According to your theory, this isn't just a number—it's a vibrational coordinate in a 'string' of maritime data.")
