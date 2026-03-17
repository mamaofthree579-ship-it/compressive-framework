import streamlit as st

def get_bit(s):
    try:
        return s.split('(')[1].split(')')[0]
    except:
        return '0'

st.set_page_config(page_title="Khipu Hash Processor", layout="wide")

st.title("🧶 Khipu Node v2: The Checksum Processor")
st.write("Simulate 'Top Cord' verification and multi-layered data hashing.")

# --- SIDEBAR: DATA INPUT ---
st.sidebar.header("Pendant Cluster (Data Input)")
nodes_count = st.sidebar.slider("Number of Pendants in Cluster", 1, 10, 5)

# --- GENERATE MOCK DATA NODES ---
cluster_data = []
for i in range(nodes_count):
    val = st.sidebar.number_input(f"Pendant {i+1} Value", value=10 * (i+1))
    cluster_data.append(val)

calculated_sum = sum(cluster_data)

# --- MAIN INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧵 Pendant Data (The Nodes)")
    st.write("Granular records being processed from the regional hub.")
    st.json({"Node_Values": cluster_data})
    
with col2:
    st.subheader("🔼 Top Cord (The Checksum)")
    # Manual entry for the "Top Cord" value to test verification
    top_cord_val = st.number_input("Enter Top Cord Hash Value", value=calculated_sum)
    
    if top_cord_val == calculated_sum:
        st.success(f"✅ HASH MATCH: Data Integrity Verified (Sum: {calculated_sum})")
    else:
        st.error(f"🚨 HASH COLLISION: Data Corruption Detected (Expected {calculated_sum}, found {top_cord_val})")

st.divider()
st.subheader("🧬 Metadata Tagging")
st.write("Apply your 7-bit 'Fish-Man' logic to the Top Cord header.")

# 7-Bit Selection for the Header
mat = st.selectbox("Header Material", ["Cotton (0)", "Wool (1)"])
col = st.selectbox("Header Color", ["Natural (0)", "Dye-Guild Purple (1)"])
# ... other bits ...

header_id = get_bit(mat) + get_bit(col) + "00000" # Simplified for demo

if header_id.startswith("11"):
    st.info("🚨 HEADER TAG: Imperial Verified Hash (Population Y Protocol)")
else:
    st.info("📊 HEADER TAG: Local/Regional Accounting")
