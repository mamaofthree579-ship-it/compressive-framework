import streamlit as st

def get_bit(s):
    try:
        return s.split('(')[1].split(')')[0]
    except:
        return '0'

st.set_page_config(page_title="Khipu Blockchain Simulator", layout="wide")

st.title("🧶 Khipu Node v3: The Blockchain Ledger")
st.write("Verifying data integrity across 'Banded Clusters' (Blocks).")

# --- SIDEBAR: BLOCK CONFIG ---
st.sidebar.header("Block Configuration")
block_id = st.sidebar.text_input("Block ID (e.g., Cusco-Delta-01)", "Inkawasi-01")
n_transactions = st.sidebar.slider("Transactions in Block", 1, 10, 3)

# --- TRANSACTION DATA (THE STRINGS) ---
tx_data = []
st.subheader(f"📦 Current Block: {block_id}")
cols = st.columns(n_transactions)

for i in range(n_transactions):
    with cols[i]:
        val = st.number_input(f"String {i+1} (Units)", value=50, key=f"tx_{i}")
        tx_data.append(val)

# THE HASH CALCULATION
current_hash = sum(tx_data)

# --- VERIFICATION INTERFACE ---
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔼 Top Cord (Block Hash)")
    # This simulates the "Imperial Header" we discussed
    imperial_hash = st.number_input("Imperial Hash Value", value=current_hash)
    
    if imperial_hash == current_hash:
        st.success(f"✅ BLOCK VALIDATED: {block_id} is immutable.")
    else:
        st.error(f"🚨 FRAUD DETECTED: Block {block_id} has been tampered with!")

with col_right:
    st.subheader("🧬 Guild Protocol")
    # 7-Bit Tagging for the Block Header
    mat = st.selectbox("Header Material", ["Cotton (0)", "Wool (1)"])
    col = st.selectbox("Header Dye", ["Natural (0)", "Purple Snail Dye (1)"])
    
    header_bits = get_bit(mat) + get_bit(col) + "00000"
    
    if header_bits.startswith("11"):
        st.info("💎 PROTOCOL: Southern Maritime Elite (Population Y)")
    else:
        st.info("🌾 PROTOCOL: Standard Agricultural Registry")

st.divider()
st.write("**Data Insight:** In String Theory terms, this block is a stabilized 'brane' of information. Any change in 'tension' (the numbers) breaks the mathematical harmony of the cluster.")
