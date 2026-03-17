import streamlit as st

st.set_page_config(page_title="Khipu Smart Contract", layout="wide")

st.title("🧶 Khipu Node v4: The Smart Contract Processor")
st.write("Simulating the Inkawasi 'Fixed-Value' Taxation Algorithm.")

# --- SIDEBAR: DEPOSIT PARAMETERS ---
st.sidebar.header("📥 Deposit Input")
commodity = st.sidebar.selectbox("Select Commodity", ["Chili Peppers", "Peanuts", "Black Beans"])
deposit_val = st.sidebar.number_input("Enter Total Units (Deposit 'a')", value=100)

# --- THE SMART CONTRACT LOGIC ---
# Standard Inkawasi Fixed Values: 10, 15, 47, 208
if commodity == "Chili Peppers":
    fixed_tax = 15 # The 'b' value for high-value goods
elif commodity == "Peanuts":
    fixed_tax = 10 # The 'b' value for standard goods
else:
    fixed_tax = 47 # Example for larger bulk black beans

remainder = deposit_val - fixed_tax # The 'c' value

# --- INTERFACE ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📦 Total Deposit (a)")
    st.metric("Total Knots", deposit_val)

with col2:
    st.subheader("⚖️ Smart Tax (b)")
    st.metric("Fixed Knot Value", fixed_tax, delta="-Fixed Deduction")
    st.info(f"**Protocol:** Inkawasi Fixed {fixed_tax} Logic")

with col3:
    st.subheader("🏦 Net Storage (c)")
    st.metric("Remainder for Basin", remainder)

st.divider()
st.subheader("🧬 Transaction Hash (a = b + c)")
if deposit_val == (fixed_tax + remainder):
    st.success(f"✅ CONTRACT EXECUTED: {deposit_val} = {fixed_tax} + {remainder}")
    st.write("**Data Integrity:** The string tension is balanced. Ledger is immutable.")
else:
    st.error("🚨 CONTRACT VOID: Mathematical Imbalance Detected!")

st.write("**Note:** This replicates the Inkawasi formulaic arrangement where tax was a pre-coded deduction.")
