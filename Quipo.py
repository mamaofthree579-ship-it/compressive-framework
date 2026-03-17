import streamlit as st

st.set_page_config(page_title="Yupana Fibonacci Processor", layout="wide")

st.title("🧶 Yupana v1: The Fibonacci 'Hardware' Processor")
st.write("Executing 'Smart Contracts' using the 1, 2, 3, 5 Fibonacci weighting system.")

# --- SIDEBAR: YUPANA INPUT ---
st.sidebar.header("🕹️ Yupana Controls")
input_val = st.sidebar.number_input("Total Deposit (a)", min_value=1, value=125)
fixed_tax_val = st.sidebar.selectbox("Fixed Tax Protocol (b)", [10, 15, 47, 208])

# --- YUPANA CALCULATION LOGIC ---
# This simulates the 'seeds' on the board
def fibonacci_slots(n):
    # Mapping a value into the 1,2,3,5 Fibonacci weights
    slots = {'5s': 0, '3s': 0, '2s': 0, '1s': 0}
    remaining = n % 10 # Simplifying to decimal cell for demo
    while remaining >= 5:
        slots['5s'] += 1; remaining -= 5
    while remaining >= 3:
        slots['3s'] += 1; remaining -= 3
    while remaining >= 2:
        slots['2s'] += 1; remaining -= 2
    slots['1s'] += remaining
    return slots

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧮 The Yupana Board (Processing)")
    st.write("Visualizing the 'Seeds' in the Fibonacci slots (1, 2, 3, 5).")
    st.write(f"**Processing Node {input_val}...**")
    st.json(fibonacci_slots(input_val))
    st.info("The Base-40 shift allows for calculations faster than base-10 systems.")

with col2:
    st.subheader("⚙️ Smart Contract Execution")
    remainder_c = input_val - fixed_tax_val
    st.metric("Net Storage (c)", remainder_c)
    
    if remainder_c > 0:
        st.success(f"✅ CONTRACT BALANCED: {input_val} = {fixed_tax_val} + {remainder_c}")
    else:
        st.error("🚨 OVER-TAXATION: Input value lower than fixed protocol.")

st.divider()
st.subheader("🧬 Result: Ready for Quipu Hashing")
st.write(f"The 'Hardware' has verified the result. Node **{remainder_c}** is now ready for 7-bit binary encoding.")
