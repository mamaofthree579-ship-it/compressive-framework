import streamlit as st

st.set_page_config(page_title="Khipu Global Fiber v30", layout="wide")

st.title("🧶 Khipu Node v30: The Global Whale-Fiber Matrix")
st.write("Simulating Data Latency, 7-Bit Encoding, and Base-10 Transmission across the 'Kelp Highway' Backbone.")

# --- DATA MATRIX CONFIGURATION ---
nodes = {
    "Bering": "Bering Strait (Gateway)",
    "Columbia": "Columbia River (Hub)",
    "Amazon": "Amazon Delta (Server)",
    "Tiwanaku": "Tiwanaku (CPU)",
    "Easter": "Easter Island (Relay)"
}

# Explicit point-to-point matrix mapping (in kilometers)
distance_matrix = {
    ("Bering", "Columbia"): 3800,
    ("Bering", "Amazon"): 12400,
    ("Bering", "Tiwanaku"): 11200,
    ("Bering", "Easter"): 10500,
    ("Columbia", "Amazon"): 8600,
    ("Columbia", "Tiwanaku"): 7900,
    ("Columbia", "Easter"): 7200,
    ("Amazon", "Tiwanaku"): 3100,
    ("Amazon", "Easter"): 7400,
    ("Tiwanaku", "Easter"): 4200,
}

# --- SIDEBAR: NETWORK & ANTHROPOLOGICAL CONTROLS ---
st.sidebar.header("🗺️ Network Topology")
origin_key = st.sidebar.selectbox("Origin Node", list(nodes.keys()), index=0, format_func=lambda x: nodes[x])
dest_key = st.sidebar.selectbox("Destination Node", list(nodes.keys()), index=2, format_func=lambda x: nodes[x])

st.sidebar.header("🪢 7-Bit Cord Encryption")
twist = st.sidebar.radio("Spin/Ply Twist Direction", ["Z-Twist (Standard)", "S-Twist (Encrypted)"])
material = st.sidebar.selectbox("Fiber Medium", ["Whale Baleen (High Tensile)", "Marine Cotton", "Camelid Wool Blend"])
attachment = st.sidebar.radio("Pendant Attachment", ["Verso (Front-to-Back)", "Recto (Back-to-Front)"])

# --- CORE ROUTING LOGIC ---
pair = (origin_key, dest_key)
reverse_pair = (dest_key, origin_key)

if origin_key == dest_key:
    dist = 0
else:
    dist = distance_matrix.get(pair, distance_matrix.get(reverse_pair, 5000))

# Whale migration calculation rules (Base: 5 knots / 9.2 km/h)
base_speed = 9.2 
if "S-Twist" in twist:
    base_speed *= 1.15  # Simulated 15% encoding performance gain via structural compression

latency_days = 0.0 if dist == 0 else dist / (base_speed * 24)

# --- USER INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Connection Status")
    st.metric("Link Distance", f"{dist:,} km")
    st.metric("Packet Latency", f"{latency_days:.1f} Days")
    st.info(f"**Active Cable:** Routing from **{nodes[origin_key]}** to **{nodes[dest_key]}** via {material}.")

with col2:
    st.subheader("🛡️ Protocol & Integrity Verification")
    if origin_key == "Bering" or dest_key == "Bering":
        st.success("✅ GATEWAY OPEN: Whale Bone Alley validation protocol verified.")
    else:
        st.warning("🔒 INTERNAL LAYER-2 TRAFFIC: Bypassing northern maritime firewall.")
    
    # Secure progress tracking
    progress_val = 1.0 if latency_days == 0 else min(1.0, 10.0 / latency_days)
    st.progress(progress_val)

st.divider()

# --- DECI-KHIPU VISUALIZATION LAYER ---
st.subheader("🧮 Result: Physical String State Vector")
st.write(f"The structural fiber array will synchronize at the **{nodes[dest_key]}** terminal node in exactly **{latency_days:.1f} days**.")

# Mathematical translation to Base-10 knots
rounded_days = int(round(latency_days))
hundreds = rounded_days // 100
tens = (rounded_days % 100) // 10
units = rounded_days % 10

st.write("### 🧵 Transmitted Record (Base-10 Knot Architecture)")
if rounded_days == 0:
    st.code("─── (No Latency / Local Loopback)")
else:
    # Build a visual string readout matching real structural archetypes
    khipu_string = "───[Main Primary Cord]───\n"
    if hundreds > 0:
        khipu_string += f"   ├── Hundreds Tier: {hundreds}x Single Knots (●)\n"
    if tens > 0:
        khipu_string += f"   ├── Tens Tier:     {tens}x Single Knots (●)\n"
    if units > 0:
        if units == 1:
            khipu_string += "   └── Units Tier:    1x Figure-Eight Knot (∞)\n"
        else:
            khipu_string += f"   └── Units Tier:    1x Long Knot with {units} turns (▰)\n"
    
    st.code(khipu_string, language="text")
    st.caption(f"Security Hash: {twist[:1]}-{material[:3].upper()}-{attachment[:1]}")
