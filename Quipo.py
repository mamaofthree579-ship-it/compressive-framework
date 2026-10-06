import streamlit as st
import math

st.set_page_config(page_title="Khipu Quantum Matrix v32", layout="wide")

st.title("🧶 Khipu Node v32: Chromatographic Matrix & Parity Engine")
st.write("Simulating S/Z Markedness, Color Channel Multiplexing, and Subsidiary Cord Error-Correction.")

# --- CORE ROUTING ARRAYS ---
nodes = {
    "Bering": "Bering Strait (Gateway)",
    "Columbia": "Columbia River (Hub)",
    "Amazon": "Amazon Delta (Server)",
    "Tiwanaku": "Tiwanaku (CPU)",
    "Easter": "Easter Island (Relay)"
}

distance_matrix = {
    ("Bering", "Columbia"): 3800, ("Bering", "Amazon"): 12400, ("Bering", "Tiwanaku"): 11200, ("Bering", "Easter"): 10500,
    ("Columbia", "Amazon"): 8600, ("Columbia", "Tiwanaku"): 7900, ("Columbia", "Easter"): 7200,
    ("Amazon", "Tiwanaku"): 3100, ("Amazon", "Easter"): 7400, ("Tiwanaku", "Easter"): 4200,
}

# --- SIDEBAR CONTROL MATRIX ---
st.sidebar.header("🗺️ Network Topology")
origin_key = st.sidebar.selectbox("Origin Node", list(nodes.keys()), index=0, format_func=lambda x: nodes[x])
dest_key = st.sidebar.selectbox("Destination Node", list(nodes.keys()), index=2, format_func=lambda x: nodes[x])

st.sidebar.header("🪢 Twist & Attachment Engine")
twist = st.sidebar.radio("Structural Axis (Urton Markedness)", ["Z-Twist (Default/Linear)", "S-Twist (Marked/Compressed)"])
attachment = st.sidebar.radio("Hitch Geometry", ["Recto (Parallel Pack)", "Verso (Orthogonal Resist)"])
material = st.sidebar.selectbox("Fiber Material Vector", ["Whale Baleen (Rigid)", "Marine Cotton (Standard)", "Camelid Wool (Elastic)"])

st.sidebar.header("🎨 Chromatographic Channels")
color_profile = st.sidebar.selectbox("Cord Color Schema", ["Solid Natural (Single Carrier)", "Bi-Chrome Barber-Pole (2-Channel MUX)", "Tri-Color Mottled (3-Channel MUX)"])

st.sidebar.header("🌿 Error Correction Layer")
subsidiaries = st.sidebar.slider("Subsidiary Parity Cords Attached", min_value=0, max_value=4, value=1, help="Subsidiary strings hanging off pendants handle parity checksums.")

# --- CORE COMPUTATIONAL PHYSICS PHYSICS Engine ---
pair, reverse_pair = (origin_key, dest_key), (dest_key, origin_key)
dist = 0 if origin_key == dest_key else distance_matrix.get(pair, distance_matrix.get(reverse_pair, 5000))

raw_data_payload_mb = 600.0
base_speed = 9.2  # km/h at 5 knots

# 1. S/Z Compression Factor
if twist == "Z-Twist (Default/Linear)":
    compression_ratio = 1.0; structural_entropy = 0.42; twist_speed_mod = 1.0
else:
    compression_ratio = 2.5; structural_entropy = 0.91; twist_speed_mod = 0.85

# 2. Color Channel Multiplexing
if "Solid" in color_profile:
    mux_channels = 1; color_attenuation = 1.0
elif "Bi-Chrome" in color_profile:
    mux_channels = 2; color_attenuation = 1.18  # 18% signal drag due to phase-splitting
else:
    mux_channels = 3; color_attenuation = 1.35  # 35% chromatic interference drag

# 3. Subsidiary Error-Correction Parity
# Each subsidiary adds data structural mass (slowing travel) but exponentially drops frame drops
packet_loss_rate = max(0.0, 4.5 - (subsidiaries * 1.5))
parity_overhead_mod = 1.0 + (subsidiaries * 0.08)

# 4. Geometry and Material Constraints
hitch_modifier = 1.12 if attachment == "Verso (Orthogonal Resist)" else 1.00
material_attenuation = {"Whale Baleen (Rigid)": 0.95, "Marine Cotton (Standard)": 1.00, "Camelid Wool (Elastic)": 1.15}

# 5. Final Equations
effective_distance = dist * hitch_modifier * color_attenuation * parity_overhead_mod
final_speed = base_speed * twist_speed_mod
latency_days = 0.0 if dist == 0 else (effective_distance * material_attenuation[material]) / (final_speed * 24)
throughput_rate = (raw_data_payload_mb * mux_channels) / compression_ratio

# --- DASHBOARD VISUALIZATIONS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dynamic Multiplexing Profiles")
    st.metric("Aggregate Transmission Rate", f"{throughput_rate:.1f} Mod-MB Equivalents")
    st.metric("Total Latency Timeframe", f"{latency_days:.2f} Days")
    st.info(f"**Split Multiplexing:** Operating **{mux_channels} concurrent color carrier paths** across the main infrastructure backbone.")

with col2:
    st.subheader("🛡️ Array Integrity & Fault Tolerance")
    st.metric("Simulated Packet Drop Rate", f"{packet_loss_rate:.2f} %")
    st.metric("Structural Data Parity Buffer", f"+{(parity_overhead_mod - 1.0)*100:.0f}% mass overhead")
    st.progress(1.0 if latency_days == 0 else min(1.0, 12.0 / latency_days))

st.divider()

# --- GEOMETRIC RECORD MAP DISPLAY ---
st.subheader("🧮 Terminal Output Vector: Physical Cord Architecture")
rounded_days = int(round(latency_days))
hundreds, tens, units = rounded_days // 100, (rounded_days % 100) // 10, rounded_days % 10

if rounded_days == 0:
    st.code("─── (0.0 Days Latency / Local Interface Loopback Mode)")
else:
    khipu_string = f"───[Main Primary Cord Matrix | {origin_key.upper()} ➔ {dest_key.upper()}]───\n"
    khipu_string += f"   ├── Active Hash Matrix Configuration: [{twist[0]}][{attachment[0]}][{color_profile[0]}][Subs:{subsidiaries}]\n"
    
    # Render primary pendant cord
    if hundreds > 0:
        khipu_string += f"   ├── Pendant String Tier (10^2): {hundreds}x Cluster Knots (●)\n"
    if tens > 0:
        khipu_string += f"   ├── Pendant String Tier (10^1): {tens}x Cluster Knots (●)\n"
    if units > 0:
        khipu_string += "   ├── Pendant String Tier (10^0): 1x Figure-Eight Knot (∞)\n" if units == 1 else f"   ├── Pendant String Tier (10^0): 1x Long Knot [{units} wraps] (▰)\n"
    
    # Render dynamic subsidiary structures for parity
    for i in range(1, subsidiaries + 1):
        khipu_string += f"   │     └── [Subsidiary Cord {i}] ── Checksum Verification Segment (◈)\n"
        
    khipu_string += f"   └── [Terminal Node Loop Closure] Status Synchronized at Destination.\n"
    st.code(khipu_string, language="text")
