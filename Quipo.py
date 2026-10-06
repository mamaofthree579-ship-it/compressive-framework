import streamlit as st
import math

st.set_page_config(page_title="Khipu Matrix Physics", layout="wide")

st.title("🧶 Khipu Node v31: S/Z Compression Physics Engine")
st.write("Simulating Data Density, Structural Entropy, and Multi-Tiered Geometric Matrices on the Kelp Highway.")

# --- ENCODING DICTIONARIES & PROPERTIES ---
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

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🗺️ Network Topology")
origin_key = st.sidebar.selectbox("Origin Node", list(nodes.keys()), index=0, format_func=lambda x: nodes[x])
dest_key = st.sidebar.selectbox("Destination Node", list(nodes.keys()), index=2, format_func=lambda x: nodes[x])

st.sidebar.header("🪢 Structural Matrix Variables")
twist = st.sidebar.radio("Structural Axis (Urton Markedness)", ["Z-Twist (Default/Linear)", "S-Twist (Marked/Compressed)"])
attachment = st.sidebar.radio("Hitch Geometry", ["Recto (Parallel Pack)", "Verso (Orthogonal Resist)"])
material = st.sidebar.selectbox("Fiber Material Vector", ["Whale Baleen (Rigid)", "Marine Cotton (Standard)", "Camelid Wool (Elastic)"])

# --- PHYSICAL ENGINE MATHEMATICS ---
pair, reverse_pair = (origin_key, dest_key), (dest_key, origin_key)
dist = 0 if origin_key == dest_key else distance_matrix.get(pair, distance_matrix.get(reverse_pair, 5000))

# Baseline parameters
raw_data_payload_mb = 450.0
base_speed = 9.2 # 5 Knots in km/h

# Asymmetrical Physics Tuning
if twist == "Z-Twist (Default/Linear)":
    compression_ratio = 1.0  # Raw uncompressed stream
    structural_entropy = 0.42 # Low structural layout complexity
    speed_factor = 1.0
else:
    compression_ratio = 2.45 # S-Twist acts as a code modifier compressing structural footprint
    structural_entropy = 0.89 # Highly variable informational density
    speed_factor = 0.85 # Tying and interpreting marked S-shunts slows raw travel speed by 15%

# Attachment physics mod
hitch_modifier = 1.12 if attachment == "Verso (Orthogonal Resist)" else 1.00
effective_distance = dist * hitch_modifier

# Material attenuation factors
material_attenuation = {"Whale Baleen (Rigid)": 0.95, "Marine Cotton (Standard)": 1.00, "Camelid Wool (Elastic)": 1.15}
mat_factor = material_attenuation[material]

# Final Payload calculations
compressed_payload_size = raw_data_payload_mb / compression_ratio
latency_days = 0.0 if dist == 0 else (effective_distance * mat_factor) / (base_speed * speed_factor * 24)

# --- GRAPHIC INTERFACE GRID ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Physics & Compression Summary")
    st.metric("Compression Efficiency Ratio", f"{compression_ratio:.2f} : 1")
    st.metric("Effective Physical Latency", f"{latency_days:.2f} Days")
    st.info(f"**Structural Footprint:** Compressing original {raw_data_payload_mb} MB payload down to **{compressed_payload_size:.1f} MB** string length equivalents.")

with col2:
    st.subheader("🧬 Material Vector Attenuation")
    st.metric("Matrix Structural Entropy", f"{structural_entropy:.2f} Sh/cord")
    st.metric("Dynamic Speed Constriction Factor", f"{speed_factor * (1/mat_factor):.2f}x")
    
    progress_val = 1.0 if latency_days == 0 else min(1.0, 12.0 / latency_days)
    st.progress(progress_val)

st.divider()

# --- DECIMAL KHIPU STRUCTURAL TRANSLATION ---
st.subheader("🧮 Resulting Physical Cord Matrix State")
rounded_days = int(round(latency_days))
hundreds, tens, units = rounded_days // 100, (rounded_days % 100) // 10, rounded_days % 10

if rounded_days == 0:
    st.code("─── (No Latency / Local Network Loopback)")
else:
    khipu_string = f"───[Main Primary Cord String Matrix | Origin: {origin_key} ➔ Dest: {dest_key}]───\n"
    khipu_string += f"   ├── Physics Modifier Tag: [{twist[:1]}][{attachment[:1]}][{material[:3].upper()}]\n"
    if hundreds > 0:
        khipu_string += f"   ├── Hundreds Tier (10^2): {hundreds}x Simple Cluster Knots (●)\n"
    if tens > 0:
        khipu_string += f"   ├── Tens Tier     (10^1): {tens}x Simple Cluster Knots (●)\n"
    if units > 0:
        khipu_string += "   └── Units Tier    (10^0): 1x Figure-Eight Knot (∞)\n" if units == 1 else f"   └── Units Tier    (10^0): 1x Long Knot [{units} wraps] (▰)\n"
    
    st.code(khipu_string, language="text")
