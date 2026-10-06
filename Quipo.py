import streamlit as st
import math

# --- MAIN PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Khipu Global Fiber Topology v33", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🧶 Khipu Node v33: Unified Global Narrative Matrix Engine")
st.write("Simulating Haversine Geographic Coordinates, Administrative Mita Protocols, and Narrative String Decoders.")

# --- GEOGRAPHIC TRUE COORDINATE DATABASE ---
nodes_geo = {
    "Bering": {"name": "Bering Strait (Gateway)", "lat": 65.6, "lon": -168.9},
    "Columbia": {"name": "Columbia River (Hub)", "lat": 46.2, "lon": -124.0},
    "Amazon": {"name": "Amazon Delta (Server)", "lat": -0.1, "lon": -49.0},
    "Tiwanaku": {"name": "Tiwanaku (CPU)", "lat": -16.5, "lon": -68.7},
    "Easter": {"name": "Easter Island (Relay)", "lat": -27.1, "lon": -109.4}
}

# Haversine Global Curve Formula Engine
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth's radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

# --- SIDEBAR CONTROL PANEL MATRIX ---
st.sidebar.header("🗺️ Geographic Topology Router")
origin_key = st.sidebar.selectbox("Origin Node", list(nodes_geo.keys()), index=0, format_func=lambda x: nodes_geo[x]["name"])
dest_key = st.sidebar.selectbox("Destination Node", list(nodes_geo.keys()), index=3, format_func=lambda x: nodes_geo[x]["name"])

st.sidebar.header("⚖️ Administrative Taxation Engine")
protocol_mode = st.sidebar.selectbox(
    "Inca Administrative Class", 
    ["Mita (Labor Tax Stream)", "Agricultural Quota (Maize/Chicha)", "Chasqui Royal Decree (High Priority)", "Military Mobilization Fleet"]
)

st.sidebar.header("🪢 Structural Mechanics Vector")
twist = st.sidebar.radio("Structural Axis (Urton Markedness)", ["Z-Twist (Default/Linear)", "S-Twist (Marked/Compressed)"])
attachment = st.sidebar.radio("Hitch Geometry", ["Recto (Parallel Pack)", "Verso (Orthogonal Resist)"])
color_profile = st.sidebar.selectbox(
    "Cord Color Schema", 
    ["Solid Natural (Single Carrier)", "Bi-Chrome Barber-Pole (2-Channel MUX)", "Tri-Color Mottled (3-Channel MUX)"]
)
subsidiaries = st.sidebar.slider("Subsidiary Parity Cords Attached", min_value=0, max_value=4, value=1)

# --- ENGINE COMPUTATIONAL LAYER ---
geo1, geo2 = nodes_geo[origin_key], nodes_geo[dest_key]
true_distance = calculate_haversine_distance(geo1["lat"], geo1["lon"], geo2["lat"], geo2["lon"])

base_speed = 9.2 # Baseline migration velocity (5 knots in km/h)

# 1. Administrative Protocol Tuning
protocol_map = {
    "Mita (Labor Tax Stream)": {"payload": 1200.0, "thickness": "8.5mm Heavy", "speed_mod": 0.90},
    "Agricultural Quota (Maize/Chicha)": {"payload": 750.0, "thickness": "5.2mm Standard", "speed_mod": 1.00},
    "Chasqui Royal Decree (High Priority)": {"payload": 50.0, "thickness": "1.8mm Ultra-Light", "speed_mod": 1.45},
    "Military Mobilization Fleet": {"payload": 2200.0, "thickness": "12.0mm Heavy Reinforced", "speed_mod": 0.75}
}
current_proto = protocol_map[protocol_mode]

# 2. Structural Matrix Modifications
if twist == "Z-Twist (Default/Linear)":
    compression_ratio = 1.0
    structural_entropy = 0.42
    twist_mod = 1.0
else:
    compression_ratio = 2.6
    structural_entropy = 0.94
    twist_mod = 0.85

mux_channels = 1 if "Solid" in color_profile else (2 if "Bi-Chrome" in color_profile else 3)
color_attenuation = 1.0 if mux_channels == 1 else (1.18 if mux_channels == 2 else 1.35)

packet_loss_rate = max(0.0, 4.5 - (subsidiaries * 1.5))
parity_overhead_mod = 1.0 + (subsidiaries * 0.08)
hitch_modifier = 1.12 if attachment == "Verso (Orthogonal Resist)" else 1.00

# 3. Final Multi-Variable System Equations
effective_distance = true_distance * hitch_modifier * color_attenuation * parity_overhead_mod
final_calculated_speed = base_speed * twist_mod * current_proto["speed_mod"]
latency_days = 0.0 if true_distance == 0 else effective_distance / (final_calculated_speed * 24)
throughput_rate = (current_proto["payload"] * mux_channels) / compression_ratio

# --- RESEARCH DASHBOARD GRID ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌐 Geographic & Logistical Array")
    st.metric("True Haversine Curve Distance", f"{true_distance:.1f} km")
    st.metric("Effective Physical Latency", f"{latency_days:.2f} Days")
    st.info(f"**Protocol Matrix:** {protocol_mode} is transmitting a **{current_proto['thickness']}** fiber structure array.")

with col2:
    st.subheader("🧮 Data-Stream Multiplex Metrics")
    st.metric("Aggregate Transmission Rate", f"{throughput_rate:.1f} Mod-MB Equivalents")
    st.metric("System Information Entropy", f"{structural_entropy:.2f} Sh/cord")
    
    # Progress visualization safely handling loopback bounds
    progress_val = 1.0 if latency_days == 0 else min(1.0, 12.0 / latency_days)
    st.progress(progress_val)

st.divider()

# --- DECIMAL KHIPU STRUCTURAL TRANSLATION ---
st.subheader("🪢 Terminal Output Vector: Physical Cord Architecture")
rounded_days = int(round(latency_days))
hundreds = rounded_days // 100
tens = (rounded_days % 100) // 10
units = rounded_days % 10

if rounded_days == 0:
    st.code("─── (0.0 Days Latency / Local Interface Loopback Mode)", language="text")
else:
    khipu_string = f"───[Main Primary Cord Matrix | {origin_key.upper()} ({geo1['lat']:.1f}°) ➔ {dest_key.upper()} ({geo2['lat']:.1f}°)]───\n"
    khipu_string += f"   ├── Administrative Signature: [{protocol_mode}] Fiber Density Vector: {current_proto['thickness']}\n"
    khipu_string += f"   ├── Active Hash Matrix Layout: [{twist}][{attachment}][{color_profile}]\n"
    
    # Base-10 Vertical Tier Generator
    if hundreds > 0:
        khipu_string += f"   ├── Pendant String Tier (10^2): {hundreds}x Simple Cluster Knots (●)\n"
    if tens > 0:
        khipu_string += f"   ├── Pendant String Tier (10^1): {tens}x Simple Cluster Knots (●)\n"
    if units > 0:
        if units == 1:
            khipu_string += "   ├── Pendant String Tier (10^0): 1x Figure-Eight Knot (∞)\n"
        else:
            khipu_string += f"   ├── Pendant String Tier (10^0): 1x Long Knot [{units} wraps] (▰)\n"
    else:
        khipu_string += "   ├── Pendant String Tier (10^0): 0x Knot Void [Explicit Decimal Zero] ( )\n"
    
    # Recursive Parity Subsections
    for i in range(1, subsidiaries + 1):
        khipu_string += f"   │     └── [Subsidiary Parity {i}] ── Checksum Verification Segment (◈)\n"
        
    khipu_string += f"   └── [Terminal Node Loop Closure] Status Synchronized at Destination.\n"
    st.code(khipu_string, language="text")

# --- NARRATIVE MATRIX CONVERTER ARRAY ---
st.subheader("📖 Narrative Matrix Mode Translation Syntax")
narrative_syntax = f"// ARCHAEOLOGICAL TRANSLATION SYSTEM //\n"
narrative_syntax += f"RECORD-TYPE: {protocol_mode.upper()} // SOURCE_NODE: {origin_key.upper()} // DEST_NODE: {dest_key.upper()}\n"
narrative_syntax += f"CORD_METRIC: Distance computed via Haversine curve to factor {true_distance:.2f} km of oceanic backbone.\n"
narrative_syntax += f"ENCODING_SYNTAX: Structural choice of {twist[:7]} reveals a base informational payload modifier of {compression_ratio}x compression.\n"
narrative_syntax += f"READOUT: 'Under authority of the administrative protocol, message reached terminal checkpoint safely in {rounded_days} planetary cycles, backed by {subsidiaries} structural validation check-cords.'"
st.code(narrative_syntax, language="python")
