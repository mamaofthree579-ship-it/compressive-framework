import streamlit as st

st.set_page_config(page_title="Khipu Global Fiber", layout="wide")

st.title("🧶 Khipu Node v29: The Global Whale-Fiber")
st.write("Simulating Data Latency across the 8,000 km 'Kelp Highway' Backbone.")

# --- SIDEBAR: NETWORK CONTROLS ---
st.sidebar.header("🗺️ Network Topology")
origin = st.sidebar.selectbox("Origin Node", ["Bering Strait (Gateway)", "Columbia River (Hub)", "Amazon Delta (Server)"])
destination = st.sidebar.selectbox("Destination Node", ["Amazon Delta (Server)", "Tiwanaku (CPU)", "Easter Island (Relay)"])

# --- LATENCY LOGIC ---
# Whale migration speed is roughly 5 knots (9.2 km/h)
# Total distance roughly 15,000 km for a full run
distance_map = {"Bering-Amazon": 12000, "Bering-Easter": 10000, "Columbia-Amazon": 8000}
dist = distance_map.get(f"{origin.split(' ')[0]}-{destination.split(' ')[0]}", 5000)

latency_days = dist / (9.2 * 24) # Days to travel at 5 knots

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📡 Connection Status")
    st.metric("Link Distance", f"{dist:,} km")
    st.metric("Packet Latency", f"{latency_days:.1f} Days")
    st.info(f"**Route:** Following the {origin} to {destination} migratory cable.")

with col2:
    st.subheader("🛡️ Firewall Status")
    if "Bering" in origin:
        st.success("✅ GATEWAY OPEN: Whale Bone Alley protocol verified.")
    else:
        st.warning("🔒 INTERNAL TRAFFIC: Bypassing northern gateway.")
    st.progress(min(1.0, 1 / (latency_days / 10)))

st.divider()
st.subheader("🧬 Result: The Global Sync")
st.write(f"The 'String' will be updated at the {destination} node in **{latency_days:.1f} days**.")
st.write("This is the 'Ancient Internet'. It is slow, but it is **unbreakable**.")
