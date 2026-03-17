import streamlit as st

st.set_page_config(page_title="Khipu Bio-Acoustic Interface", layout="wide")

st.title("🧶 Khipu Node v8: The Bio-Acoustic Interface")
st.write("Simulating the synergy between Alkaloid chemistry and Gallery acoustics.")

# --- SIDEBAR: BIO-CHEMICAL INPUT ---
st.sidebar.header("🧪 Bio-Chemical State")
alkaloid = st.sidebar.selectbox("Active Substance", ["None", "San Pedro (Mescaline)", "Anadenanthera (DMT)"])
gallery_res = st.sidebar.number_input("Gallery Resonance (Hz)", value=110)

# --- SYNERGY LOGIC ---
# If DMT/Mescaline is present, the 'threshold' for entrainment is significantly lower
sensitivity = 2.0 if alkaloid != "None" else 1.0
gamma_potential = (40 * sensitivity) / 2

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Chemical Status")
    st.write(f"**Alkaloid:** {alkaloid}")
    if alkaloid != "None":
        st.success(f"🚨 DMN SUPPRESSED: Brain entropy increased. Sensitivity at {sensitivity}x.")
    else:
        st.warning("⚠️ DMN ACTIVE: Logical 'Firewall' is resisting entrainment.")

with col2:
    st.subheader("🌀 Resonance Coupling")
    st.metric("Gamma Potential", f"{gamma_potential} %")
    if gamma_potential >= 40 and alkaloid != "None":
        st.success("✅ INTERFACE UNLOCKED: Full 'String' Database Access.")
    else:
        st.info("🧘 STATE: Preparatory / Ritual Observation.")

st.divider()
st.subheader("🧬 Result: The 'Living' Database")
st.write(f"With **{alkaloid}** in the system, the architecture is no longer 'stone'; it is an extension of the nervous system.")
st.write("In String Theory terms, the 'Observer' and the 'String' have become a single vibrating system.")
