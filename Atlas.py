import streamlit as st
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Atlas M1P Simulator", layout="wide")

# --- Constants from our Engineering Specs ---
PLATFORM_MAX_GRAVITIC_LOAD_KG = 100000  # 100 metric tons
PLATFORM_POWER_CELLS = 4
CELL_CAPACITY_KWH = 75
BASE_POWER_DRAW_KW = 5 # Idle power for the Atlas Core and emitters

# --- Simulator UI ---
st.title("Project Atlas M1P - Interactive Simulator")
st.markdown("Test the capabilities and limits of the Atlas Mark I Platform. Adjust the parameters in the sidebar to see how the system responds.")

st.sidebar.title("Simulation Controls")
st.sidebar.header("1. Load Configuration")
load_mass_kg = st.sidebar.slider("Cargo Mass (kg)", min_value=1000, max_value=500000, value=50000, step=1000)

st.sidebar.header("2. Platform Configuration")
num_platforms = st.sidebar.number_input("Number of Linked M1P Platforms", min_value=1, max_value=10, value=1)
power_source = st.sidebar.radio("Power Source", ["Battery Cells", "Tethered Grid"])

st.sidebar.header("3. Operator Controls ('The Leash')")
mass_reduction_setting_pct = st.sidebar.slider("Mass Reduction Setting (%)", min_value=0, max_value=100, value=99, step=1)

# --- Core Simulation Logic ---
# Calculate total system capacity
total_system_capacity_kg = num_platforms * PLATFORM_MAX_GRAVITIC_LOAD_KG

# Check for overload and calculate achievable reduction
is_overload = load_mass_kg > total_system_capacity_kg
max_achievable_reduction_pct = (total_system_capacity_kg / load_mass_kg) * 100 if is_overload else 100
applied_reduction_pct = min(mass_reduction_setting_pct, max_achievable_reduction_pct)

# Calculate the results
effective_mass_kg = load_mass_kg * (1 - (applied_reduction_pct / 100))
negated_mass_kg = load_mass_kg - effective_mass_kg

# Calculate power consumption (a simplified but logical model)
# Power scales with how much mass is being negated. 1 kW per ton negated.
power_draw_kw = BASE_POWER_DRAW_KW + (negated_mass_kg / 1000) if applied_reduction_pct > 0 else 0

# Calculate battery runtime if not tethered
total_battery_kwh = num_platforms * PLATFORM_POWER_CELLS * CELL_CAPACITY_KWH
runtime_hours = (total_battery_kwh / power_draw_kw) if power_draw_kw > 0 else float('inf')

# --- Display Results ---
st.header("System Status & Performance")
col1, col2, col3 = st.columns(3)

# Column 1: Load Status
with col1:
    st.subheader("Load Analysis")
    st.metric("Original Cargo Mass", f"{load_mass_kg:,} kg")
    st.metric("Effective Mass (Felt by Operator)", f"{int(effective_mass_kg):,} kg", delta=f"{-int(negated_mass_kg):,} kg")
    st.progress(applied_reduction_pct / 100)
    st.write(f"Applied Mass Reduction: **{applied_reduction_pct:.1f}%**")

# Column 2: Platform Status
with col2:
    st.subheader("Platform Performance")
    st.metric("Total System Capacity", f"{total_system_capacity_kg:,} kg")
    
    if is_overload:
        st.warning(f"OVERLOAD WARNING! Load exceeds max capacity. Mass reduction is limited to {max_achievable_reduction_pct:.1f}%.")
    else:
        st.success("System capacity is sufficient for this load.")
    
    st.info(f"The Atlas Stability Core is actively stabilizing a load of **{int(effective_mass_kg):,} kg**.")

# Column 3: Power Status
with col3:
    st.subheader("Power Consumption")
    st.metric("Current Power Draw", f"{power_draw_kw:.2f} kW")
    
    if power_source == "Tethered Grid":
        st.success("System is running on tethered grid power. Runtime is unlimited.")
    else:
        st.metric("Total Battery Capacity", f"{total_battery_kwh:,} kWh")
        if runtime_hours == float('inf'):
            st.info("System is idle. No power is being drawn.")
        else:
            st.info(f"Estimated Runtime on Batteries: **{runtime_hours:.2f} hours**")

st.header("Visual Representation")

# --- Simple bar chart visualization ---
# Represents the "work" the platform is doing
st.markdown("The blue bar represents the portion of the mass being negated by the Atlas platform.")
remaining_mass_pct = (effective_mass_kg / load_mass_kg) * 100
negated_mass_pct = 100 - remaining_mass_pct

# Create a visual that looks like a horizontal stacked bar
st.write(f"""
<div style="background-color: #e0e0e0; border-radius: 5px; height: 50px; width: 100%; display: flex;">
    <div style="background-color: #007bff; width: {negated_mass_pct}%; height: 100%; border-radius: 5px 0 0 5px; display: flex; align-items: center; justify-content: center; color: white;">
        Negated ({int(negated_mass_kg):,} kg)
    </div>
    <div style="display: flex; align-items: center; justify-content: center; width: {remaining_mass_pct}%;">
        Remaining ({int(effective_mass_kg):,} kg)
    </div>
</div>
""", unsafe_allow_html=True)
