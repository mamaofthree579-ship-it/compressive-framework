import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import physics_core as pc
import anim_renderers as ar

st.set_page_config(page_title="Manas-Yantra Grand Unified Deck", layout="wide")
st.title("Project Manas-Yantra: Unified Flight Deck Simulation Matrix")
st.markdown("---")

# --- Interface Controls Setup ---
st.sidebar.title("🎛️ Flight Command Core")
yantra_algorithm = st.sidebar.selectbox("Load Symbolic Algorithm (The Yantra):", [
    "Shri Yantra (Linear Vector Propagation)", "Shiva-Shakti Kundalini (Vortex Coherence)", 
    "Chakra Mandala (Isotropic Sphere Mask)", "Trishula Vector (Directed Focus Point)"
])
parsed_yantra = pc.parse_yantra_symbol(yantra_algorithm)
system_choice = parsed_yantra["mode"]
st.sidebar.info(f"Operational Mode: **{system_choice}**")

brainwave_state = st.sidebar.radio("Pilot Cognitive State:", ["Alpha (Flow State)", "Beta (Analytical Focus)", "Gamma (Peak Neural Coherence)"])
simulated_voltage, eeg_amplitude, calculated_stress = (45.2, 12.5, 150.4) if "Alpha" in brainwave_state else ((88.7, 22.1, 295.6) if "Beta" in brainwave_state else (142.3, 35.8, 474.1))

mantra_input = st.sidebar.selectbox("Stabilizing Acoustic Resonance (Vocal Code):", ["OM (Isotropic Carrier Wave)", "HAM (Mercury Core Resonance)", "VAM (Fluidic Density Stabilizer)"])
vocal_freq, core_attenuation = (136.1, 0.95) if "OM" in mantra_input else ((480.0, 1.45) if "HAM" in mantra_input else (220.0, 1.15))

magnetic_field_strength = st.sidebar.slider("Magnetic Field Flux (Tesla)", 10, 250, 120)
plasma_temperature = st.sidebar.slider("Mercury Plasma Temp (Kelvin)", 1000, 15000, 6500, 500)

# --- NEW PLANETARY NAVIGATION MATRIX CONTROLS ---
st.sidebar.subheader("🌍 Earth Resonance Environment")
flight_altitude = st.sidebar.slider("Flight Operational Altitude (km)", 0.5, 120.0, 12.0, 0.5)
local_geomagnetic_flux = st.sidebar.slider("Geomagnetic Flux (Micro-Tesla)", 20.0, 70.0, 48.0, 1.0)

if system_choice == "Propulsion & Maneuvers":
    well_depth_mod = st.sidebar.slider("Well Depth (-M)", 10, 120, int((magnetic_field_strength*0.4) + (simulated_voltage*0.3)))
    asymmetry_scale = st.sidebar.slider("Gradient Vector Skew (α)", 0.1, 2.5, float((plasma_temperature/15000)*1.5 + (eeg_amplitude*0.02)), 0.1)
elif system_choice == "Invisibility System":
    envelope_radius = st.sidebar.slider("Diffraction Radius (m-rad)", 2.0, 15.0, float((magnetic_field_strength*0.05) + 3.0), 0.5)
    diffraction_mod = st.sidebar.slider("Refractive Index Rate", 1.0, 12.0, float((simulated_voltage*0.05) + (plasma_temperature/5000)), 0.5)
else:
    beam_intensity = st.sidebar.slider("Focal Core Compression Intensity", 1.0, 6.0, float((magnetic_field_strength*0.015) + (simulated_voltage*0.02)), 0.1)

st.sidebar.subheader("🔬 Advanced Material & Aero Profilers")
alloy_selection = st.sidebar.selectbox("Select Target Alloy Specimen:", ["Kundodara Loha (Engine Core)", "Raajaloha (Airframe Hull)", "Vishvodara Metal"])
target_velocity_mach = st.sidebar.slider("Target Flight Velocity (Mach)", 0.0, 8.0, 2.4, 0.2)
atmospheric_density = st.sidebar.slider("Atmospheric Density (kg/m³)", 0.01, 1.22, 0.45, 0.05)

# --- Multi-Column Visual Layout Rendering ---
display_col, telemetry_col = st.columns([2.2, 1.0])

with display_col:
    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.markdown("#### 🌌 Active Macro-Field Map")
        if system_choice == "Propulsion & Maneuvers": 
            st.image(ar.render_propulsion(well_depth_mod, asymmetry_scale), use_container_width=True)
        elif system_choice == "Invisibility System": 
            st.image(ar.render_invisibility(envelope_radius, diffraction_mod), use_container_width=True)
        else: 
            st.image(ar.render_weapon(beam_intensity), use_container_width=True)
            
    with row1_right:
        st.markdown("#### 🧲 Core MHD Mercury Plasma Loop")
        st.image(ar.render_plasma_vortex(plasma_temperature, magnetic_field_strength), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 🔷 Hyperspatial Core Topology Projection (4D Coordinate Fold)")
    left_pad, center_slot, right_pad = st.columns([0.15, 1.0, 0.15])
    with center_slot: 
        st.image(ar.render_tesseract(simulated_voltage), use_container_width=True)

with telemetry_col:
    st.markdown("### 📊 Grand Unified System Metrics")
    st.metric(label="MHD Vortex Angular Velocity", value=f"{(magnetic_field_strength * 72.5):,.0f} rad/s")
    st.metric(label="Substrate Electrical Resonance Gain", value=f"{simulated_voltage:.1f} mV")
    st.metric(label="Acoustic Carrier Lock Frequency", value=f"{vocal_freq:.1f} Hz")
    
    st.markdown("---")
    m_name, diff_coeff, sat_pct = pc.calculate_biometallurgy(alloy_selection, plasma_temperature)
    eff_Cd, drag_force = pc.calculate_aerodynamics(target_velocity_mach, atmospheric_density, well_depth_mod if system_choice == "Propulsion & Maneuvers" else 0.0)
    
    # Calculate live planetary Schumann coupling matrix metrics
    c_factor, mag_stab, target_harm, lock_txt = pc.calculate_planetary_resonance_coupling(vocal_freq, flight_altitude, local_geomagnetic_flux * 1e-6)
    
    st.write(f"🔬 **{m_name}** at {plasma_temperature}K holds a diffusion rate of **{diff_coeff:.2e} m²/s**, providing structural saturation metrics of **{sat_pct:.2f}%**.")
    st.write(f"✈️ At **Mach {target_velocity_mach}**, localized medium compression drops the airframe drag down to **{eff_Cd:.3f}**, anchoring current kinetic drag resistance to **{drag_force:.2f} kN**.")
    
    st.markdown("---")
    st.markdown("**🌍 Earth Cavity Telemetry:**")
    st.write(f"📡 Stabilizer mapping tracks harmonic resonance proximity to Earth's **{target_harm}th sub-harmonic band**.")
    st.write(f"Coherence Index: **{c_factor:.3f}** | Stabilization Gain: **{mag_stab:.2e} N-m**")
    st.info(f"Status Matrix: **{lock_txt}**")

# --- DIAGNOSTICS GRAPH MATRICES ---
st.markdown("---")
st.header("🛡️ Sub-Hull Structural Analytics & Planetary Coupling Vectors")
diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.subheader("🛡️ Raajaloha Airframe Fatigue Predictor")
    base_load = (magnetic_field_strength * 1.5) + (plasma_temperature / 200.0)
    attenuated_load = base_load / core_attenuation
    yield_strength_pct = min(100.0, (attenuated_load / 500.0) * 100.0)
    
    fig_hull, ax_hull = plt.subplots(figsize=(6, 3))
    ax_hull.set_facecolor('black'); fig_hull.patch.set_facecolor('black')
    stress_x = np.linspace(0, 10, 100)
    strain_y = 100 * (1 - np.exp(-stress_x / 3.0))
    ax_hull.plot(stress_x, strain_y, color='cyan', label='Raajaloha Elastic Limit')
    ax_hull.axvline(x=attenuated_load / 50.0, color='red', linestyle='--', label='Active Structural Vector')
    ax_hull.set_xlabel('Applied Hyperspatial Stress Metric', color='white')
    ax_hull.set_ylabel('Micro-Strain Deformation (%)', color='white')
    ax_hull.tick_params(colors='white'); ax_hull.legend(facecolor='black', edgecolor='white', labelcolor='white')
    st.pyplot(fig_hull)
    st.metric(label="Calculated Material Hull Fatigue Yield Capacity", value=f"{yield_strength_pct:.2f} %")

with diag_col2:
    st.subheader("🌐 Coupled Earth-Stabilizer Phase Wavefront")
    # Call the new real-time atmospheric coupling wave visualizer module
    stream_data = ar.render_planetary_interference_field(vocal_freq, c_factor)
    st.image(stream_data, use_container_width=True)
