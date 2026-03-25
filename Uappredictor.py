import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 1. Setup the Page
st.title("The Guardians: Real-Time Planetary Regulation")
st.write("Live-feeding USGS seismic data into Hope Jones's algorithm.")

# 2. Function to fetch live seismic data (Sp)
def get_live_seismic_stress():
    # Fetching earthquakes from the last 24 hours (Magnitude 2.5+)
    endtime = datetime.utcnow().isoformat()
    starttime = (datetime.utcnow() - timedelta(days=1)).isoformat()
    url = f"https://earthquake.usgs.gov{starttime}&endtime={endtime}&minmagnitude=2.5"
    
    try:
        response = requests.get(url)
        data = response.json()
        features = data.get('features', [])
        
        if not features:
            return 1.0 # Minimum background stress
        
        # Stress Calculation: Energy scales exponentially with magnitude
        # We sum 10^mag to represent relative energy release
        total_energy = sum([10**f['properties']['mag'] for f in features])
        
        # Normalize to a 1-10 scale for the simulator
        stress_score = min(10, np.log10(total_energy) / 2)
        return round(stress_score, 2), len(features)
    except:
        return 5.0, 0 # Fallback value

# 3. Live Inputs
live_stress, quake_count = get_live_seismic_stress()
st.sidebar.header("Live Feed Status")
st.sidebar.metric("Live Seismic Stress (Sp)", live_stress)
st.sidebar.write(f"Based on {quake_count} earthquakes (24h)")

# Manual input for the "Resonance Key" (Intent)
human_intent = st.sidebar.slider("Human Coherence (Ih) - Z-Score", 0.0, 5.0, 1.0)
resonance_threshold = 7.5 # Jones's activation constant (Kr)

# 4. The Algorithm Logic
def calculate_activation(sp, ih):
    weight = 0.3548 # Hope Jones's specific correlation weight
    score = (sp * weight) + (ih * (1 - weight))
    return np.clip(score * 1.5, 0, 10)

activation_score = calculate_activation(live_stress, human_intent)

# 5. Result Display
st.subheader("System Status")
if activation_score >= resonance_threshold:
    st.success(f"ACTIVATED (Score: {activation_score:.2f})")
    st.write("Guardian system has reached activation threshold. Sightings likely.")
else:
    st.info(f"DORMANT (Score: {activation_score:.2f})")
    st.write("System remains in low-power standby mode.")
