import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Dashboard Header
st.title("The Guardians: Live Planetary & UAP Prediction Map")
st.write("Visualizing planetary stress (Sp) triggers across the global grid.")

# 2. Fetch Live Seismic Data with Coordinates
def get_live_map_data():
    endtime = datetime.utcnow().isoformat()
    starttime = (datetime.utcnow() - timedelta(days=7)).isoformat() # Past 7 days for better map density
    url = f"https://earthquake.usgs.gov{starttime}&endtime={endtime}&minmagnitude=4.0"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Extract coordinates and magnitudes
        locations = []
        for f in data['features']:
            locations.append({
                'latitude': f['geometry']['coordinates'][1],
                'longitude': f['geometry']['coordinates'][0],
                'magnitude': f['properties']['mag']
            })
        
        df = pd.DataFrame(locations)
        # Calculate Stress Score (Sp) from average magnitude
        stress_score = df['magnitude'].mean() if not df.empty else 1.0
        return df, round(stress_score, 2)
    except:
        return pd.DataFrame(columns=['latitude', 'longitude']), 5.0

# 3. Process Inputs
df_map, live_stress = get_live_map_data()

st.sidebar.header("Parameters")
st.sidebar.metric("Average Seismic Stress (Sp)", live_stress)
human_intent = st.sidebar.slider("Human Coherence (Ih) - Z-Score", 0.0, 5.0, 1.0)
resonance_threshold = 7.5

# 4. Jones Algorithm Execution
def calculate_activation(sp, ih):
    weight = 0.3548 # Jones's correlation coefficient
    # Activation Score normalized to a 10-point scale
    score = (sp * weight) + (ih * (1 - weight))
    return np.clip(score * 1.5, 0, 10)

activation_score = calculate_activation(live_stress, human_intent)

# 5. Display Prediction and Map
st.subheader("Planetary Stress Map (Active Triggers)")
if not df_map.empty:
    st.map(df_map) # Streamlit built-in mapping component
else:
    st.write("No major seismic triggers detected in the last 7 days.")

st.subheader("System Status")
if activation_score >= resonance_threshold:
    st.error(f"SYSTEM ACTIVE (Score: {activation_score:.2f})")
    st.write("Predictive Model: UAP activity likely near mapped seismic coordinates.")
else:
    st.success(f"SYSTEM DORMANT (Score: {activation_score:.2f})")
    st.write("The 'Guardians' remain in low-power standby mode.")
