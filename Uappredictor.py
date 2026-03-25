import streamlit as st
import pandas as pd
import requests
import numpy as np

# 1. Stable Seismic Feed (Sp >= 4.5)
def get_stable_seismic_data():
    # Using the pre-generated 'Past Day' Summary Feed for stability
    url = "https://earthquake.usgs.gov"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            points = []
            for f in features:
                coords = f['geometry']['coordinates']
                points.append({
                    'lat': coords[1], # Latitude is index 1
                    'lon': coords[0], # Longitude is index 0
                    'mag': f['properties']['mag'],
                    'type': 'Live Stress'
                })
            return pd.DataFrame(points)
        return pd.DataFrame(columns=['lat', 'lon', 'mag', 'type'])
    except Exception:
        # Returns empty DF on connection failure so the app doesn't crash
        return pd.DataFrame(columns=['lat', 'lon', 'mag', 'type'])

# 2. Tier 1 UAP Data Loader
@st.cache_data
def get_tier_1_historical():
    url = "https://raw.githubusercontent.com"
    try:
        df = pd.read_csv(url).dropna(subset=['latitude', 'longitude'])
        # Filtering for Tier 1: Duration > 1 hour
        tier_1 = df[df['duration (seconds)'] > 3600].copy()
        tier_1 = tier_1.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        tier_1['type'] = 'Tier 1 Historical'
        return tier_1[['lat', 'lon', 'type']]
    except:
        return pd.DataFrame(columns=['lat', 'lon', 'type'])

# 3. Build Dashboard
st.title("Guardian Model v4.5: Robust Predictor")

live_df = get_stable_seismic_data()
uap_df = get_tier_1_historical()

# Calculate Sp based on live feed
sp_value = live_df['mag'].mean() if not live_df.empty else 1.0
st.sidebar.metric("Live Planetary Stress (Sp)", f"{sp_value:.2f}")

# Map Logic
if not live_df.empty or not uap_df.empty:
    # Overlaying both datasets
    combined = pd.concat([live_df, uap_df], ignore_index=True)
    st.map(combined, color='type')
    st.write("🔴 **Live Stress Points (M4.5+)** | 🔵 **Tier 1 Historical Sightings**")
else:
    st.warning("Data stream temporarily unavailable. Check connection.")
