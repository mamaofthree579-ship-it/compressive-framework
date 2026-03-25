import streamlit as st
import pandas as pd
import requests
import io
import numpy as np
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="Guardian Predictor", layout="wide")
st.title("The Guardians: Planetary Regulation System")

# 2. Stable Data Fetcher (CORGIS Mirror)
@st.cache_data
def get_uap_data():
    url = "https://corgis-edu.github.io"
    try:
        response = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(response.text))
        # RENAME IS CRITICAL: Streamlit needs 'lat' and 'lon'
        df = df.rename(columns={
            'Location.Latitude': 'lat',
            'Location.Longitude': 'lon',
            'Data.Date.Time': 'date_string'
        })
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

# 3. Live Seismic Fetcher (Sp)
def get_live_seismic():
    url = "https://earthquake.usgs.gov"
    try:
        data = requests.get(url).json()
        points = []
        for f in data['features']:
            c = f['geometry']['coordinates']
            points.append({'lat': c[1], 'lon': c[0], 'mag': f['properties']['mag'], 'type': 'Live Stress'})
        return pd.DataFrame(points)
    except:
        return pd.DataFrame()

# 4. Processing
df = get_uap_data()
live_seismic = get_live_seismic()

# Sidebar: Planetary Boundary Adjustment
st.sidebar.header("Planetary Health Check")
boundaries_crossed = st.sidebar.slider("Boundaries Surpassed (out of 9)", 0, 9, 7)
# Jones Logic: More boundaries crossed = Lower activation threshold
resonance_threshold = 9.0 - (boundaries_crossed * 0.4) 

# 5. Build the Map
if not df.empty:
    # Filter for 'Tier 1' (Sightings > 1 hour) to avoid a blank map from the time-delta
    tier_1_df = df[df['Data.Encounter.Duration'] > 3600].copy()
    tier_1_df['type'] = 'Historical Node'
    
    # Combine with Live Seismic
    combined = pd.concat([tier_1_df[['lat', 'lon', 'type']], live_seismic[['lat', 'lon', 'type']]])
    
    st.subheader(f"Current Activation Threshold (Kr): {resonance_threshold:.1f}")
    st.map(combined, color='type')
    
    st.write("🔴 **Red/Live:** Seismic Stress points (M4.5+)")
    st.write("🔵 **Blue/Historical:** Tier 1 Guardian Manifestations")
else:
    st.error("Dataset is empty or columns mismatched.")
