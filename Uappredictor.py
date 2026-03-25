import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta

# 1. Stable Seismic Feed (The Trigger)
def get_seismic_trigger():
    url = "https://earthquake.usgs.gov"
    try:
        data = requests.get(url, timeout=5).json()
        points = []
        for f in data['features']:
            coords = f['geometry']['coordinates']
            points.append({'lat': coords[1], 'lon': coords[0], 'mag': f['properties']['mag'], 'type': 'Live Stress'})
        return pd.DataFrame(points)
    except:
        return pd.DataFrame()

# 2. Optimized UAP Loader (The Manifestation)
@st.cache_data
def get_stable_uap_data():
    # Using a slightly smaller, more stable dataset mirror
    url = "https://raw.githubusercontent.com"
    try:
        # Load only necessary columns to save memory/bandwidth
        df = pd.read_csv(url, usecols=['datetime', 'latitude', 'longitude', 'shape'])
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['date'] = pd.to_datetime(df['datetime'], errors='coerce')
        return df.dropna(subset=['lat', 'lon', 'date'])
    except:
        return pd.DataFrame()

# 3. Main Logic
st.title("Guardian Predictor: Time-Sync Mode")

# User Controls
months_back = st.sidebar.slider("Time Delta (Months)", 1, 120, 24)
live_stress = get_seismic_trigger()
all_uaps = get_stable_uap_data()

if not all_uaps.empty:
    # Apply Time Delta Filter
    cutoff = datetime.now() - timedelta(days=30 * months_back)
    recent_uaps = all_uaps[all_uaps['date'] >= cutoff].copy()
    recent_uaps['type'] = 'Time-Synced Sighting'

    # Mapping
    if not live_stress.empty:
        st.subheader(f"Global Correlation: Past {months_back} Months")
        st.map(pd.concat([live_stress, recent_uaps[['lat', 'lon', 'type']]]))
        st.write("🔴 **Live Stress** | 🔵 **Recent Guardian Activity**")
    else:
        st.warning("Seismic feed busy. Showing historical nodes only.")
        st.map(recent_uaps)
else:
    st.error("Data stream failed. GitHub is throttling the connection.")
