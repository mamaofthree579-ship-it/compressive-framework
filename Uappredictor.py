import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

# 1. Haversine Calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# 2. Live Seismic Stress (Sp)
def get_live_seismic():
    url = "https://earthquake.usgs.gov"
    try:
        data = requests.get(url).json()
        return pd.DataFrame([{'lat': f['geometry'][1], 'lon': f['geometry'][0], 'mag': f['properties']['mag'], 'type': 'Live Stress'} for f in data['features']])
    except: return pd.DataFrame()

# 3. Time-Filtered Historical Data
@st.cache_data
def get_filtered_uaps(months_back):
    url = "https://raw.githubusercontent.com"
    df = pd.read_csv(url).dropna(subset=['latitude', 'longitude'])
    df['date'] = pd.to_datetime(df['date_time'], errors='coerce')
    
    # Apply Time Delta
    cutoff = datetime.now() - timedelta(days=30 * months_back)
    recent_df = df[df['date'] >= cutoff].copy()
    recent_df = recent_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    recent_df['type'] = 'Recent Sighting'
    return recent_df[['lat', 'lon', 'type', 'date']]

# 4. Interface
st.title("Guardian Model: Time Delta Analysis")
months = st.sidebar.slider("Time Window (Months Back)", 1, 60, 12)
radius = st.sidebar.slider("Proximity Radius (km)", 100, 1000, 500)

live_df = get_live_seismic()
recent_uaps = get_filtered_uaps(months)

# Calculate Correlation Density
if not live_df.empty and not recent_uaps.empty:
    matches = []
    for _, quake in live_df.iterrows():
        dist = haversine(quake['lat'], quake['lon'], recent_uaps['lat'], recent_uaps['lon'])
        matches.append(recent_uaps[dist <= radius])
    
    final_map_df = pd.concat([live_df] + matches).drop_duplicates()
    st.map(final_map_df, color='type')
    st.write(f"Showing {len(pd.concat(matches))} sightings matching the {months}-month delta within {radius}km of live stress.")
