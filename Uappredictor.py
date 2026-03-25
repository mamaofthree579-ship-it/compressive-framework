import streamlit as st
import pandas as pd
import requests
import io
import numpy as np

# 1. Haversine Distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# 2. Robust Data Load (Fixes Tokenization Error)
@st.cache_data
def get_uap_data():
    url = "https://corgis-edu.github.io"
    try:
        # 'on_bad_lines' skips rows that don't match the expected column count
        response = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip', engine='python')
        return df.rename(columns={'Location.Latitude': 'lat', 'Location.Longitude': 'lon'})
    except: return pd.DataFrame()

# 3. Seismic & Cluster Logic
st.title("Guardian Predictor: Cluster Analysis")
df_uap = get_uap_data()
# Fetching live M4.5+ seismic data
resp = requests.get("https://earthquake.usgs.gov").json()
live_stress = pd.DataFrame([{'lat': f['geometry'], 'lon': f['geometry'], 'mag': f['properties']['mag']} for f in resp['features']])

if not live_stress.empty and not df_uap.empty:
    # Cluster Detection: Find UAPs within 300km of live stress
    active_nodes = []
    for _, quake in live_stress.iterrows():
        dist = haversine(quake['lat'], quake['lon'], df_uap['lat'], df_uap['lon'])
        matches = df_uap[dist <= 300].copy()
        matches['type'] = 'Active Cluster'
        active_nodes.append(matches)
    
    # Display Map
    display_df = pd.concat([live_stress.assign(type='Live Stress')] + active_nodes)
    st.map(display_df, color='type')
    st.write(f"Detected **{len(pd.concat(active_nodes))} historical nodes** in active cluster zones.")
