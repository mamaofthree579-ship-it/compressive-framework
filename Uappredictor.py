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

# 2. Robust UAP Data Load
@st.cache_data
def get_uap_data():
    url = "https://corgis-edu.github.io"
    try:
        response = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip', engine='python')
        # Ensure we have clean floats for coordinates
        df = df.rename(columns={'Location.Latitude': 'lat', 'Location.Longitude': 'lon'})
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        return df.dropna(subset=['lat', 'lon'])
    except: return pd.DataFrame()

# 3. Correct USGS JSON Feed
def get_live_seismic():
    # This specific URL provides the actual JSON data
    url = "https://earthquake.usgs.gov"
    try:
        resp = requests.get(url, timeout=10).json()
        points = []
        for f in resp['features']:
            coords = f['geometry']['coordinates']
            points.append({
                'lat': coords[1], 
                'lon': coords[0], 
                'mag': f['properties']['mag'],
                'type': 'Live Stress'
            })
        return pd.DataFrame(points)
    except: return pd.DataFrame()

# 4. Processing & Magnitude Weighting
st.title("Guardian Predictor: Weighted Cluster Analysis")
df_uap = get_uap_data()
live_stress = get_live_seismic()

if not live_stress.empty and not df_uap.empty:
    active_clusters = []
    
    for _, quake in live_stress.iterrows():
        # MAGNITUDE WEIGHT: Radius = (Magnitude^2) * 10 
        # (e.g., M5 = 250km, M7 = 490km)
        dynamic_radius = (quake['mag'] ** 2) * 10
        
        dist = haversine(quake['lat'], quake['lon'], df_uap['lat'], df_uap['lon'])
        matches = df_uap[dist <= dynamic_radius].copy()
        matches['type'] = 'Active Node'
        active_clusters.append(matches)
    
    # 5. Build Map
    all_points = pd.concat([live_stress.assign(type='Seismic Trigger')] + active_clusters)
    st.map(all_points, color='type')
    
    st.sidebar.write(f"Live Triggers: {len(live_stress)}")
    st.sidebar.write("🔴 **Red:** Seismic Trigger")
    st.sidebar.write("🔵 **Blue:** Activated Guardian Node")
else:
    st.warning("Awaiting live data stream...")

