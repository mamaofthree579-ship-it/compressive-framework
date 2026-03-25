import streamlit as st
import pandas as pd
import requests
import io
import numpy as np

# 1. Haversine Distance Logic
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# 2. Robust UAP Data Load (CORGIS Mirror)
@st.cache_data
def get_uap_data():
    url = "https://corgis-edu.github.io"
    try:
        response = requests.get(url, timeout=10)
        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip', engine='python')
        df = df.rename(columns={'Location.Latitude': 'lat', 'Location.Longitude': 'lon', 'Location.City': 'city'})
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        return df.dropna(subset=['lat', 'lon'])
    except: return pd.DataFrame()

# 3. Live Seismic Feed (Filtering for Significance)
def get_significant_seismic():
    # Fetching M4.5+ (we filter for 5.5+ in the logic)
    url = "https://earthquake.usgs.gov"
    try:
        resp = requests.get(url, timeout=10).json()
        points = []
        for f in resp['features']:
            coords = f['geometry']['coordinates']
            points.append({
                'lat': coords[1], 'lon': coords[0], 
                'mag': f['properties']['mag'], 'place': f['properties']['place']
            })
        return pd.DataFrame(points)
    except: return pd.DataFrame()

# 4. Main Application
st.title("The Guardians: System Activation Log")
df_uap = get_uap_data()
live_stress = get_significant_seismic()

if not live_stress.empty and not df_uap.empty:
    log_entries = []
    active_clusters = []
    
    # 5. Logic: Filter for Higher Magnitude Response
    major_triggers = live_stress[live_stress['mag'] >= 5.5]
    
    for _, quake in major_triggers.iterrows():
        # Radius expands with magnitude (Sp^2 * 10)
        radius = (quake['mag'] ** 2) * 10
        dist = haversine(quake['lat'], quake['lon'], df_uap['lat'], df_uap['lon'])
        matches = df_uap[dist <= radius].copy()
        
        if not matches.empty:
            matches['type'] = 'Activated Node'
            active_clusters.append(matches)
            log_entries.append(f"⚠️ **ALERT:** M{quake['mag']} event near {quake['place']} has activated {len(matches)} Guardian nodes within {int(radius)}km.")

    # 6. Display Map & Log
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if active_clusters:
            map_data = pd.concat([major_triggers.assign(type='Seismic Trigger')] + active_clusters)
            st.map(map_data, color='type')
        else:
            st.info("No High-Magnitude (M5.5+) triggers detected in the last 24 hours.")
            st.map(live_stress.assign(type='Low Stress'))

    with col2:
        st.subheader("System Log")
        if log_entries:
            for entry in log_entries:
                st.write(entry)
        else:
            st.write("System Status: **DORMANT**. No major stressors detected.")

else:
    st.warning("Synchronizing with planetary data streams...")
