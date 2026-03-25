import streamlit as st
import pandas as pd
import requests
import numpy as np

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# 1. Stable Seismic Feed (M4.5+)
def get_stable_seismic():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    try:
        resp = requests.get(url, timeout=10).json()
        points = []
        for f in resp['features']:
            coords = f['geometry']['coordinates']
            points.append({'lat': coords[1], 'lon': coords[0], 'mag': f['properties']['mag'], 'type': 'Live Stress'})
        return pd.DataFrame(points)
    except:
        return pd.DataFrame(columns=['lat', 'lon', 'mag', 'type'])

# 2. Tier 1 Historical Loader
@st.cache_data
def get_historical_uaps():
    url = "https://raw.githubusercontent.com"
    try:
        df = pd.read_csv(url).dropna(subset=['latitude', 'longitude'])
        t1 = df[df['duration (seconds)'] > 3600].copy()
        t1 = t1.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        t1['type'] = 'Tier 1 Historical'
        return t1[['lat', 'lon', 'type']]
    except:
        return pd.DataFrame(columns=['lat', 'lon', 'type'])

# 3. Main Dashboard
st.title("Guardian Predictor: Proximity Analysis")

radius = st.sidebar.slider("Activation Radius (km)", 100, 2000, 500)
live_df = get_stable_seismic()
uap_df = get_historical_uaps()

# 4. Filter for Proximity
triggered_uaps = []
if not live_df.empty and not uap_df.empty:
    for _, quake in live_df.iterrows():
        # Vectorized distance check
        distances = haversine(quake['lat'], quake['lon'], uap_df['lat'], uap_df['lon'])
        close_uaps = uap_df[distances <= radius].copy()
        if not close_uaps.empty:
            close_uaps['trigger_mag'] = quake['mag']
            triggered_uaps.append(close_uaps)

# 5. Display
if triggered_uaps:
    final_df = pd.concat(triggered_uaps).drop_duplicates()
    st.subheader(f"Historical Sites within {radius}km of Live Stress")
    st.map(pd.concat([live_df, final_df]))
    st.write(f"Found {len(final_df)} historical nodes potentially 'primed' by current activity.")
else:
    st.info("No historical Tier 1 sites within range of current live stress.")
    st.map(live_df)
