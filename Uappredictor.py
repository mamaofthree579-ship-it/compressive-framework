import streamlit as st
import pandas as pd
import requests
import numpy as np

# 1. Improved Major Stress Fetcher
def get_major_triggers():
    # Attempting to fetch significant quakes from the last 30 days
    url = "https://earthquake.usgs.gov"
    
    try:
        response = requests.get(url, timeout=10)
        # Check if the response is actually successful
        if response.status_code == 200:
            data = response.json()
            features = data.get('features', [])
            
            if not features:
                st.warning("No Major Stress (M6.0+) detected in the current window.")
                return pd.DataFrame(columns=['lat', 'lon', 'type'])
            
            # Extracting coordinates safely
            points = []
            for f in features:
                coords = f['geometry']['coordinates']
                points.append({
                    'lat': coords[1], # Latitude
                    'lon': coords[0], # Longitude
                    'type': 'Major Stress'
                })
            return pd.DataFrame(points)
        else:
            st.error(f"USGS API Error: Status {response.status_code}")
            return pd.DataFrame(columns=['lat', 'lon', 'type'])
            
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return pd.DataFrame(columns=['lat', 'lon', 'type'])

# 2. Historical Data Loader (Stays the same)
@st.cache_data
def get_tier_1_uaps():
    url = "https://raw.githubusercontent.com"
    try:
        df = pd.read_csv(url).dropna(subset=['latitude', 'longitude'])
        tier_1_df = df[df['duration (seconds)'] > 3600].copy() 
        tier_1_df = tier_1_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        tier_1_df['type'] = 'Tier 1 Sighting'
        return tier_1_df[['lat', 'lon', 'type']]
    except:
        return pd.DataFrame(columns=['lat', 'lon', 'type'])

# 3. Execution with Null-Checking
st.title("Tier 1 Guardian Correlation Map")

major_stress = get_major_triggers()
tier_1_uaps = get_tier_1_uaps()

# Only combine if data exists to avoid Concatenation Errors
if not major_stress.empty or not tier_1_uaps.empty:
    combined_map = pd.concat([major_stress, tier_1_uaps])
    st.map(combined_map, color='type')
else:
    st.info("Insufficient data to generate map.")
