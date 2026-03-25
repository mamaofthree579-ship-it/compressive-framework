import streamlit as st
import pandas as pd
import requests

st.title("Tier 1 Guardian Correlation Map")

# 1. Fetch Major Seismic Triggers (Sp > 6.0)
def get_major_triggers():
    # USGS API for Mag 6.0+ in the last 30 days
    url = "https://earthquake.usgs.gov"
    resp = requests.get(url).json()
    return pd.DataFrame([{'lat': f['geometry'][1], 'lon': f['geometry'][0], 'type': 'Major Stress'} for f in resp['features']])

# 2. Filter for Tier 1 Historical Sightings
@st.cache_data
def get_tier_1_uaps():
    url = "https://raw.githubusercontent.com"
    df = pd.read_csv(url).dropna(subset=['latitude', 'longitude'])
    # Filtering for 'Tier 1' indicators: long duration or physical descriptions
    tier_1_df = df[df['duration (seconds)'] > 3600].copy() 
    tier_1_df = tier_1_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    tier_1_df['type'] = 'Tier 1 Sighting'
    return tier_1_df[['lat', 'lon', 'type']]

# 3. Execution & Mapping
major_stress = get_major_triggers()
tier_1_uaps = get_tier_1_uaps()
combined_map = pd.concat([major_stress, tier_1_uaps])

st.map(combined_map, color='type')
st.write("🟡 **Yellow:** Tier 1 Historical Manifestations")
st.write("🔴 **Red:** Major Live Seismic Triggers")
