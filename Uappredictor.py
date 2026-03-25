import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

st.title("Guardian Verification: Seismic vs. Historical UAP Map")

# 1. Fetch Live Seismic Data (Sp)
def get_live_seismic():
    url = "https://earthquake.usgs.gov"
    resp = requests.get(url).json()
    points = [{'lat': f['geometry'][1], 'lon': f['geometry'][0], 'type': 'Seismic Stress'} for f in resp['features']]
    return pd.DataFrame(points)

# 2. Fetch Historical UAP Data (NUFORC)
@st.cache_data
def get_historical_uaps():
    # Loading a cleaned sample dataset from a common public repo
    url = "https://raw.githubusercontent.com/DataHerb/nuforc-ufo-records/master/dataset/nuforc_ufo_records.csv"
    df = pd.read_csv(url).dropna(subset=['latitude', 'longitude'])
    df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    df['type'] = 'Historical UAP'
    return df[['lat', 'lon', 'type']].sample(500) # Sample for performance

# 3. Combine and Map
seismic_df = get_live_seismic()
uap_df = get_historical_uaps()
combined_df = pd.concat([seismic_df, uap_df])

st.subheader("Global Grid Overlap")
st.map(combined_df, color='type', size=20)

# Legend Logic
st.write("🔴 **Red Dots:** Live Seismic Stress (Sp)")
st.write("🔵 **Blue Dots:** Historical UAP Reports")
