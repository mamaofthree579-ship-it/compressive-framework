import streamlit as st
import pandas as pd
import requests
import io
from datetime import datetime, timedelta

# New Stable Data Source (CORGIS Mirror)
@st.cache_data
def get_filtered_uaps(months_back):
    # CORGIS ufo_sightings.csv is a reliable, smaller mirror of NUFORC
    url = "https://corgis-edu.github.io"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            # Use StringIO to handle the text stream directly
            df = pd.read_csv(io.StringIO(response.text))
            
            # Map CORGIS columns to your algorithm's keys
            # CORGIS uses 'Data.Date.Time' and 'Location.Latitude' / 'Location.Longitude'
            df = df.rename(columns={
                'Location.Latitude': 'lat',
                'Location.Longitude': 'lon',
                'Data.Date.Time': 'date_time'
            })
            
            # Clean and Filter
            df['date'] = pd.to_datetime(df['date_time'], errors='coerce')
            cutoff = datetime.now() - timedelta(days=30 * months_back)
            
            # In CORGIS, dates are historical; if 'recent' is empty, we'll show 'all'
            recent_df = df[df['date'] >= cutoff].copy()
            if recent_df.empty:
                st.warning("No recent sightings in this mirror; displaying historical Tier 1 nodes.")
                recent_df = df.head(500).copy() # Fallback to significant historical data
                
            recent_df['type'] = 'Guardian Node'
            return recent_df[['lat', 'lon', 'type', 'date']]
        else:
            st.error(f"Mirror Error: Status {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Connection Failed: {e}")
        return pd.DataFrame()

