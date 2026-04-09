import streamlit as st
import requests
import pandas as pd
import datetime
import plotly.express as px

st.set_page_config(page_title="Terrestrial + Celestial Monitor", layout="wide")
st.title("🌍☄️ Unified Earth & Space Monitoring System")

NASA_API_KEY = st.secrets.get("NASA_API_KEY", "DEMO_KEY")

@st.cache_data(ttl=600)
def get_earthquakes(start_date, end_date):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "minmagnitude": 2.5
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"USGS API error: {e}")
        return pd.DataFrame()

    quakes = []
    for f in data["features"]:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        quakes.append({
            "place": props["place"],
            "mag": props["mag"],
            "time": pd.to_datetime(props["time"], unit="ms"),
            "lon": coords[0],
            "lat": coords[1],
            "depth": coords[2]
        })
    return pd.DataFrame(quakes)

@st.cache_data(ttl=600)
def get_asteroids(date):
    url = f"https://api.nasa.gov/neo/rest/v1/feed"
    params = {"start_date": date, "end_date": date, "api_key": NASA_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"NASA API error: {e}")
        return pd.DataFrame()

    asteroids = []
    for date_key in data.get("near_earth_objects", {}):
        for obj in data["near_earth_objects"][date_key]:
            asteroids.append({
                "name": obj["name"],
                "hazardous": obj["is_potentially_hazardous_asteroid"],
                "velocity": float(obj["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]),
                "miss_distance_km": float(obj["close_approach_data"][0]["miss_distance"]["kilometers"]),
                "diameter_m": obj["estimated_diameter"]["meters"]["estimated_diameter_max"]
            })
    return pd.DataFrame(asteroids)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    date_range = st.date_input("Date range",
                               [datetime.date.today() - datetime.timedelta(days=1), datetime.date.today()])
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]

with st.spinner("Loading data..."):
    eq_df = get_earthquakes(start_date, end_date)
    ast_df = get_asteroids(end_date) # NASA only does 1 day at a time easily

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌍 Avg Earthquake Magnitude", round(eq_df["mag"].mean(), 2) if not eq_df.empty else "N/A")
col2.metric("🔢 Total Quakes", len(eq_df))
col3.metric("☄️ Avg Asteroid Velocity", f"{int(ast_df['velocity'].mean()):,} km/h" if not ast_df.empty else "N/A")
col4.metric("⚠️ Hazardous NEOs", ast_df["hazardous"].sum() if not ast_df.empty else 0)

# Tabs for cleaner layout
tab1, tab2, tab3 = st.tabs(["Earthquakes", "Asteroids", "Risk Analysis"])

with tab1:
    st.subheader("🗺 Global Earthquake Map")
    if not eq_df.empty:
        fig = px.scatter_geo(eq_df, lat="lat", lon="lon", size="mag", color="depth",
                             hover_data=["place", "mag", "depth", "time"],
                             projection="natural earth")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(eq_df.sort_values("mag", ascending=False))
    else:
        st.info("No earthquakes ≥ M2.5 in selected range")

with tab2:
    st.subheader("☄️ Near-Earth Objects")
    if not ast_df.empty:
        fig2 = px.scatter(ast_df, x="miss_distance_km", y="velocity", color="hazardous",
                          size="diameter_m", hover_name="name", log_x=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(ast_df.sort_values("hazardous", ascending=False))
    else:
        st.info("No NEOs found for this date")

with tab3:
    mag_score = min(eq_df["mag"].max() / 10 * 10, 10) if not eq_df.empty else 0
    vel_score = min(ast_df["velocity"].mean() / 100000 * 10, 10) if not ast_df.empty else 0
    risk_score = round(mag_score * 0.6 + vel_score * 0.4, 1)

    st.metric("⚠️ Global Risk Score (0-10)", risk_score)
    st.progress(risk_score / 10)
    with st.expander("How is risk calculated?"):
        st.write("""
        Risk = 60% max earthquake magnitude + 40% avg asteroid velocity
        Both normalized to 0-10 scale. M10 quake = 10. 100,000 km/h avg velocity = 10.
        This is a demo model, not scientific.
        """)

    if risk_score > 5:
        st.error("🚨 HIGH RISK DETECTED – ALERT SYSTEM TRIGGERED")
        st.toast("High risk detected!", icon="🚨")
    else:
        st.success("✅ System Stable")
