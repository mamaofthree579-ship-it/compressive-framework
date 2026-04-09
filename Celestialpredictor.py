import streamlit as st
import requests
import pandas as pd
import datetime
import plotly.express as px

# -----------------------------
# CONFIG
# -----------------------------
NASA_API_KEY = "DEMO_KEY"

st.set_page_config(page_title="Terrestrial + Celestial Monitor", layout="wide")

st.title("🌍☄️ Unified Earth & Space Monitoring System")

# -----------------------------
# FETCH USGS EARTHQUAKE DATA
# -----------------------------
def get_earthquakes():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "minmagnitude": 2.5
    }
    response = requests.get(url, params=params)
    data = response.json()

    quakes = []
    for f in data["features"]:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        quakes.append({
            "place": props["place"],
            "mag": props["mag"],
            "time": props["time"],
            "lon": coords[0],
            "lat": coords[1],
            "depth": coords[2]
        })
    return pd.DataFrame(quakes)

# -----------------------------
# FETCH NASA ASTEROID DATA
# -----------------------------
def get_asteroids():
    today = datetime.date.today()
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={today}&end_date={today}&api_key={NASA_API_KEY}"
    response = requests.get(url)
    data = response.json()

    asteroids = []
    for date in data["near_earth_objects"]:
        for obj in data["near_earth_objects"][date]:
            asteroids.append({
                "name": obj["name"],
                "hazardous": obj["is_potentially_hazardous_asteroid"],
                "velocity": float(obj["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]),
                "miss_distance_km": float(obj["close_approach_data"][0]["miss_distance"]["kilometers"])
            })
    return pd.DataFrame(asteroids)

# -----------------------------
# LOAD DATA
# -----------------------------
eq_df = get_earthquakes()
ast_df = get_asteroids()

# -----------------------------
# RISK CALCULATION
# -----------------------------
def calculate_risk(eq_df, ast_df):
    quake_score = eq_df["mag"].mean() if not eq_df.empty else 0
    asteroid_score = ast_df["velocity"].mean() / 10000 if not ast_df.empty else 0

    risk = (quake_score * 0.7) + (asteroid_score * 0.3)
    return round(risk, 2)

risk_score = calculate_risk(eq_df, ast_df)

# -----------------------------
# DISPLAY METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🌍 Avg Earthquake Magnitude", round(eq_df["mag"].mean(), 2) if not eq_df.empty else 0)
col2.metric("☄️ Avg Asteroid Velocity", int(ast_df["velocity"].mean()) if not ast_df.empty else 0)
col3.metric("⚠️ Global Risk Score", risk_score)

# -----------------------------
# MAP
# -----------------------------
st.subheader("🗺 Global Earthquake Map")

if not eq_df.empty:
    fig = px.scatter_geo(
        eq_df,
        lat="lat",
        lon="lon",
        size="mag",
        hover_name="place",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ASTEROID TABLE
# -----------------------------
st.subheader("☄️ Near-Earth Objects")
st.dataframe(ast_df)

# -----------------------------
# ALERT SYSTEM (SIMULATED)
# -----------------------------
if risk_score > 5:
    st.error("🚨 HIGH RISK DETECTED – ALERT SYSTEM TRIGGERED")
else:
    st.success("✅ System Stable")
