import streamlit as st
import requests
import pandas as pd
import datetime
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(page_title="Terrestrial + Celestial Monitor", layout="wide")
st.title("🌍☄️ Unified Earth & Space Monitoring System")

NASA_API_KEY = st.secrets.get("NASA_API_KEY", "DEMO_KEY")
EMAIL_ENABLED = "EMAIL_SENDER" in st.secrets

# -----------------------------
# DATA FETCH FUNCTIONS
# -----------------------------
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
            "date": pd.to_datetime(props["time"], unit="ms").date(),
            "lon": coords[0],
            "lat": coords[1],
            "depth": coords[2]
        })
    return pd.DataFrame(quakes)

@st.cache_data(ttl=600)
def get_asteroids_for_range(start_date, end_date):
    """NASA NeoWs supports max 7 days per call"""
    url = f"https://api.nasa.gov/neo/rest/v1/feed"
    params = {"start_date": start_date, "end_date": end_date, "api_key": NASA_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=15)
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
                "date": pd.to_datetime(date_key).date(),
                "hazardous": obj["is_potentially_hazardous_asteroid"],
                "velocity": float(obj["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]),
                "miss_distance_km": float(obj["close_approach_data"][0]["miss_distance"]["kilometers"]),
                "diameter_m": obj["estimated_diameter"]["meters"]["estimated_diameter_max"]
            })
    return pd.DataFrame(asteroids)

# -----------------------------
# EMAIL ALERT FUNCTION
# -----------------------------
def send_email_alert(risk_score, eq_df, ast_df):
    if not EMAIL_ENABLED:
        return False, "Email not configured in secrets"

    try:
        sender = st.secrets["EMAIL_SENDER"]
        password = st.secrets["EMAIL_PASSWORD"]
        recipients = st.secrets["EMAIL_RECIPIENTS"] # list of emails

        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"🚨 HIGH RISK ALERT: Score {risk_score}/10"

        max_quake = eq_df.loc[eq_df['mag'].idxmax()] if not eq_df.empty else None
        hazardous_count = ast_df["hazardous"].sum() if not ast_df.empty else 0

        body = f"""
        High risk conditions detected at {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

        Risk Score: {risk_score}/10

        Earthquake Summary:
        - Total quakes ≥M2.5: {len(eq_df)}
        - Max magnitude: {max_quake['mag']} near {max_quake['place']} if max_quake is not None else 'N/A'

        Asteroid Summary:
        - NEOs tracked today: {len(ast_df)}
        - Potentially hazardous: {hazardous_count}

        View dashboard for details.
        """
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        return True, "Alert email sent"
    except Exception as e:
        return False, f"Email failed: {e}"

# -----------------------------
# SIDEBAR + DATE LOGIC
# -----------------------------
with st.sidebar:
    st.header("Controls")
    today = datetime.date.today()
    date_range = st.date_input("Select date range",
                               [today - datetime.timedelta(days=1), today],
                               max_value=today)
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range[0]

    st.divider()
    st.subheader("Alert Settings")
    risk_threshold = st.slider("Alert threshold", 0.0, 10.0, 5.0, 0.5)
    enable_email = st.toggle("Enable email alerts", value=EMAIL_ENABLED)
    if enable_email and not EMAIL_ENABLED:
        st.warning("Add EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENTS to secrets.toml")

# -----------------------------
# LOAD DATA
# -----------------------------
with st.spinner("Loading data..."):
    eq_df = get_earthquakes(start_date, end_date)
    # NASA only allows 7 days max per call, so we cap it
    neo_start = max(start_date, end_date - datetime.timedelta(days=6))
    ast_df = get_asteroids_for_range(neo_start, end_date)

    # 7-day trend data - always last 7 days regardless of picker
    trend_start = today - datetime.timedelta(days=6)
    eq_trend_df = get_earthquakes(trend_start, today)
    ast_trend_df = get_asteroids_for_range(trend_start, today)

# -----------------------------
# METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌍 Avg Earthquake Magnitude", round(eq_df["mag"].mean(), 2) if not eq_df.empty else "N/A")
col2.metric("🔢 Total Quakes", len(eq_df))
col3.metric("☄️ Avg Asteroid Velocity", f"{int(ast_df['velocity'].mean()):,} km/h" if not ast_df.empty else "N/A")
col4.metric("⚠️ Hazardous NEOs", ast_df["hazardous"].sum() if not ast_df.empty else 0)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Earthquakes", "Asteroids", "7-Day Trends", "Risk Analysis"])

with tab1:
    st.subheader("🗺 Global Earthquake Map")
    if not eq_df.empty:
        fig = px.scatter_geo(eq_df, lat="lat", lon="lon", size="mag", color="depth",
                             hover_data=["place", "mag", "depth", "time"],
                             projection="natural earth")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(eq_df.sort_values("mag", ascending=False), use_container_width=True)
    else:
        st.info("No earthquakes ≥ M2.5 in selected range")

with tab2:
    st.subheader("☄️ Near-Earth Objects")
    if not ast_df.empty:
        fig2 = px.scatter(ast_df, x="miss_distance_km", y="velocity", color="hazardous",
                          size="diameter_m", hover_name="name", log_x=True,
                          labels={"hazardous": "Potentially Hazardous"})
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(ast_df.sort_values("hazardous", ascending=False), use_container_width=True)
    else:
        st.info("No NEOs found for this date range")

with tab3:
    st.subheader("📈 7-Day Trend Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        if not eq_trend_df.empty:
            daily_quakes = eq_trend_df.groupby("date").agg(
                avg_mag=("mag", "mean"),
                count=("mag", "count")
            ).reset_index()
            fig_trend1 = px.line(daily_quakes, x="date", y="avg_mag", markers=True,
                                 title="Average Magnitude by Day",
                                 labels={"avg_mag": "Avg Magnitude", "date": "Date"})
            st.plotly_chart(fig_trend1, use_container_width=True)
        else:
            st.info("No earthquake trend data")

    with col_b:
        if not ast_trend_df.empty:
            daily_neos = ast_trend_df.groupby("date").agg(
                neo_count=("name", "count"),
                hazardous_count=("hazardous", "sum")
            ).reset_index()
            fig_trend2 = px.bar(daily_neos, x="date", y="neo_count",
                                title="NEOs per Day",
                                labels={"neo_count": "NEO Count", "date": "Date"})
            fig_trend2.add_scatter(x=daily_neos["date"], y=daily_neos["hazardous_count"],
                                   mode="lines+markers", name="Hazardous")
            st.plotly_chart(fig_trend2, use_container_width=True)
        else:
            st.info("No NEO trend data")

with tab4:
    mag_score = min(eq_df["mag"].max() / 10 * 10, 10) if not eq_df.empty else 0
    vel_score = min(ast_df["velocity"].mean() / 100000 * 10, 10) if not ast_df.empty else 0
    risk_score = round(mag_score * 0.6 + vel_score * 0.4, 1)

    st.metric("⚠️ Global Risk Score (0-10)", risk_score)
    st.progress(risk_score / 10)

    if risk_score > risk_threshold:
        st.error(f"🚨 HIGH RISK DETECTED – Score {risk_score} > {risk_threshold}")
        st.toast("High risk detected!", icon="🚨")

        if enable_email and EMAIL_ENABLED:
            if "last_email_sent" not in st.session_state or \
               (datetime.datetime.now() - st.session_state.last_email_sent).seconds > 3600: # 1hr cooldown
                success, msg = send_email_alert(risk_score, eq_df, ast_df)
                if success:
                    st.success(msg)
                    st.session_state.last_email_sent = datetime.datetime.now()
                else:
                    st.warning(msg)
            else:
                st.caption("Email cooldown active – prevents spam")
    else:
        st.success("✅ System Stable")

    with st.expander("How is risk calculated?"):
        st.write("""
        **Risk = 60% max earthquake magnitude + 40% avg asteroid velocity**
        Both normalized to 0-10 scale. M10 quake = 10. 100,000 km/h avg velocity = 10.
        This is a demo model, not scientific. Adjust threshold in sidebar.
        """)
