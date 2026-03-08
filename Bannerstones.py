import streamlit as st
import math

st.header("Bannerstone volume approx")
a = st.slider("Semi‑major axis (cm)", 4, 10, 6)
b = st.slider("Semi‑minor axis (cm)", 2, 6, 3)
c = st.slider("Vertical semi‑axis (cm)", 1, 4, 2)
r = st.slider("Hole radius (cm)", 0.5, 1.5, 0.8)

V_ellipsoid = 4/3 * math.pi * a * b * c
V_hole = 2 * (1/3 * math.pi * r**2 * a)
V = V_ellipsoid - V_hole
st.write(f"Volume ≈ {V:.1f} cm³")
