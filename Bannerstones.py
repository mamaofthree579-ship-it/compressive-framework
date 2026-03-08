import streamlit as st
import numpy as np, math, matplotlib.pyplot as plt

st.header("Bannerstone flywheel explorer")
a = st.slider("Semi‑major axis (cm)", 4.0, 10.0, 6.0)
b = st.slider("Semi‑minor axis (cm)", 2.0, 6.0, 3.0)
c = st.slider("Vertical semi‑axis (cm)", 1.0, 4.0, 2.0)
density = 2.65
torque = 0.05
omega0 = 10

V_ell = 4/3*math.pi*a*b*c
radii = np.linspace(0.5,1.5,50)
inertias, spin_times = [], []
for r in radii:
    V = V_ell - 2*(1/3*math.pi*r**2*a)
    m = V*density
    I = (1/5)*m*(b**2+c**2)
    inertias.append(I)
    spin_times.append(I*omega0/torque)

fig, ax1 = plt.subplots()
ax1.plot(radii, inertias, 'b-', label='Inertia')
ax1.set_xlabel('Hole radius (cm)')
ax1.set_ylabel('Inertia (g·cm²)', color='b')
ax2 = ax1.twinx()
ax2.plot(radii, spin_times, 'r-', label='Spin‑down (s)')
ax2.set_ylabel('Spin‑down time (s)', color='r')
st.pyplot(fig)

hardness = st.selectbox("Stone hardness", ["Quartzite", "Granite", "Slate"], index=0)
k = {"Quartzite":1e-4, "Granite":2e-4, "Slate":5e-4}[hardness]
depths = [k*torque*t for t in spin_times]

fig, ax = plt.subplots()
ax.plot(radii, depths, 'g-')
ax.set_xlabel('Hole radius (cm)')
ax.set_ylabel('Est. depth per spin (mm)')
st.pyplot(fig)
