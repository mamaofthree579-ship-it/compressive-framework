import streamlit as st
import numpy as np, math, matplotlib.pyplot as plt

st.header("Bannerstone drill simulator")
a = st.slider("Semi‑major axis (cm)", 4.0, 10.0, 6.0)
b = st.slider("Semi‑minor axis (cm)", 2.0, 6.0, 3.0)
c = st.slider("Vertical semi‑axis (cm)", 1.0, 4.0, 2.0)
hardness = st.selectbox("Stone", ["Quartzite","Granite","Slate"])
k = {"Quartzite":1e-4,"Granite":2e-4,"Slate":5e-4}[hardness]

density, torque, omega0 = 2.65, 0.05, 10
V_ell = 4/3*math.pi*a*b*c
radii = np.linspace(0.5,1.5,50)
inertias, spin_times, depths = [], [], []
for r in radii:
    V = V_ell - 2*(1/3*math.pi*r**2*a)
    m = V*density
    I = (1/5)*m*(b**2+c**2)
    inertias.append(I)
    t = I*omega0/torque
    spin_times.append(t)
    depths.append(k*torque*t)

fig, ax1 = plt.subplots()
ax1.plot(radii, inertias, 'b-', label='Inertia')
ax1.set_xlabel('Hole radius (cm)')
ax1.set_ylabel('Inertia (g·cm²)', color='b')
ax2 = ax1.twinx()
ax2.plot(radii, depths, 'g-', label='Depth/spin')
ax2.set_ylabel('Est. depth/spin (mm)', color='g')
st.pyplot(fig)

eff = [d/i for d,i in zip(depths, inertias)]
fig2, ax = plt.subplots()
ax.plot(radii, eff, 'm-')
ax.set_xlabel('Hole radius (cm)')
ax.set_ylabel('Depth/Inertia (mm per g·cm²)')
st.pyplot(fig2)

area = [2*math.pi*rad*a for rad in radii]
eff2 = [d/ar for d,ar in zip(depths, area)]
fig3, ax = plt.subplots()
ax.plot(radii, eff2, 'c-')
ax.set_xlabel('Hole radius (cm)')
ax.set_ylabel('Depth per area (mm/cm²)')
st.pyplot(fig3)

import pandas as pd
df = pd.DataFrame({
    "radius": radii,
    "inertia": inertias,
    "depth_per_spin": depths,
    "efficiency": eff
})
st.download_button("Download data", df.to_csv(index=False), "bannerstone_data.csv")
