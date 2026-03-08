import streamlit as st, numpy as np, math, matplotlib.pyplot as plt, pandas as pd
st.header("Butterfly bannerstone drill sim")
a, b = 8.0, 2.0 # elongated wing shape
c = st.slider("Vertical semi‑axis (cm)", 1.0,4.0,2.0)
hardness = st.selectbox("Stone", ["Quartzite","Granite","Slate"])
k = {"Quartzite":1e-4,"Granite":2e-4,"Slate":5e-4}[hardness]

density, torque, omega0 = 2.65, 0.05, 10
V_ell = 4/3*math.pi*a*b*c
radii = np.linspace(0.5,1.5,50)
areas = [2*math.pi*r*a for r in radii]
effs = []
for r in radii:
    V = V_ell - 2*(1/3*math.pi*r**2*a)
    m = V*density
    I = (1/5)*m*(b**2+c**2)
    d = k*torque*(I*omega0/torque)
    effs.append(d/(2*math.pi*r*a))

fig, ax = plt.subplots()
ax.plot(radii, effs, 'c-')
for rad in [0.32,0.64,0.95]:
    ax.plot([rad,rad],[0,max(effs)],'r--')
ax.set_xlabel("Hole radius (cm)")
ax.set_ylabel("Depth per area")
st.pyplot(fig)

st.download_button("Export", pd.DataFrame({"r":radii,"eff":effs}).to_csv(index=False), "butterfly.csv")
