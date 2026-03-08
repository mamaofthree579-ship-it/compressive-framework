import streamlit as st
import numpy as np, math, matplotlib.pyplot as plt, pandas as pd

st.header("Bannerstone drill simulator")
a = st.slider("Semi‑major axis (cm)", 4.0, 10.0, 6.0)
b = st.slider("Semi‑minor axis (cm)", 2.0, 6.0, 3.0)
c = st.slider("Vertical semi‑axis (cm)", 1.0, 4.0, 2.0)
hardness = st.selectbox("Stone", ["Quartzite","Granite","Slate"])
k = {"Quartzite":1e-4,"Granite":2e-4,"Slate":5e-4}[hardness]

density, torque, omega0 = 2.65, 0.05, 10
V_ell = 4/3*math.pi*a*b*c
radii = np.linspace(0.5,1.5,50)
inertias, depths = [], []
for r in radii:
    V = V_ell - 2*(1/3*math.pi*r**2*a)
    m = V*density
    I = (1/5)*m*(b**2+c**2)
    inertias.append(I)
    depths.append(k*torque*(I*omega0/torque))

area = [2*math.pi*rad*a for rad in radii]
eff_area = [d/ar for d,ar in zip(depths, area)]

fig, ax = plt.subplots()
ax.plot(radii, eff_area, 'c-')
for rad in [0.32,0.64,0.95]:
    ax.plot([rad,rad],[0,max(eff_area)],'r--')
ax.text(0.64, max(eff_area)*0.9, "bannerstone radius range")
ax.set_xlabel('Hole radius (cm)')
ax.set_ylabel('Depth per area (mm/cm²)')
st.pyplot(fig)

Bs = np.linspace(2,6,20)
Cs = np.linspace(1,4,20)
heat = np.zeros((len(Bs),len(Cs)))
for i,bv in enumerate(Bs):
    for j,cv in enumerate(Cs):
        I = (1/5)*(V_ell-2/3*math.pi*0.64**2*a)*density*(bv**2+cv**2)
        heat[i,j] = (k*torque*(I*omega0/torque))/(2*math.pi*0.64*a)
fig,ax=plt.subplots()
im=ax.imshow(heat, origin='lower', extent=[1,4,2,6])
plt.colorbar(im, label='eff @ 0.64cm')
st.pyplot(fig)

df = pd.DataFrame({"radius":radii,"eff_area":eff_area})
st.download_button("Download data", df.to_csv(index=False), "efficiency.csv")
