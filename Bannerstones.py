import streamlit as st, numpy as np, math, matplotlib.pyplot as plt, pandas as pd
st.header("Geniculate bannerstone drill sim")
a1,b1,c1 = 3.5,1.5,1.5 # segment1
a2,b2,c2 = 2.5,1.2,1.2 # segment2
c = st.slider("Vertical tweak", 1.0,4.0,2.0)
hardness = st.selectbox("Stone", ["Quartzite","Granite","Slate"])
k = {"Quartzite":1e-4,"Granite":2e-4,"Slate":5e-4}[hardness]

density, torque, omega0 = 2.65, 0.05, 10
V = (4/3*math.pi*a1*b1*c1 + 4/3*math.pi*a2*b2*c2)
radii = np.linspace(0.5,1.5,50)
effs = []
for r in radii:
    V_hole = V - (1/3*math.pi*r**2*(a1+a2))
    m = V_hole*density
    I = (1/5)*m*((b1**2+c1**2)+(b2**2+c2**2))/2
    d = k*torque*(I*omega0/torque)
    effs.append(d/(2*math.pi*r*(a1+a2)))
fig, ax = plt.subplots()
ax.plot(radii, effs, 'g-')
for rad in [0.32,0.64,0.95]:
    ax.plot([rad,rad],[0,max(effs)],'r--')
ax.set_xlabel('Hole radius'); ax.set_ylabel('Efficiency')
st.pyplot(fig)
st.download_button('Export', pd.DataFrame({'r':radii,'eff':effs}).to_csv(index=False), 'geniculate.csv')
