import streamlit as st
import numpy as np, math, matplotlib.pyplot as plt

a,b,c = 6,3,2
density = 2.65
V_ell = 4/3*math.pi*a*b*c
torque = 0.05
omega0 = 10

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
