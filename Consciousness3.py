import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Gut-Heart-Brain Wave Intersection")

st.sidebar.header("Params")
f_gut = st.sidebar.slider("Gut freq (Hz)", 0.05, 0.3, 0.12)
f_heart = st.sidebar.slider("Heart freq (Hz)", 0.8, 1.5, 1.1)
f_brain = st.sidebar.slider("Brain freq (Hz)", 30.0, 45.0, 38.0)
K = st.sidebar.slider("Coupling", 0.1,1.5,0.6)
gamma = st.sidebar.slider("Decoherence", 0.001,0.1,0.02)

t = np.linspace(0,10,2000)
drive = 1 + K*np.sin(2*np.pi*0.2*t)

gut = np.exp(-gamma*t)*drive*np.exp(1j*2*np.pi*f_gut*t)
heart = np.exp(-gamma*0.5*t)*drive*np.exp(1j*2*np.pi*f_heart*t)
brain = np.exp(-gamma*t)*drive*np.exp(1j*2*np.pi*f_brain*t)

joint = gut*heart*brain
env = np.abs(joint)
phase = np.angle(joint)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,4))
ax1.plot(t, env); ax1.set_ylabel("Joint envelope")
ax2.plot(t, phase); ax2.set_ylabel("Joint phase"); ax2.set_xlabel("Time (s)")
st.pyplot(fig)
st.caption("Gut-heart-brain product; envelope shows intersection coherence, phase gives internal frequency")
