import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

st.title("Consciousness Coherence Simulator")

# sidebar
st.sidebar.header("Parameters")
K = st.sidebar.slider("Coupling strength (K)", 0.1, 1.5, 0.6)
gamma = st.sidebar.slider("Decoherence rate (gamma)", 0.001, 0.1, 0.02)
drive_freq = st.sidebar.slider("Drive frequency (Hz)", 0.05, 0.5, 0.2)

uploaded = st.file_uploader("Upload EEG CSV (time,amp)", type="csv")

t = np.linspace(0,10,2000)
drive = 1 + K*np.sin(2*np.pi*drive_freq*t)

def wave(freq, amp, g):
    theta = 2*np.pi*freq*t
    return amp*np.exp(-g*t)*drive*np.exp(1j*theta)

if uploaded:
    data = np.loadtxt(uploaded, delimiter=",")
    t = data[:,0]
    eeg = data[:,1]
    fft = np.abs(np.fft.rfft(eeg))
    freqs = np.fft.rfftfreq(len(eeg), t[1]-t[0])
    dom = freqs[np.argmax(fft)]
    st.write("Dominant freq from upload:", dom)
    psi_b = wave(dom, 1.0, gamma)
else:
    psi_b = wave(38.0, 1.0, gamma) # preloaded awake default

psi_h = wave(1.1, 0.5, gamma*0.5)
psi_g = wave(0.12, 0.2, gamma*0.2)

P = np.abs(psi_b*psi_h*psi_g)**2
P_norm = P/np.sum(P)
win=200
ent = [entropy(P_norm[i:i+win]) for i in range(0,len(P_norm)-win,win)]
t_ent = np.arange(len(ent))*win*(t[1]-t[0])

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6,4))
ax1.plot(t,P); ax1.set_ylabel("Joint density")
ax2.plot(t_ent,ent); ax2.set_ylabel("Entropy"); ax2.set_xlabel("Time (s)")
st.pyplot(fig)
st.caption("Low entropy dips ≈ moments of high coherence (toy consciousness events).")
