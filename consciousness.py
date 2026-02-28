import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

st.title("Consciousness Coherence Simulator")
st.sidebar.header("Parameters")
K = st.sidebar.slider("Coupling strength (K)", 0.1, 1.5, 0.6)
gamma = st.sidebar.slider("Decoherence rate (gamma)", 0.001, 0.1, 0.02)
drive_freq = st.sidebar.slider("Drive frequency (Hz)", 0.05, 0.5, 0.2)

t = np.linspace(0, 10, 2000)
drive = 1 + K*np.sin(2*np.pi*drive_freq*t)

def wave(freq, amp, g):
    theta = 2*np.pi*freq*t
    return amp*np.exp(-g*t)*drive*np.exp(1j*theta)

psi_b = wave(38.0, 1.0, gamma)
psi_h = wave(1.1, 0.5, gamma*0.5)
psi_g = wave(0.12, 0.2, gamma*0.2)

P = np.abs(psi_b*psi_h*psi_g)**2
P_norm = P/np.sum(P)

win = 200
ent = [entropy(P_norm[i:i+win]) for i in range(0, len(P_norm)-win, win)]
t_ent = np.arange(len(ent))*win*(t[1]-t[0])

# pseudo-code; replace with neurodatasets.load(...)
eeg_awake = np.sin(2*np.pi*38*t) + 0.2*np.random.randn(len(t))
eeg_sleep = np.sin(2*np.pi*4*t) + 0.4*np.random.randn(len(t))  # slow delta

def sim_from_trace(trace):
    ω = freq_from_fft(trace)        # fit dominant freq
    drive = 1 + 0.6*np.sin(2*np.pi*0.2*t)  # breath drive
    ψ = wave(ω,1.0,0.02)           # using earlier wave()
    P = np.abs(ψ)**2; P/=P.sum()
    return np.mean([entropy(P[i:i+200]) for i in range(0,len(P)-200,200)])

print("Awake entropy:", sim_from_trace(eeg_awake))
print("Sleep entropy:", sim_from_trace(eeg_sleep))
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6,4))
ax1.plot(t, P)
ax1.set_ylabel("Joint density")
ax2.plot(t_ent, ent)
ax2.set_ylabel("Entropy")
ax2.set_xlabel("Time (s)")
st.pyplot(fig)

st.caption("Low entropy dips ≈ moments of high coherence (toy consciousness events).")
