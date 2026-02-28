import streamlit as st
import numpy as np
from scipy.stats import entropy

t = np.linspace(0,10,2000)
K = st.sidebar.slider("Coupling K", 0.1,1.5,0.6)
gamma = st.sidebar.slider("Decoherence γ", 0.001,0.1,0.02)

drive = 1 + K*np.sin(2*np.pi*0.2*t)

def wave(freq):
    return np.exp(-gamma*t)*drive*np.exp(1j*2*np.pi*freq*t)

# single oscillator entropy
ψ = wave(38.0)
P=np.abs(ψ)**2; P/=P.sum()
win=200
ent = [entropy(P[i:i+win]) for i in range(0,len(P)-win,win)]
st.subheader("Entropy")
st.line_chart(ent)

# network PLV
freqs = 38 + np.random.randn(10)*0.5
waves = np.array([wave(f) for f in freqs])
amp = np.abs(waves) + 1e-8      # avoid zero-division
phases = waves/amp
plv = np.abs(phases.mean(axis=0))
st.line_chart(plv)
