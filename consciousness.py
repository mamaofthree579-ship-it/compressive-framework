import streamlit as st
import numpy as np
from scipy.stats import entropy

t = np.linspace(0,10,2000)

def wave(freq, K=0.6, gamma=0.02):
    drive = 1 + K*np.sin(2*np.pi*0.2*t)
    return np.exp(-gamma*t)*drive*np.exp(1j*2*np.pi*freq*t)

def ent(freq):
    ψ = wave(freq); P=np.abs(ψ)**2; P/=P.sum(); win=200
    return np.mean([entropy(P[i:i+win]) for i in range(0,len(P)-win,win)])

awake = ent(38.0)
sleep = ent(4.0)
st.write("Awake entropy (38 Hz):", awake)
st.write("Sleep entropy (4 Hz):", sleep)
