import numpy as np
from scipy.stats import entropy

def wave_from_freq(freq, t, K=0.6, gamma=0.02, f_drive=0.2):
    drive = 1 + K*np.sin(2*np.pi*f_drive*t)
    theta = 2*np.pi*freq*t
    return np.exp(-gamma*t)*drive*np.exp(1j*theta)

def entropy_from_trace(freq, t):
    ψ = wave_from_freq(freq, t)
    P = np.abs(ψ)**2
    P = P/np.sum(P)
    win=200
    return np.mean([entropy(P[i:i+win]) for i in range(0,len(P)-win,win)])

t = np.linspace(0,10,2000)
awake = entropy_from_trace(38.0, t)
sleep = entropy_from_trace(4.0, t)
st.write("38 Hz (awake) entropy:", awake)
st.write("4 Hz (sleep) entropy:", sleep)
