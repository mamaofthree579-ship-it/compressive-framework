import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.io import loadmat
import io
import pandas as pd

st.title("Consciousness Coherence vs EEG")

st.sidebar.header("Parameters")
K = st.sidebar.slider("Coupling strength (K)", 0.1,1.5,0.6)
gamma = st.sidebar.slider("Decoherence rate (gamma)", 0.001,0.1,0.02)
drive_freq = st.sidebar.slider("Drive frequency (Hz)", 0.05,0.5,0.2)

t = np.linspace(0,10,2000)
drive = 1 + K*np.sin(2*np.pi*drive_freq*t)

def wave(freq, amp, g):
    return amp*np.exp(-g*t)*drive*np.exp(1j*2*np.pi*freq*t)

psi_b = wave(38.0,1.0,gamma)
psi_h = wave(1.1,0.5,gamma*0.5)
psi_g = wave(0.12,0.2,gamma*0.2)
P = np.abs(psi_b*psi_h*psi_g)**2
P_norm = P/np.sum(P)
win = 200
ent_theory = [entropy(P_norm[i:i+win]) for i in range(0,len(P_norm)-win,win)]
t_ent = np.arange(len(ent_theory))*win*(t[1]-t[0])

uploaded = st.file_uploader("Upload EEG (CSV, MAT, XLSX)", type=["csv","mat","xlsx"])
if uploaded:
    if uploaded.name.endswith(".mat"):
        mat = loadmat(uploaded); key=[k for k in mat.keys() if not k.startswith("__")][0]
        eeg_raw = mat[key].squeeze(); eeg = eeg_raw[:,1:].mean(axis=1) if eeg_raw.ndim>1 else eeg_raw
    elif uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded); eeg = df.values[:,1:].mean(axis=1) if df.shape[1]>1 else df.values.mean(axis=1)
    else:
        s = io.StringIO(uploaded.getvalue().decode("utf-8"))
        data = np.genfromtxt(s, delimiter=None, filling_values=0)
        eeg = data[:,1:].mean(axis=1) if data.ndim>1 else data
    eeg = eeg - np.mean(eeg)
    P_eeg = np.abs(np.fft.rfft(eeg))**2; P_eeg /= P_eeg.sum()
    ent_eeg = [entropy(P_eeg[i:i+win]) for i in range(0,len(P_eeg)-win,win)]
    t_eeg = np.arange(len(ent_eeg))*win*(t[1]-t[0])
else:
    ent_eeg = []; t_eeg = []

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,4))
ax1.plot(t, P); ax1.set_ylabel("Joint density")
ax2.plot(t_ent, ent_theory, label="Theory"); 
if ent_eeg: ax2.plot(t_eeg, ent_eeg, label="EEG")
ax2.set_ylabel("Entropy"); ax2.set_xlabel("Time (s)"); ax2.legend()
st.pyplot(fig)
st.caption("Low entropy ≈ coherence; compare theory dips to EEG")
