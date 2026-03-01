import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.io import loadmat
import io
import pandas as pd

st.title("Consciousness Coherence Simulator")

st.sidebar.header("Parameters")
K = st.sidebar.slider("Coupling strength (K)", 0.1,1.5,0.6)
gamma = st.sidebar.slider("Decoherence rate (gamma)", 0.001,0.1,0.02)
drive_freq = st.sidebar.slider("Drive frequency (Hz)", 0.05,0.5,0.2)
fs = st.sidebar.slider("Sampling rate (Hz)", 1,500,100)

uploaded = st.file_uploader("Upload EEG (CSV, MAT, XLSX)", type=["csv","mat","xlsx"])

if uploaded:
    if uploaded.name.endswith(".mat"):
        mat = loadmat(uploaded)
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        eeg = mat[key].squeeze()
    elif uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded); eeg = df.iloc[:,0].values
    else:
        s = io.StringIO(uploaded.getvalue().decode("utf-8"))
        data = np.genfromtxt(s, delimiter=None, filling_values=0)
        eeg = data[:,0] if data.ndim > 1 else data
    t = np.arange(len(eeg))/fs
    eeg = eeg - np.mean(eeg)
else:
    fs = 100; t = np.linspace(0,10,2000); eeg = None

def wave(f,a,g):
    d = 1 + K*np.sin(2*np.pi*drive_freq*t)
    return a*np.exp(-g*t)*d*np.exp(1j*2*np.pi*f*t)

if eeg is not None:
    fft = np.abs(np.fft.rfft(eeg))
    freqs = np.fft.rfftfreq(len(eeg),1/fs)
    dom = abs(freqs[np.argmax(fft)]) or 38.0
    st.write("Dominant freq:",dom)
    psi_b = wave(dom,1.0,gamma)
else:
    psi_b = wave(38.0,1.0,gamma)

psi_h = wave(1.1,0.5,gamma*0.5)
psi_g = wave(0.12,0.2,gamma*0.2)

P = np.abs(psi_b*psi_h*psi_g)**2
P_norm = P/np.sum(P)
win = min(200,max(10,len(P_norm)//10))
ent = [entropy(P_norm[i:i+win]) for i in range(0,len(P_norm)-win,win)]
ent = np.array(ent)
ent_norm = (ent - ent.min())/(ent.max()-ent.min()) if ent.max()!=ent.min() else ent*0
t_ent = np.arange(len(ent_norm))*win/fs

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6,4))
ax1.plot(t,P); ax1.set_ylabel("Joint density")
ax2.plot(t_ent,ent_norm); ax2.set_ylabel("Entropy (norm)"); ax2.set_xlabel("Time (s)")
st.pyplot(fig)
st.caption("Low entropy dips ≈ moments of high coherence")
