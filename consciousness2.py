import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.io import loadmat
from scipy.signal import hilbert
import io
import pandas as pd

st.title("Coherence vs EEG Dynamics")

st.sidebar.header("Parameters")
K = st.sidebar.slider("Coupling strength (K)", 0.1,1.5,0.6)
gamma = st.sidebar.slider("Decoherence rate (gamma)", 0.001,0.1,0.02)
drive_freq = st.sidebar.slider("Drive frequency (Hz)", 0.05,0.5,0.2)
fs = 100

uploaded = st.file_uploader("Upload EEG (CSV, MAT, XLSX)", type=["csv","mat","xlsx"])

if uploaded:
    if uploaded.name.endswith(".mat"):
        mat = loadmat(uploaded); key=[k for k in mat.keys() if not k.startswith("__")][0]
        eeg_raw = mat[key].squeeze()
        eeg = eeg_raw[:,1:].mean(axis=1) if eeg_raw.ndim>1 else eeg_raw
    elif uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded); eeg = df.values[:,1:].mean(axis=1) if df.shape[1]>1 else df.values.mean(axis=1)
    else:
        s = io.StringIO(uploaded.getvalue().decode("utf-8"))
        data = np.genfromtxt(s, delimiter=None, filling_values=0)
        eeg = data[:,1:].mean(axis=1) if data.ndim>1 else data
    t = np.arange(len(eeg))/fs
    raw_eeg = eeg.copy()
    eeg = eeg - np.mean(eeg)
else:
    st.stop()

def wave(f,a,g):
    d = 1 + K*np.sin(2*np.pi*drive_freq*t)
    return a*np.exp(-g*t)*d*np.exp(1j*2*np.pi*f*t)

def window_feats(x):
    p = np.mean(x**2)
    env = np.mean(np.abs(hilbert(x))) if len(x)>2 else 0
    fx = np.abs(np.fft.rfft(x)); fx_freqs = np.fft.rfftfreq(len(x),1/fs)
    dom = fx_freqs[np.argmax(fx)] if len(fx)>0 else 0
    return env * p * dom

fft = np.abs(np.fft.rfft(eeg)); freqs = np.fft.rfftfreq(len(eeg),1/fs)
dom = abs(freqs[np.argmax(fft)]) or 38.0
psi_b = wave(dom,1.0,gamma); psi_h = wave(1.1,0.5,gamma*0.5); psi_g = wave(0.12,0.2,gamma*0.2)
P = np.abs(psi_b*psi_h*psi_g)**2; P_norm = P/np.sum(P)
win = min(200,max(10,len(P_norm)//10))
dyn_theory = np.array([window_feats(P_norm[i:i+win]) for i in range(0,len(P_norm)-win,win)])
dyn_eeg = np.array([window_feats(raw_eeg[i:i+win]) for i in range(0,len(raw_eeg)-win,win)])
n = min(len(dyn_theory),len(dyn_eeg))
def norm(a):
    return (a - a.min())/(a.max()-a.min()) if a.max()!=a.min() else a*0
dyn_t_n = norm(dyn_theory[:n]); dyn_e_n = norm(dyn_eeg[:n])
if n>1:
    r,_ = pearsonr(dyn_t_n, dyn_e_n)
    st.write("Correlation (shape):", r)
t_ent = np.arange(n)*win/fs

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(t, raw_eeg); ax[0].set_ylabel("EEG avg")
ax[1].plot(t_ent,dyn_t_n,label="Theory coherence"); ax[1].plot(t_ent,dyn_e_n,label="EEG coherence")
ax[1].set_ylabel("Norm coherence"); ax[1].set_xlabel("Time (s)"); ax[1].legend()
st.pyplot(fig)
st.caption("Coherence = envelope × power × dominant freq per window; tune K, gamma, drive_freq to fit")
