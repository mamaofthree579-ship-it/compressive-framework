import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr
from scipy.io import loadmat
import io
import pandas as pd

st.title("Coherence vs EEG Entropy")

st.sidebar.header("Parameters")
K = st.sidebar.slider("Coupling strength (K)", 0.1,1.5,0.6)
gamma = st.sidebar.slider("Decoherence rate (gamma)", 0.001,0.1,0.02)
drive_freq = st.sidebar.slider("Drive frequency (Hz)", 0.05,0.5,0.2)
fs = st.sidebar.slider("Sampling rate (Hz)", 1,500,100)

uploaded = st.file_uploader("Upload EEG (CSV, MAT, XLSX)", type=["csv","mat","xlsx"])

if uploaded:
    if uploaded.name.endswith(".mat"):
        mat = loadmat(uploaded); key=[k for k in mat.keys() if not k.startswith("__")][0]
        eeg = mat[key].squeeze()
    elif uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded); eeg = df.iloc[:,0].values
    else:
        s = io.StringIO(uploaded.getvalue().decode("utf-8"))
        data = np.genfromtxt(s, delimiter=None, filling_values=0)
        eeg = data[:,0] if data.ndim>1 else data
    t = np.arange(len(eeg))/fs; eeg = eeg - np.mean(eeg)
else:
    st.stop()

def wave(f,a,g):
    d = 1 + K*np.sin(2*np.pi*drive_freq*t)
    return a*np.exp(-g*t)*d*np.exp(1j*2*np.pi*f*t)

def window_entropy(x):
    x = x + np.random.uniform(-1e-6,1e-6,len(x))
    hist,_ = np.histogram(x,bins=50,density=True)
    hist = hist[hist>0]
    return entropy(hist) if len(hist)>0 else 0

def window_entropy_nojit(x):
    hist,_ = np.histogram(x,bins=50,density=True)
    hist = hist[hist>0]
    return entropy(hist) if len(hist)>0 else 0

fft = np.abs(np.fft.rfft(eeg)); freqs = np.fft.rfftfreq(len(eeg),1/fs)
dom = abs(freqs[np.argmax(fft)]) or 38.0
psi_b = wave(dom,1.0,gamma); psi_h = wave(1.1,0.5,gamma*0.5); psi_g = wave(0.12,0.2,gamma*0.2)
P = np.abs(psi_b*psi_h*psi_g)**2; P_norm = P/np.sum(P)
win = min(200,max(10,len(P_norm)//10))
ent_theory = np.array([window_entropy(P_norm[i:i+win]) for i in range(0,len(P_norm)-win,win)])

eeg_norm = (eeg - eeg.min())/(eeg.max()-eeg.min()) if eeg.max()!=eeg.min() else eeg*0
ent_eeg = np.array([window_entropy_nojit(eeg_norm[i:i+win]) for i in range(0,len(eeg_norm)-win,win)])
n = min(len(ent_theory),len(ent_eeg))
if n>1:
    r,_ = pearsonr(ent_theory[:n],ent_eeg[:n])
    st.write("Correlation:", r)
t_ent = np.arange(n)*win/fs

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(t, eeg); ax[0].set_ylabel("EEG")
ax[1].plot(t_ent,ent_theory[:n],label="Theory"); ax[1].plot(t_ent,ent_eeg[:n],label="EEG")
ax[1].set_ylabel("Entropy"); ax[1].set_xlabel("Time (s)"); ax[1].legend()
st.pyplot(fig)
st.caption("Top: raw EEG; bottom: entropy fit; tune parameters to align theory")
