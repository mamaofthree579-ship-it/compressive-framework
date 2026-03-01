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
fs = st.sidebar.slider("Sampling rate (Hz)", 1,500,100)

uploaded = st.file_uploader("Upload EEG (CSV, MAT, XLSX)", type=["csv","mat","xlsx"])

if uploaded:
    if uploaded.name.endswith(".mat"):
        mat = loadmat(uploaded); key=[k for k in mat.keys() if not k.startswith("__")][0]
        data = mat[key].squeeze()
    elif uploaded.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded); data = df.values
    else:
        s = io.StringIO(uploaded.getvalue().decode("utf-8"))
        data = np.genfromtxt(s, delimiter=None, filling_values=0)
    eeg1 = data[:,0] if data.ndim>1 else data
    eeg2 = data[:,1] if data.ndim>1 and data.shape[1]>1 else None
    t = np.arange(len(eeg1))/fs
    eeg1 = eeg1 - np.mean(eeg1); eeg1_norm = (eeg1-eeg1.min())/(eeg1.max()-eeg1.min()) if eeg1.max()!=eeg1.min() else np.zeros_like(eeg1)
    if eeg2 is not None:
        eeg2 = eeg2 - np.mean(eeg2); eeg2_norm = (eeg2-eeg2.min())/(eeg2.max()-eeg2.min()) if eeg2.max()!=eeg2.min() else np.zeros_like(eeg2)
    else:
        eeg2_norm = np.zeros_like(eeg1)
else:
    fs=100; t=np.linspace(0,10,2000); eeg1_norm=np.zeros_like(t); eeg2_norm=np.zeros_like(t)

def wave(f,a,g,norm):
    d = 1 + K*norm
    return a*np.exp(-g*t)*d*np.exp(1j*2*np.pi*f*t)

if uploaded:
    fft1=np.abs(np.fft.rfft(eeg1)); freqs=np.fft.rfftfreq(len(eeg1),1/fs)
    dom1 = abs(freqs[np.argmax(fft1)]) or 38.0
    if eeg2 is not None:
        fft2=np.abs(np.fft.rfft(eeg2)); dom2=abs(freqs[np.argmax(fft2)]) or 38.0
        st.write("Dominant freqs:",dom1,dom2)
        psi_b1=wave(dom1,1.0,gamma,eeg1_norm); psi_b2=wave(dom2,1.0,gamma,eeg2_norm)
        plv = np.abs(np.mean(np.exp(1j*(np.angle(psi_b1)-np.angle(psi_b2)))))
        st.write("PLV:",plv)
    else:
        psi_b1=wave(dom1,1.0,gamma,eeg1_norm)
else:
    psi_b1=wave(38.0,1.0,gamma,eeg1_norm)

psi_h = wave(1.1,0.5,gamma*0.5,eeg1_norm); psi_g = wave(0.12,0.2,gamma*0.2,eeg1_norm)
P = np.abs(psi_b1*psi_h*psi_g)**2
P = np.where(P < 1e-10, 0, P)
P_norm = P/np.sum(P) if np.sum(P)>0 else P
win=min(200,max(10,len(P_norm)//10))
ent=[entropy(P_norm[i:i+win]) for i in range(0,len(P_norm)-win,win)]
ent=np.array(ent); ent_norm=(ent-ent.min())/(ent.max()-ent.min()) if ent.max()!=ent.min() else ent*0
t_ent=np.arange(len(ent_norm))*win/fs

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6,4))
ax1.plot(t,P); ax1.set_ylabel("Joint density")
ax2.plot(t_ent,ent_norm); ax2.set_ylabel("Entropy (norm)"); ax2.set_xlabel("Time (s)")
st.pyplot(fig)
st.caption("Entropy dips ≈ coherence; density clipped; PLV if 2 channels")
