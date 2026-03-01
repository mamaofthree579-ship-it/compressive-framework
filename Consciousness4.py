import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import loadmat
import io
import pandas as pd

st.title("EEG–Joint PLV safe")

f_gut=0.12; f_heart=1.1; f_brain=38.0
K=st.sidebar.slider("Coupling",0.1,1.5,0.6)
gamma=st.sidebar.slider("Decoherence",0.001,0.1,0.02)

t=np.linspace(0,10,2000)
drive=1+K*np.sin(2*np.pi*0.2*t)
joint = np.exp(-gamma*t)*drive*np.exp(1j*2*np.pi*f_gut*t) * np.exp(-gamma*0.5*t)*drive*np.exp(1j*2*np.pi*f_heart*t) * np.exp(-gamma*t)*drive*np.exp(1j*2*np.pi*f_brain*t)
phase_joint = np.angle(joint)

uploaded=st.file_uploader("Upload EEG",type=["csv","mat","xlsx"])
eeg=None
if uploaded:
    try:
        if uploaded.name.endswith(".mat"):
            mat=loadmat(uploaded); key=[k for k in mat.keys() if not k.startswith("__")][0]
            eeg_raw=mat[key].squeeze(); eeg = eeg_raw[:,1:].mean(axis=1) if eeg_raw.ndim>1 else eeg_raw
        elif uploaded.name.endswith(".xlsx"):
            df=pd.read_excel(uploaded); eeg=df.values[:,1:].mean(axis=1) if df.shape[1]>1 else df.values.mean(axis=1)
        else:
            s=io.StringIO(uploaded.getvalue().decode("utf-8"))
            data=np.genfromtxt(s,delimiter=None,filling_values=0)
            eeg=data[:,1:].mean(axis=1) if data.ndim>1 else data
    except Exception as ex:
        st.error("Load error: "+str(ex))

if eeg is not None:
    eeg=eeg-np.mean(eeg)
    st.write("EEG mean:", np.mean(eeg), "std:", np.std(eeg))
    st.line_chart(eeg)
    t_eeg=np.linspace(0,10,len(eeg))
    phase_eeg=np.angle(hilbert(eeg))
    phase_eeg_unwrapped=np.unwrap(phase_eeg)
    phase_joint_interp = np.interp(t_eeg, t, phase_joint)
    plv = np.abs(np.mean(np.exp(1j*(phase_joint_interp - phase_eeg_unwrapped))))
    st.write("Phase-Locking Value:", plv)
    fig, ax = plt.subplots()
    ax.plot(t_eeg, phase_eeg_unwrapped, label="EEG phase")
    ax.plot(t_eeg, phase_joint_interp, label="Joint phase")
    ax.set_ylabel("Unwrapped phase"); ax.legend()
    st.pyplot(fig)
else:
    st.write("shape:", getattr(eeg, 'shape', None))
