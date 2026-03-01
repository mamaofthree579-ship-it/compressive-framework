import streamlit as st
import numpy as np
uploaded=st.file_uploader("Upload")
if uploaded:
    data=np.genfromtxt(uploaded,delimiter=None)
    st.write("shape:", data.shape)
    eeg=data if data.ndim==1 else data[:,0]
    st.line_chart(eeg)
