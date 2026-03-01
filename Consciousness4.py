import streamlit as st
import numpy as np

st.title("EEG chart")
uploaded=st.file_uploader("Upload")
if uploaded:
    data=np.genfromtxt(uploaded, delimiter=None)
    st.write("shape:", data.shape)
    eeg=data[:,0] if data.ndim>1 else data
    st.line_chart(eeg)
else:
    st.write("Upload file")
