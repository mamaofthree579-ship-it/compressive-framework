import streamlit as st
import numpy as np

st.title("EEG chart with column picker")
uploaded=st.file_uploader("Upload")
if uploaded:
    data=np.genfromtxt(uploaded, delimiter=None)
    st.write("shape:", data.shape)
    if data.ndim>1:
        col=st.sidebar.slider("Column",0,data.shape[1]-1,0)
        eeg=data[:,col]
    else:
        eeg=data
    st.write("std:", np.std(eeg))
    st.line_chart(eeg)
else:
    st.write("Upload file")
