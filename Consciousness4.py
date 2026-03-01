import streamlit as st
import numpy as np

st.title("EEG raw plot")
uploaded=st.file_uploader("Upload")
if uploaded:
    try:
        data=np.genfromtxt(uploaded, delimiter=None)
        st.write("shape:", data.shape)
        if data.ndim>1:
            col=st.sidebar.slider("Column",0,data.shape[1]-1,0)
            eeg=data[:,col]
        else:
            eeg=data
        eeg = np.nan_to_num(eeg)
        st.write("std:", np.std(eeg))
        st.line_chart(eeg)
    except Exception as ex:
        st.error("Error: "+str(ex))
else:
    st.write("Upload file")
