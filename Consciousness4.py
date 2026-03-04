import streamlit as st
import numpy as np

st.title("EEG safe plot")
if 'eeg' not in st.session_state:
    st.session_state.eeg=None

uploaded=st.file_uploader("Upload")
if uploaded and st.session_state.eeg is None:
    try:
        data=np.genfromtxt(uploaded, delimiter=None)
        if data.ndim>1:
            col=st.sidebar.slider("Column",0,data.shape[1]-1,0)
            eeg=data[:,col]
        else:
            eeg=data
        st.session_state.eeg=np.nan_to_num(eeg)
    except Exception as ex:
        st.error(str(ex))

eeg=st.session_state.eeg
if eeg is not None:
    st.line_chart(eeg)
else:
    st.write("Upload file")
