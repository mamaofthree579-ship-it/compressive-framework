import streamlit as st
import numpy as np
import pandas as pd
import io
from scipy.io import loadmat

uploaded=st.file_uploader("EEG",type=["csv","mat","xlsx"])
if uploaded:
    # load as before
    eeg = ... # same loading logic
    eeg = eeg - np.mean(eeg)
    st.line_chart(eeg)
    st.write("len:", len(eeg), "std:", np.std(eeg))
