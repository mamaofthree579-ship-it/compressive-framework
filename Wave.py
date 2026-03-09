import numpy as np
import streamlit as st

h = np.sin(np.linspace(0, 2*np.pi, 100))
np.save("h_placeholder.npy", h.real)
# load it back to verify
loaded = np.load("h_placeholder.npy")
st.write(loaded[:5])
st.line_chart(h)
