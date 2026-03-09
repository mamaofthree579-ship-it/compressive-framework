import numpy as np

h = np.sin(np.linspace(0, 2*np.pi, 100))
np.save("h_placeholder.npy", h.real)
# load it back to verify
loaded = np.load("h_placeholder.npy")
st.write(loaded[:5])
