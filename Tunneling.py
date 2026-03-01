import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("CGUP Frequency Ladder Demo")

alpha = st.sidebar.slider("α*", 0.0, 0.3, 0.1, 0.01)
lam = st.sidebar.slider("λ", 0.3, 0.7, 0.5, 0.01)
beta = 0.4
N = 6
omega_GR = 0.3737 # real part only for ladder plot

def cgup_reals(alpha, lam):
    return [omega_GR + (alpha**2)*omega_GR*(lam**n + beta*lam**(n+1)) for n in range(N)]

freqs = cgup_reals(alpha, lam)

# Ladder plot
fig, ax = plt.subplots()
for i, f in enumerate(freqs):
    ax.axvline(f, ymin=0, ymax=0.8, label=f'n={i}' if i<3 else None)
ax.set_xlabel('Frequency (arb. units)')
ax.set_yticks([])
ax.set_title(f'Ladder (α*={alpha:.2f}, λ={lam:.2f})')
st.pyplot(fig)

# Show spacing
st.write("Spacings Δ:")
for i in range(len(freqs)-1):
    st.write(f"Δ{i} = {freqs[i+1]-freqs[i]:.5f}")
