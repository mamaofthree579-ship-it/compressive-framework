import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("CGUP Ringdown Explorer")

# Sidebar sliders
alpha = st.sidebar.slider("Coupling α*", 0.0, 0.3, 0.1, 0.01)
lam   = st.sidebar.slider("Scaling λ",   0.3, 0.7, 0.5, 0.01)
beta  = 0.4  # fixed for now
N = 4

# GR fundamental (2,2,0) in 1/M units
omega_GR = 0.3737 - 0.08896j

def cgup_freqs(alpha, lam):
    return [omega_GR + (alpha**2)*omega_GR*(lam**n + beta*lam**(n+1))
            for n in range(N)]

t = np.linspace(0, 0.2, 2000)
h = np.zeros_like(t)
freqs = cgup_freqs(alpha, lam)
for w in freqs:
    h += np.exp(w.imag*t) * np.cos(w.real*t)

# Plot waveform
fig, ax = plt.subplots()
ax.plot(t, h)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Strain (arb.)")
ax.set_title(f"α*={alpha:.2f}, λ={lam:.2f}")
st.pyplot(fig)

# Show frequencies
st.subheader("CGUP frequencies")
for n, w in enumerate(freqs):
    st.write(f"n={n}: {w.real:.4f}  {w.imag:+.4f}j  (Δ={w.real-omega_GR.real:.4e})")
