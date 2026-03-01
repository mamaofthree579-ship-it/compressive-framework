import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("CGUP Frequency Ladder – with GR bound")

# --- sliders ---
alpha = st.sidebar.slider("α* (coupling)", 0.0, 0.2, 0.08, 0.005)
lam = st.sidebar.slider("λ (scaling)", 0.3, 0.7, 0.5, 0.01)
beta = 0.4
N = 6

# GR fundamental (2,2,0) real part, in arbitrary units
omega_GR = 0.3737

def cgup_freqs(alpha, lam):
    return [omega_GR + (alpha**2)*omega_GR*(lam**n + beta*lam**(n+1))
            for n in range(N)]

freqs = cgup_freqs(alpha, lam)
deltas = [freqs[i+1]-freqs[i] for i in range(len(freqs)-1)]

# --- ladder plot ---
fig, ax = plt.subplots()
for f in freqs:
    ax.axvline(f, color='tab:blue', lw=2)
# GR consistency band (±0.06 around GR freq)
ax.axvspan(omega_GR-0.06, omega_GR+0.06, color='gray', alpha=0.2,
           label='GR 90% bound')
ax.set_xlabel('Frequency (arb.)')
ax.set_yticks([])
ax.set_title(f'Ladder: α*={alpha:.3f}, λ={lam:.3f}')
ax.legend()
st.pyplot(fig)

# --- spacing output ---
st.subheader("Spacings Δ")
for i,d in enumerate(deltas):
    st.write(f"Δ{i} = {d:.5f} (ratio to previous: {d/deltas[i-1]:.3f} )" 
             if i>0 else f"Δ{i} = {d:.5f}")

# optional: show if ladder breaches GR band
if any(abs(f-omega_GR) > 0.06 for f in freqs):
    st.warning("Some ladder tones lie outside the current GR 90% bound.")
else:
    st.success("All tones sit inside the GR bound – consistent with GW250114.")
