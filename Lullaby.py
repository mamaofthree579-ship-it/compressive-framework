import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def chord_pressure(t, amp):
    freqs = [392, 494, 587, 784]
    return sum(amp * np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)

def model(y, t, sound, amp, coup):
    Ca, Repair, D = y
    F = chord_pressure(t, amp) if sound else 0.0
    dCa = -0.1 * Ca + coup * F
    dRepair = 0.2 * Ca - 0.05 * Repair
    dD = -0.3 * Repair + 0.1 - 0.05 * D
    return [dCa, dRepair, dD]

st.title("Lullaby‑chord repair simulation")

sound = st.checkbox("Play chord pressure?", value=True)
amp = st.slider("Pressure amplitude (Pa)", 0.0, 0.05, 0.02, 0.005)
coup = st.slider("Ca coupling", 0.0, 0.02, 0.005, 0.001)

t = np.linspace(0, 200, 1000)
y0 = [0.0, 0.0, 1.0]
sol = odeint(model, y0, t, args=(sound, amp, coup))

fig, ax = plt.subplots()
ax.plot(t, sol[:, 2], label="Damage D(t)")
ax.set_xlabel("Time")
ax.set_ylabel("Relative damage")
ax.legend()
st.pyplot(fig)
