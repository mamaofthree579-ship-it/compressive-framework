import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- chord pressure waveform (simplified) ---
def chord_pressure(t):
    # fundamental frequencies for G Maj7 (rough example)
    freqs = [392, 494, 587, 784]  # Hz
    amp = 0.02  # Pa, ~40 dB
    return sum(amp*np.sin(2*np.pi*f*t) for f in freqs)/len(freqs)

# --- ODE system ---
def model(y, t, sound=False):
    Ca, Repair, D = y
    F = chord_pressure(t) if sound else 0.0
    dCa = -0.1*Ca + 0.05*F
    dRepair = 0.2*Ca - 0.05*Repair
    dD = -0.3*Repair + 0.1  # constant damage source
    return [dCa, dRepair, dD]

st.title("Lullaby‑chord cell‑repair simulation")
sound = st.checkbox("Play chord pressure?", value=True)
t = np.linspace(0, 200, 1000)
y0 = [0.0, 0.0, 1.0]  # start with damage D=1
sol = odeint(model, y0, t, args=(sound,))

fig, ax = plt.subplots()
ax.plot(t, sol[:,2], label="Damage D(t)")
ax.set_xlabel("Time")
ax.set_ylabel("Relative damage")
ax.legend()
st.pyplot(fig)
