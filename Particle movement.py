import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# App title
st.title("Compression Field Evolution")

# Sidebar controls
st.sidebar.header("Parameters")
lambda_val = st.sidebar.slider("λ (lambda)", 0.1, 5.0, 1.0, 0.1)
C0 = st.sidebar.slider("C₀", 0.1, 3.0, 1.0, 0.1)
G = st.sidebar.slider("G", 0.1, 5.0, 1.0, 0.1)

st.sidebar.header("Initial conditions")
C_init = st.sidebar.slider("C(0)", -3.0, 3.0, 2.0, 0.1)
Cdot_init = st.sidebar.slider("C'(0)", -2.0, 2.0, 0.0, 0.1)

t_max = st.sidebar.slider("Max time", 1.0, 50.0, 10.0, 1.0)

def V(C):
    return 0.25 * lambda_val * (C**2 - C0**2)**2

def dV_dC(C):
    return lambda_val * C * (C**2 - C0**2)

def equations(t, y):
    C, Cdot = y
    rho = 0.5 * Cdot**2 + V(C)
    H = np.sqrt((8*np.pi*G/3) * rho)
    Cddot = -3*H*Cdot - dV_dC(C)
    return [Cdot, Cddot]

if st.button("Run simulation"):
    y0 = [C_init, Cdot_init]
    sol = solve_ivp(equations, [0, t_max], y0, t_eval=np.linspace(0, t_max, 1000))

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0])
    ax.set_xlabel("Time")
    ax.set_ylabel("C(t)")
    ax.set_title("Compression Field Evolution")
    st.pyplot(fig)
else:
    st.write("Adjust parameters in the sidebar, then click **Run simulation**.")
