# 🌀 Wave Compression — Foundations of the Compressive Framework

> **Notebook 1 of 10**  
> Establishing the fundamental **wave compression dynamics** that give rise to all emergent particle behavior in the Compressive Framework (CF).

---

## 1. Concept Overview

The **Compressive Framework (CF)** begins with the **wavefunction ψ(t, x)**, which represents distributed presence — energy, probability, and informational amplitude.

Compression occurs when ψ(t, x) interacts with **spatial curvature** and **internal feedback** fields, gradually reducing its spread while amplifying local density.

This dynamic sets the foundation for particle formation and the coupling of **graviton**, **chronon**, and **cognon** modes.

---

## 2. Foundational Equation

The compressive wave equation can be written as a modified Schrödinger form:

\[
i \hbar \frac{\partial \psi}{\partial t} =
-\frac{\hbar^2}{2m} \nabla^2 \psi
+ \alpha |\psi|^2 \psi
- \beta \nabla^2(|\psi|^2)\psi
\]

where:

| Symbol | Meaning |
|:--------|:--------|
| \( \alpha \) | Nonlinear self-attraction (compression gain) |
| \( \beta \) | Diffusion-like resistance term |
| \( \psi(t,x) \) | Distributed wave amplitude |
| \( m \) | Effective inertial parameter |
| \( \hbar \) | Reduced Planck constant (scaling factor) |

This model balances **nonlinear focusing** with **spatial diffusion**, naturally producing compression thresholds that seed particle-like stability.

---

## 3. Visualization — 1D Wave Compression

```python
# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = np.linspace(-10, 10, 400)
t = np.linspace(0, 5, 200)
alpha, beta = 1.0, 0.1

# Initial Gaussian wave
psi0 = np.exp(-x**2)

def compressive_step(psi, dt=0.02, dx=0.05):
    laplacian = np.gradient(np.gradient(psi, dx), dx)
    nonlinear = alpha * np.abs(psi)**2 * psi
    diffusion = -beta * laplacian * np.abs(psi)**2
    return psi + dt * (1j * (-laplacian + nonlinear + diffusion))

psi = psi0.copy()
for _ in range(100):
    psi = compressive_step(psi)

plt.figure(figsize=(8, 4))
plt.plot(x, np.abs(psi0)**2, '--', label='Initial |ψ|²')
plt.plot(x, np.abs(psi)**2, label='Compressed |ψ|²')
plt.title("Wave Compression in 1D Field ψ(t, x)")
plt.xlabel("x")
plt.ylabel("Amplitude Density")
plt.legend()
plt.grid(True)
plt.show()
