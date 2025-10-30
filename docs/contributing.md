# Contributing to the Compressive Framework (CF–DPF)

Thank you for your interest in contributing to **CF–DPF (Compressive Framework — Dynamic Particle Formation)**!  
This project explores theoretical physics and simulation concepts through interactive notebooks and visual models.

---

## 🧭 Contribution Guidelines

### 1. Setting up your environment
Before contributing, ensure you have the dependencies installed:

```bash
pip install -r requirements.txt

Then serve the documentation locally:

mkdocs serve

2. Branching & Workflow
	•	main → stable release branch (only reviewed merges)
	•	dev → active development branch (PRs go here)
	•	feature/ → for new modules or notebooks
	•	fix/ → for bug patches or small corrections

Example: git checkout -b feature/notebook11_new_effect

git checkout -b feature/notebook11_new_effect

```python
# code
```

4. Submitting Pull Requests
	1.	Fork the repository
	2.	Create your feature branch
	3.	Commit changes with clear messages
	4.	Submit a pull request to the dev branch

⸻

🧠 Helpful Commands

Build site manually: mkdocs build

Validate notebooks: python .github/scripts/execute_notebooks.py


⸻

💡 Need Help?

If you have questions, open an issue with a descriptive title.
For conceptual discussion or physics questions, use Discussions.

⸻

— The CF–DPF Development Team

---

### ⚙️ **`api.md`**
*(Documenting available modules and potential interfaces)*

```markdown
# CF–DPF API Reference

This document outlines the **programmatic interface** and structure of the Compressive Framework.  
It is intended for developers integrating CF–DPF modules into research tools or simulations.

---

## 🧩 Core Modules

| Module | Description |
|:-------|:-------------|
| `cf.wave` | Handles wave compression, resonance, and harmonic transformations. |
| `cf.chronon` | Models chronon (quantum time) dynamics. |
| `cf.particle` | Manages particle formation, coalescence, and interactions. |
| `cf.fields` | Describes interaction fields (graviton, cognon, etc.) and nonlocal binding. |
| `cf.vacuum` | Models vacuum symmetry breaking and domain formation. |
| `cf.visuals` | Utility for plots, animations, and visual diagnostics. |

---

## 🧠 Example Usage

```python
from cf.wave import WaveCompressor
from cf.particle import ParticleSimulator

wave = WaveCompressor(frequency=4.2, amplitude=1.0)
particles = ParticleSimulator(wave)
particles.simulate(steps=1000)
particles.plot()
```


⸻

🧮 Math and Physics Utilities

These are helper functions used across multiple notebooks: from cf.utils import hbar, c, planck_length

E = hbar * 2 * np.pi * frequency
print(f"Energy: {E:.3e} J")


⸻

🚀 Future API Goals
	•	Introduce unified simulation class: CFUniverse()
	•	Add parallelized backends for larger-scale particle simulations
	•	Build API wrappers for visualization dashboards

⸻

Auto-generated from core documentation and notebooks.
Use mkdocs build to regenerate.

