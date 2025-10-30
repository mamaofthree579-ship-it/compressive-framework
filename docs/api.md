# CF–DPF API Reference

This document outlines the **Compressive Framework — Dynamic Particle Formation (CF–DPF)** API.  
It summarizes the logical module structure, key classes, and utility functions referenced throughout the notebooks.

---

## 🧩 Core Architecture

| Module | Description |
|:--------|:-------------|
| `cf.wave` | Handles compression, propagation, and resonance of field waves. |
| `cf.chronon` | Models time-quantized units (chronons) and temporal phase flow. |
| `cf.particle` | Generates stable particle configurations from residual field compression. |
| `cf.motion` | Governs transitions from distributed wave behavior to localized motion. |
| `cf.binding` | Manages fractional presence, nonlocal binding, and field entanglement. |
| `cf.coalescence` | Simulates multi-particle merging and stability thresholds. |
| `cf.field` | Defines field-level interactions between cognons (data), gravitons (gravity), and chronons (time). |
| `cf.curvature` | Handles quantized curvature feedback and geometry distortion. |
| `cf.vacuum` | Simulates vacuum symmetry breaking and domain formation. |
| `cf.utils` | Shared math, constants, and visualization utilities. |

---

## ⚙️ Example Workflow

```python
from cf.wave import WaveCompressor
from cf.chronon import ChrononField
from cf.particle import ParticleFormation
from cf.motion import WaveToParticle
from cf.field import FieldInteraction

# Initialize a wave
wave = WaveCompressor(frequency=3.2, amplitude=1.0)

# Model temporal compression
chronon_field = ChrononField(time_quantum=1e-21)
chronon_field.propagate(wave)

# Generate stable particles from the wavefront
particle = ParticleFormation(wave)
particle.nucleate(resolution=500)

# Transition wave state to localized motion
transition = WaveToParticle(wave, particle)
transition.visualize()

# Introduce gravitational / informational coupling
field = FieldInteraction()
field.bind(wave, particle)


⸻

🧮 Utilities and Constants

The cf.utils module provides key physical constants and math helpers.

from cf.utils import hbar, c, planck_length, resonance_energy

E = resonance_energy(frequency=4.0)
print(f"Energy: {E:.3e} J")

Available constants:
	•	hbar — Reduced Planck constant
	•	c — Speed of light
	•	planck_length — Quantum scale base length
	•	tau — Time period of one quantum cycle

Available utilities:
	•	normalize_wave(data) — rescales amplitude data
	•	energy_density(wave) — returns the energy distribution
	•	phase_shift(field, delta_t) — applies temporal offsets

⸻

🧠 Field Entities

Entity | Symbol | Description
------------------------------

Chronon -> χ -> Represents a discrete quantum of time

Graviton -> ϕ -> Mediates curvature and mass interaction

Cognon -> ψ -> Encodes informational density and pattern memory

These are represented programmatically through the cf.field module and can interact or superimpose via coupling rules.

⸻

🌀 Visualization API

The cf.visuals toolkit provides utilities for rendering and analysis.

from cf.visuals import plot_field, animate_transition

plot_field(field)
animate_transition(wave, particle)

Functions:
	•	plot_field(field) → 2D field magnitude map
	•	animate_transition(wave, particle) → smooth transformation animation
	•	render_vacuum_domain(domain) → visualize domain formation & collapse

⸻

🚀 Simulation Pipeline Summary

Step
Notebook
Description
1
01_wave_compression
Wave compression dynamics
2
02_chronon_dynamics
Quantized time structures
3
03_particle_formation
Stable particle nucleation
4
04_wave_to_particle_motion
Motion emergence
5
05_fractional_presence
Nonlocal field binding
6
06_particle_coalescence
Multi-particle merging
7
07_field_interactions
Cross-field coherence
8
08_quantized_curvature
Geometric feedback
9
09_vacuum_symmetry_breaking
Symmetry and phase imbalance
10
10_vacuum_domain_formation
Stable domain formation


⸻

🧩 Future API Goals
	•	Add CFUniverse() unified simulation orchestrator
	•	Support GPU acceleration for wave propagation
	•	Integrate real-time visualization dashboards
	•	Create JSON-based experiment templates for reproducibility

⸻

Auto-generated documentation based on conceptual and mathematical models within the CF–DPF framework.
Use mkdocs build or mkdocs serve to regenerate.
