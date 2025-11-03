# Double-Slit Observer Simulator

## Purpose
This simulator models the quantum double-slit experiment under the **Compressive Framework** hypothesis. It allows comparing how various forms of observation—detectors, instruments, or humans—alter the harmonic field responsible for interference.

## Concepts
| Parameter | Meaning |
|------------|----------|
| `observer_type` | Determines the nature of the harmonic perturbation (detector / instrument / human) |
| `observer_strength` | Scales the magnitude of perturbation |
| `global_noise` | Quantum fluctuation background |
| `leak` and `decay` | Represent curvature leakage and relaxation in the residual field |
| `thresh` | Threshold for nucleation detection (particle formation) |

## Key Findings
- **Without observation**, stable harmonic interference fringes emerge.
- **With detectors**, phase resetting near the slits reduces coherence.
- **With humans**, a nonstationary, multi-harmonic field emerges—fringes show complex modulations and localized decoherence.

## Visual Outputs
- **Intensity Map** — shows |ψ|² interference energy.
- **Residual Field** — accumulates curvature and nucleation events.
- **Tracked Particles** — points where the field condenses into discrete detections.
- **Side-by-Side Comparison** — baseline vs. observed fields.
- **GIF Export** — 15 FPS animation for presentations.

## Interpretation
Observation introduces a harmonic coupling between the observer’s local potential field and the quantum waveform. This coupling reshapes the interference field through additional phase correlations or decoherence harmonics. The effect magnitude and symmetry depend on observer type and interaction strength.

In the **human case**, intrinsic temporal stochasticity and multi-frequency feedback create more complex waveform collapses—consistent with our compressive harmonics theory of participatory curvature.
