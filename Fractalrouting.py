import numpy as np

MODES = 3  # number of harmonic modes

def simulate_advanced(G, K, steps, noise_level, positions):
    N = len(G.nodes)

    # Initialize multi-mode phases
    phases = np.random.uniform(0, 2*np.pi, (N, MODES))
    velocities = np.zeros((N, MODES))

    coherence = []
    variance = []
    slips = []

    gamma = 0.05  # damping factor
    delay_factor = 0.1

    for t in range(steps):
        prev_phases = phases.copy()
        new_phases = phases.copy()

        for i in range(N):
            for m in range(MODES):
                interaction = 0

                for j in G.neighbors(i):
                    # Spatial distance decay
                    dist = np.linalg.norm(positions[i] - positions[j])
                    spatial_weight = np.exp(-dist)

                    # Delay approximation
                    phase_delay = delay_factor * dist

                    interaction += spatial_weight * np.sin(
                        (phases[j][m] - phase_delay) - phases[i][m]
                    )

                # Cross-mode coupling
                cross = sum(
                    np.sin(phases[i][k] - phases[i][m])
                    for k in range(MODES) if k != m
                ) * 0.05

                # Update with damping
                velocities[i][m] += K * interaction + cross
                velocities[i][m] *= (1 - gamma)

                new_phases[i][m] += velocities[i][m]

        # Apply noise after disturbance
        if t > steps // 3:
            new_phases += np.random.normal(0, noise_level, (N, MODES))

        phases = new_phases

        # --- Metrics ---
        flat_phases = phases.flatten()

        R = np.abs(np.mean(np.exp(1j * flat_phases)))
        coherence.append(R)

        variance.append(np.var(flat_phases))

        slip_count = np.sum(np.abs(phases - prev_phases) > 1.0)
        slips.append(slip_count)

    return coherence, variance, slips, phases
