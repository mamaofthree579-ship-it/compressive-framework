import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(layout="wide")
st.title("CF-D Advanced Harmonic Field Simulation")

# -----------------------------
# CONTROLS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    N = st.slider("Nodes", 20, 120, 60)

with col2:
    K = st.slider("Coupling Strength", 0.01, 1.0, 0.2)

with col3:
    steps = st.slider("Steps", 100, 600, 300)

with col4:
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.3)

MODES = 3
gamma = 0.05
delay_factor = 0.15

disturbance_time = int(steps * 0.33)

# -----------------------------
# GRAPH STRUCTURES
# -----------------------------
def create_graph(arm, N):
    if arm == "CF-A":
        return nx.path_graph(N)

    elif arm == "CF-B":
        G = nx.path_graph(N)
        for i in range(0, N-3, 4):
            G.add_edge(i, i+3)
        return G

    elif arm == "CF-C":
        return nx.cycle_graph(N)

    elif arm == "CF-D":
        h = int(np.log2(N))
        return nx.balanced_tree(r=2, h=h)

# -----------------------------
# METRICS
# -----------------------------
def compute_coherence(phases):
    flat = phases.flatten()
    return np.abs(np.mean(np.exp(1j * flat)))

def compute_variance(phases):
    return np.var(phases)

def detect_phase_slips(phases, prev_phases, threshold=1.0):
    return np.sum(np.abs(phases - prev_phases) > threshold)

# -----------------------------
# SIMULATION
# -----------------------------
def simulate_advanced(G, K, steps, noise_level, positions):
    N = len(G.nodes)

    phases = np.random.uniform(0, 2*np.pi, (N, MODES))
    velocities = np.zeros((N, MODES))

    coherence = []
    variance = []
    slips = []

    for t in range(steps):
        prev_phases = phases.copy()
        new_phases = phases.copy()

        for i in range(N):
            for m in range(MODES):
                interaction = 0

                for j in G.neighbors(i):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    spatial_weight = np.exp(-dist)

                    delay = delay_factor * dist

                    interaction += spatial_weight * np.sin(
                        (phases[j][m] - delay) - phases[i][m]
                    )

                # Cross-mode coupling
                cross = sum(
                    np.sin(phases[i][k] - phases[i][m])
                    for k in range(MODES) if k != m
                ) * 0.05

                velocities[i][m] += K * interaction + cross
                velocities[i][m] *= (1 - gamma)

                new_phases[i][m] += velocities[i][m]

        # Apply disturbance
        if t > disturbance_time:
            new_phases += np.random.normal(0, noise_level, (N, MODES))

        phases = new_phases

        coherence.append(compute_coherence(phases))
        variance.append(compute_variance(phases))
        slips.append(detect_phase_slips(phases, prev_phases))

    return coherence, variance, slips, phases

# -----------------------------
# RUN SIMULATION
# -----------------------------
arms = ["CF-A", "CF-B", "CF-C", "CF-D"]
results = {}

for arm in arms:
    G = create_graph(arm, N)

    # Assign spatial positions
    positions = {i: np.random.rand(2) for i in range(len(G.nodes))}

    results[arm] = simulate_advanced(G, K, steps, noise_level, positions)

# -----------------------------
# PLOT: COHERENCE
# -----------------------------
st.subheader("Coherence Over Time")
fig1, ax1 = plt.subplots()

for arm in arms:
    ax1.plot(results[arm][0], label=arm)

ax1.axvline(disturbance_time, linestyle="--")
ax1.set_xlabel("Time")
ax1.set_ylabel("Coherence")
ax1.legend()

st.pyplot(fig1)

# -----------------------------
# PLOT: VARIANCE
# -----------------------------
st.subheader("Phase Variance (Energy Distribution)")
fig2, ax2 = plt.subplots()

for arm in arms:
    ax2.plot(results[arm][1], label=arm)

ax2.axvline(disturbance_time, linestyle="--")
ax2.set_xlabel("Time")
ax2.set_ylabel("Variance")
ax2.legend()

st.pyplot(fig2)

# -----------------------------
# PLOT: PHASE SLIPS
# -----------------------------
st.subheader("Phase-Slip Events")
fig3, ax3 = plt.subplots()

for arm in arms:
    ax3.plot(results[arm][2], label=arm)

ax3.axvline(disturbance_time, linestyle="--")
ax3.set_xlabel("Time")
ax3.set_ylabel("Slip Count")
ax3.legend()

st.pyplot(fig3)

# -----------------------------
# NETWORK VISUALIZATION
# -----------------------------
st.subheader("Final Network State")

selected_arm = st.selectbox("Select Arm", arms)

G = create_graph(selected_arm, N)
positions = {i: np.random.rand(2) for i in range(len(G.nodes))}

final_phases = results[selected_arm][3]

node_colors = [np.mean(final_phases[i]) for i in range(len(final_phases))]

fig4, ax4 = plt.subplots()

nx.draw(
    G,
    positions,
    node_color=node_colors,
    cmap=plt.cm.hsv,
    node_size=60,
    with_labels=False,
    ax=ax4
)

st.pyplot(fig4)

# -----------------------------
# SUMMARY METRICS
# -----------------------------
st.subheader("Summary Metrics")

for arm in arms:
    coherence = results[arm][0]
    variance = results[arm][1]
    slips = results[arm][2]

    st.markdown(f"### {arm}")
    st.write(f"Final Coherence: {coherence[-1]:.3f}")
    st.write(f"Average Variance: {np.mean(variance):.3f}")
    st.write(f"Total Phase Slips: {np.sum(slips)}")

# -----------------------------
# INTERPRETATION
# -----------------------------
st.markdown("## Interpretation Guide")

st.markdown("""
- Higher coherence → stronger synchronization  
- Lower variance → better energy distribution  
- Fewer phase slips → greater stability  
- Faster recovery after disturbance → resilience  

**Hypothesis:**
CF-D (fractal) should outperform others under:
- Noise
- Delay
- Spatial decay
""")
