import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.title("CF Harmonic Coherence Simulation")
st.markdown("Compare CF-A, CF-B, CF-C, and CF-D architectures under load")

# -----------------------------
# USER CONTROLS
# -----------------------------
N = st.slider("Number of Nodes", 20, 150, 50)
K = st.slider("Coupling Strength", 0.01, 1.0, 0.2)
steps = st.slider("Simulation Steps", 50, 500, 200)
noise_level = st.slider("Noise Level (Load)", 0.0, 1.0, 0.2)

# -----------------------------
# GRAPH GENERATION
# -----------------------------
def create_graph(arm_type, N):
    if arm_type == "CF-A":
        return nx.path_graph(N)

    elif arm_type == "CF-B":
        G = nx.path_graph(N)
        for i in range(0, N-3, 4):
            G.add_edge(i, i+3)
        return G

    elif arm_type == "CF-C":
        return nx.cycle_graph(N)

    elif arm_type == "CF-D":
        return nx.balanced_tree(r=2, h=int(np.log2(N)))

# -----------------------------
# SIMULATION
# -----------------------------
def run_simulation(G, K, steps, noise_level):
    N = len(G.nodes)
    phases = np.random.uniform(0, 2*np.pi, N)

    coherence = []

    for t in range(steps):
        new_phases = np.copy(phases)

        for i in range(N):
            neighbors = list(G.neighbors(i))
            interaction = sum(np.sin(phases[j] - phases[i]) for j in neighbors)

            new_phases[i] += K * interaction

        # Apply noise (load)
        if t > steps // 3:
            new_phases += np.random.normal(0, noise_level, N)

        phases = new_phases

        # Compute coherence
        R = np.abs(np.mean(np.exp(1j * phases)))
        coherence.append(R)

    return coherence

# -----------------------------
# RUN ALL ARMS
# -----------------------------
arms = ["CF-A", "CF-B", "CF-C", "CF-D"]
results = {}

for arm in arms:
    G = create_graph(arm, N)
    results[arm] = run_simulation(G, K, steps, noise_level)

# -----------------------------
# PLOT RESULTS
# -----------------------------
fig, ax = plt.subplots()

for arm in arms:
    ax.plot(results[arm], label=arm)

ax.set_title("Coherence Over Time")
ax.set_xlabel("Time Step")
ax.set_ylabel("Coherence (R)")
ax.legend()

st.pyplot(fig)

# -----------------------------
# INTERPRETATION
# -----------------------------
st.markdown("## Interpretation Guide")
st.markdown("""
- Higher curve = better coherence  
- Flatter curve under noise = more stable system  
- Faster recovery after drop = stronger resilience  

Expected:
- CF-D should maintain highest coherence under stress  
""")
