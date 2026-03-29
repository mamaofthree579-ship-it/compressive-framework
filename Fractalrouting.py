import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(layout="wide")
st.title("CF-D Harmonic Coherence Simulation (Advanced)")

# -----------------------------
# CONTROLS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    N = st.slider("Nodes", 20, 150, 60)

with col2:
    K = st.slider("Coupling", 0.01, 1.0, 0.2)

with col3:
    steps = st.slider("Steps", 100, 600, 300)

with col4:
    noise_level = st.slider("Noise", 0.0, 1.0, 0.3)

disturbance_time = int(steps * 0.33)

# -----------------------------
# GRAPH TYPES
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
    return np.abs(np.mean(np.exp(1j * phases)))

def compute_variance(phases):
    return np.var(phases)

def detect_phase_slips(phases, prev_phases, threshold=1.0):
    diff = np.abs(phases - prev_phases)
    return np.sum(diff > threshold)

# -----------------------------
# SIMULATION
# -----------------------------
def simulate(G, K, steps, noise_level):
    N = len(G.nodes)
    phases = np.random.uniform(0, 2*np.pi, N)

    coherence = []
    variance = []
    slips = []

    for t in range(steps):
        prev_phases = np.copy(phases)
        new_phases = np.copy(phases)

        for i in range(N):
            neighbors = list(G.neighbors(i))
            interaction = sum(np.sin(phases[j] - phases[i]) for j in neighbors)
            new_phases[i] += K * interaction

        # Apply disturbance
        if t > disturbance_time:
            new_phases += np.random.normal(0, noise_level, N)

        phases = new_phases

        coherence.append(compute_coherence(phases))
        variance.append(compute_variance(phases))
        slips.append(detect_phase_slips(phases, prev_phases))

    return coherence, variance, slips, phases

# -----------------------------
# RUN SIMULATIONS
# -----------------------------
arms = ["CF-A", "CF-B", "CF-C", "CF-D"]
results = {}

for arm in arms:
    G = create_graph(arm, N)
    results[arm] = simulate(G, K, steps, noise_level)

# -----------------------------
# PLOTS
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
st.subheader("Final Network State (Phase Coloring)")

selected_arm = st.selectbox("Select Arm to Visualize", arms)
G = create_graph(selected_arm, N)
final_phases = results[selected_arm][3]

pos = nx.spring_layout(G, seed=42)

fig4, ax4 = plt.subplots()
nx.draw(
    G,
    pos,
    node_color=final_phases,
    cmap=plt.cm.hsv,
    node_size=50,
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
    st.write(f"Avg Variance: {np.mean(variance):.3f}")
    st.write(f"Total Phase Slips: {np.sum(slips)}")

# -----------------------------
# INTERPRETATION
# -----------------------------
st.markdown("## Interpretation")

st.markdown("""
**What to look for:**

- Higher coherence = better synchronization  
- Lower variance = better energy distribution  
- Fewer phase slips = higher stability  
- Faster recovery after disturbance = resilience  

**Hypothesis:**
CF-D should:
- Maintain highest coherence  
- Show lowest variance  
- Exhibit minimal phase slips  
- Recover fastest after noise  
""")
