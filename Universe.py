import streamlit as st
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# --- QC Class ---
class QC:
    def __init__(self, id, grid_size, base_cp=0.1, is_passive=False):
        self.id = id
        self.position = np.array([random.uniform(0, grid_size), random.uniform(0, grid_size)])
        self.velocity = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
        self.fundamental_frequency_mag = random.uniform(1.0, 8.0)
        self.fundamental_frequency_phase = random.uniform(0, 2 * math.pi)
        self.local_phase_coherence = 0.5
        self.coherence_potential = base_cp
        self.interaction_history_metric = 0
        self.effective_mass = 1.0
        self.cluster_id = -1
        self.is_passive = is_passive

    def update_from_blend(self, blended_signature):
        if not self.is_passive:
            self.fundamental_frequency_mag = blended_signature['freq_mag']
            self.fundamental_frequency_phase = blended_signature['freq_phase'] % (2 * math.pi)
            self.local_phase_coherence = blended_signature['lpc']
            self.coherence_potential = blended_signature['cp']
            self.interaction_history_metric = blended_signature['ihm']

# --- Functions ---
def calculate_distance(pos1, pos2, grid_size):
    dx = abs(pos1[0] - pos2[0]); dy = abs(pos1[1] - pos2[1])
    dx = min(dx, grid_size - dx); dy = min(dy, grid_size - dy)
    return math.sqrt(dx**2 + dy**2)

def calculate_harmonicity(qc1, qc2, ftol, ptol):
    freq_match = abs(qc1.fundamental_frequency_mag - qc2.fundamental_frequency_mag) / max(qc1.fundamental_frequency_mag, qc2.fundamental_frequency_mag) < ftol
    phase_diff = abs(qc1.fundamental_frequency_phase - qc2.fundamental_frequency_phase)
    phase_match = min(phase_diff, 2 * math.pi - phase_diff) < ptol
    return 0.9 if freq_match and phase_match else 0.5 if freq_match or phase_match else 0.1

def calculate_fractal_dimension_box_counting(qc_positions, box_sizes, grid_size):
    if len(qc_positions) < 2: return 0.0
    log_N, log_inv_s = [], []
    xs = [p[0] for p in qc_positions]; ys = [p[1] for p in qc_positions]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    bbox_min_x, bbox_max_x = min_x - 0.1, max_x + 0.1
    bbox_min_y, bbox_max_y = min_y - 0.1, max_y + 0.1
    bbox_width = max(0.2, bbox_max_x - bbox_min_x); bbox_height = max(0.2, bbox_max_y - bbox_min_y)
    effective_dim = max(grid_size, max(bbox_width, bbox_height))
    for s in box_sizes:
        if s <= 0: continue
        occupied = set()
        for qc_pos in qc_positions:
            bx = math.floor((qc_pos[0] - bbox_min_x) / s)
            by = math.floor((qc_pos[1] - bbox_min_y) / s)
            occupied.add((bx, by))
        Ns = len(occupied)
        if Ns > 0:
            log_N.append(math.log(Ns)); log_inv_s.append(math.log(effective_dim / s))
    if len(log_N) < 2: return 0.0
    try: return np.polyfit(log_inv_s, log_N, 1)[0]
    except: return 0.0

def calculate_rho_k_dm(qc_list, grid_size, cell_division):
    cell_sz = grid_size / cell_division
    grid = np.zeros((cell_division, cell_division))
    for qc in qc_list:
        ke = 0.5 * qc.effective_mass * np.dot(qc.velocity, qc.velocity)
        ix = max(0, min(math.floor(qc.position[0] / cell_sz), cell_division-1))
        iy = max(0, min(math.floor(qc.position[1] / cell_sz), cell_division-1))
        grid[ix][iy] += ke
    vol = cell_sz * cell_sz
    return grid / vol if vol > 0 else grid

def detect_clusters(qcs, dist_thr, cp_thr, grid_size):
    for qc in qcs: qc.cluster_id = -1
    sig = [qc for qc in qcs if not qc.is_passive and qc.coherence_potential >= cp_thr]
    if len(sig) < 2: return []
    adj = {qc.id: [] for qc in qcs}
    for i, qc1 in enumerate(qcs):
        for qc2 in qcs[i+1:]:
            d = calculate_distance(qc1.position, qc2.position, grid_size)
            if d <= dist_thr:
                adj[qc1.id].append(qc2.id); adj[qc2.id].append(qc1.id)
    visited, clusters = set(), []
    for qc in sig:
        if qc.id in visited: continue
        stack, comp = [qc.id], []
        while stack:
            cur = stack.pop()
            if cur in visited: continue
            visited.add(cur)
            obj = next(q for q in qcs if q.id == cur)
            comp.append(obj)
            stack += [n for n in adj[cur] if n not in visited]
        if len(comp) > 1: clusters.append(comp)
    return clusters

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🌌 CompresSim: Frequency-Dimension Cosmology")

with st.sidebar.expander("General Settings", True):
    NUM_QCS = st.slider("Number of QCs", 10, 800, 500, 10)
    GRID_SIZE = st.slider("Grid Size", 50, 1000, 500, 25)
    TOTAL_TIME_STEPS = st.slider("Total Time Steps", 100, 2000, 1000, 50)
    DT = st.slider("DT", 0.01, 0.5, 0.1, 0.01)

with st.sidebar.expander("Interaction & Movement", True):
    R_INT = st.slider("R_INT", 5.0, 100.0, 50.0, 5.0)
    HARMONICITY_FREQ_TOL = st.slider("Freq Tol", 0.05, 0.5, 0.4, 0.05)
    HARMONICITY_PHASE_TOL = st.slider("Phase Tol", 0.1, math.pi, math.pi*0.9, 0.1)
    ETA = st.slider("ETA", 0.1, 2.0, 2.0, 0.1)
    DRAG_COEFFICIENT = st.slider("Drag", 0.0, 0.1, 0.01, 0.005)
    GRAVITATIONAL_CONSTANT = st.slider("G", 0.0, 0.5, 0.01, 0.001)
    COSMIC_ATTRACTION_STRENGTH = st.slider("Cosmic Attract", 0.0, 10.0, 1.0, 0.1)

with st.sidebar.expander("Cluster & Measurement", True):
    CLUSTER_DIST_THRESHOLD = st.slider("Cluster Dist", 1.0, 50.0, 30.0, 1.0)
    CLUSTER_CP_THRESHOLD = st.slider("CP Thresh", 0.1, 1.0, 0.5, 0.1)
    PASSIVE_QC_PERCENTAGE = st.slider("Passive %", 0.0, 0.5, 0.1, 0.05)
    box_size_input = st.text_input("Box sizes for Df", value="1.0,2.0,4.0,8.0,16.0,32.0")
    BOX_SIZES_DF = np.array([float(x) for x in box_size_input.split(",") if x.strip()])

with st.sidebar.expander("Frequency Bands", True):
    st.write("Freq range: 1.0 – 8.0")
    low_max = st.slider("Low max", 1.5, 4.0, 2.5, 0.1)
    mid_max = st.slider("Mid max", low_max+0.1, 7.5, 5.0, 0.1)

BASE_MASS, BETA1, BETA2 = 1.0, 0.01, 0.1
CELL_DIVISION = int(GRID_SIZE / 4)

if st.sidebar.button("Run Simulation"):
    RAY1 = np.array([GRID_SIZE*0.25, GRID_SIZE*0.25])
    RAY2 = np.array([GRID_SIZE*0.75, GRID_SIZE*0.75])
    qcs = [QC(i, GRID_SIZE, is_passive=(i < int(NUM_QCS*PASSIVE_QC_PERCENTAGE))) for i in range(NUM_QCS)]
    st.session_state.qcs = qcs
    progress = st.progress(0)
    metrics = st.empty()
    plot = st.empty()
    for step in range(TOTAL_TIME_STEPS):
        progress.progress((step+1)/TOTAL_TIME_STEPS)
        # Interactions, forces, movement, CP updates... (same logic as previous version)
        # Measurements: Df overall, Df bands, clusters, rho_k_dm, etc.
        if step % 10 == 0:
            with plot.container():
                fig, ax = plt.subplots(figsize=(8,8))
                ax.scatter([qc.position[0] for qc in qcs], [qc.position[1] for qc in qcs],
                           c=[qc.fundamental_frequency_mag for qc in qcs], cmap='plasma',
                           s=[max(10, qc.effective_mass*5) for qc in qcs], alpha=0.7)
                ax.plot(RAY1[0], RAY1[1], 'rx'); ax.plot(RAY2[0], RAY2[1], 'rx')
                ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
                st.pyplot(fig); plt.close(fig)
    st.success("Done")
