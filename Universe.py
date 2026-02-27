import random
import math
import numpy as np
from scipy.spatial.distance import pdist, squareform

# --- 1. QC Class Definition ---
class QC:
    def __init__(self, id, grid_size):
        self.id = id
        self.position = np.array([random.uniform(0, grid_size), random.uniform(0, grid_size)])
        self.velocity = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
        
        # Data Signature Components
        self.fundamental_frequency_mag = random.uniform(1.0, 2.0)
        self.fundamental_frequency_phase = random.uniform(0, 2 * math.pi) # 0 to 2pi
        self.local_phase_coherence = 0.5 # Normalized 0 to 1
        self.coherence_potential = 0.1 # Dynamic scalar, starts low
        self.interaction_history_metric = 0 # Cumulative interactions

        # Derived Properties
        self.effective_mass = 1.0 # Starts at base mass
        self.cluster_id = -1 # To track which cluster it belongs to

    def get_data_signature(self):
        # Returns a combined representation of its internal state for blending
        return {
            'freq_mag': self.fundamental_frequency_mag,
            'freq_phase': self.fundamental_frequency_phase,
            'lpc': self.local_phase_coherence,
            'cp': self.coherence_potential,
            'ihm': self.interaction_history_metric
        }

    def update_from_blend(self, blended_signature):
        # Update QC's internal state from a blended signature
        self.fundamental_frequency_mag = blended_signature['freq_mag']
        self.fundamental_frequency_phase = blended_signature['freq_phase'] % (2 * math.pi) # Keep phase within 0 to 2pi
        self.local_phase_coherence = blended_signature['lpc']
        self.coherence_potential = blended_signature['cp']
        self.interaction_history_metric = blended_signature['ihm']

# --- 2. Simulation Parameters ---
GRID_SIZE = 10.0 # 2D Grid size (10x10)
NUM_QCS = 7
TOTAL_TIME_STEPS = 200
DT = 0.1 # Time step duration

# Interaction & Movement Constants - TUNED
R_INT = 2.5 # Interaction radius (increased)
R_MERGE = 0.1 # Merger distance
ALPHA = 0.01 # Interaction strength scaling
HARMONICITY_FREQ_TOL = 0.2 # Freq magnitude tolerance (increased)
HARMONICITY_PHASE_TOL = math.pi / 2 # Phase tolerance (increased)
ETA = 0.2 # Movement learning rate (increased)

# Mass Scaling Constants (m_i = m_0 + beta1 * h_i + beta2 * c_i)
BASE_MASS = 1.0
BETA1 = 0.01
BETA2 = 0.1

# Measurement Parameters
CELL_DIVISION_RHO_K_DM = 5 # For a 5x5 grid for rho_k_dm
# TUNED: More granular box sizes for Df calculation, especially for tight clusters
BOX_SIZES_DF = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6])

# NEW: Cluster Detection Parameters
CLUSTER_DIST_THRESHOLD = 1.0 # QCs within this distance are considered connected
CLUSTER_CP_THRESHOLD = 0.5 # QCs must have CP above this to be part of a "significant" cluster

# --- 3. Measurement Functions ---

def calculate_distance(pos1, pos2, grid_size):
    # Calculate toroidal distance for periodic boundaries
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy) # Corrected bug: use dy, not dx
    return math.sqrt(dx**2 + dy**2)

def calculate_harmonicity(qc1, qc2):
    # Simplified harmonicity based on frequency and phase alignment
    freq_match = abs(qc1.fundamental_frequency_mag - qc2.fundamental_frequency_mag) / \
                 max(qc1.fundamental_frequency_mag, qc2.fundamental_frequency_mag) < HARMONICITY_FREQ_TOL
    phase_diff = abs(qc1.fundamental_frequency_phase - qc2.fundamental_frequency_phase)
    phase_match = min(phase_diff, 2 * math.pi - phase_diff) < HARMONICITY_PHASE_TOL

    if freq_match and phase_match:
        return 0.9 # High coherence
    elif freq_match or phase_match:
        return 0.5 # Partial coherence
    return 0.1 # Low coherence

def calculate_fractal_dimension_box_counting(qc_positions, box_sizes, grid_size):
    if len(qc_positions) < 2:
        return 0.0

    log_N = []
    log_inv_s = []

    # Get min/max coords to define the bounding box of the cluster
    min_x = min(p[0] for p in qc_positions)
    max_x = max(p[0] for p in qc_positions)
    min_y = min(p[1] for p in qc_positions)
    max_y = max(p[1] for p in qc_positions)
    
    # Define a slightly larger bounding box than the cluster itself for box counting
    # This ensures that even for very tight clusters, we have some 'space' to count boxes
    bbox_min_x, bbox_max_x = min_x - 0.1, max_x + 0.1
    bbox_min_y, bbox_max_y = min_y - 0.1, max_y + 0.1
    
    # If the cluster is a single point or very small, ensure a minimal bbox size
    bbox_width = max(0.2, bbox_max_x - bbox_min_x)
    bbox_height = max(0.2, bbox_max_y - bbox_min_y)

    for s in box_sizes:
        if s <= 0: continue
        occupied_boxes = set()
        
        for qc_pos in qc_positions:
            # Determine which box the QC falls into relative to the cluster's bounding box
            box_x_index = math.floor((qc_pos[0] - bbox_min_x) / s)
            box_y_index = math.floor((qc_pos[1] - bbox_min_y) / s)
            occupied_boxes.add((box_x_index, box_y_index))

        N_s = len(occupied_boxes)
        if N_s > 0:
            log_N.append(math.log(N_s))
            # Scale 1/s relative to the bounding box of the cluster, not the entire grid
            log_inv_s.append(math.log(max(bbox_width, bbox_height) / s))

    if len(log_N) < 2:
        return 0.0

    try:
        slope, intercept = np.polyfit(log_inv_s, log_N, 1)
        return slope
    except Exception:
        return 0.0

def calculate_rho_k_dm(qc_list, grid_size, cell_division):
    cell_size_x = grid_size / cell_division
    cell_size_y = grid_size / cell_division

    kinetic_energy_grid = np.zeros((cell_division, cell_division))

    for qc in qc_list:
        v_magnitude_sq = np.dot(qc.velocity, qc.velocity)
        qc_kinetic_energy = 0.5 * qc.effective_mass * v_magnitude_sq

        cell_x_index = math.floor(qc.position[0] / cell_size_x)
        cell_y_index = math.floor(qc.position[1] / cell_size_y)

        cell_x_index = max(0, min(cell_x_index, cell_division - 1))
        cell_y_index = max(0, min(cell_y_index, cell_division - 1))

        kinetic_energy_grid[cell_x_index][cell_y_index] += qc_kinetic_energy
    
    rho_k_dm_grid = np.zeros_like(kinetic_energy_grid)
    cell_volume = cell_size_x * cell_size_y

    if cell_volume > 0:
        rho_k_dm_grid = kinetic_energy_grid / cell_volume
    
    return rho_k_dm_grid

# NEW: Cluster Detection Function
def detect_clusters(qcs, dist_threshold, cp_threshold, grid_size):
    significant_qcs = [qc for qc in qcs if qc.coherence_potential >= cp_threshold]
    if len(significant_qcs) < 2:
        for qc in qcs: qc.cluster_id = -1
        return [] # No clusters if less than 2 significant QCs

    # Build adjacency list for connected components
    adj = {qc.id: [] for qc in significant_qcs}
    for i, qc1 in enumerate(significant_qcs):
        for j, qc2 in enumerate(significant_qcs):
            if i >= j: continue # Avoid self-compare and redundant pairs
            dist = calculate_distance(qc1.position, qc2.position, grid_size)
            if dist <= dist_threshold:
                adj[qc1.id].append(qc2.id)
                adj[qc2.id].append(qc1.id)

    # Find connected components (clusters) using BFS
    visited = set()
    clusters = []
    
    for qc in significant_qcs:
        if qc.id not in visited:
            current_cluster_members = []
            queue = [qc.id]
            visited.add(qc.id)
            
            while queue:
                current_qc_id = queue.pop(0)
                current_qc_obj = next(q for q in significant_qcs if q.id == current_qc_id)
                current_cluster_members.append(current_qc_obj) # Get QC object
                
                for neighbor_id in adj[current_qc_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            
            if len(current_cluster_members) > 1: # Only consider clusters with more than one QC
                clusters.append(current_cluster_members)
            else: # If a single QC, reset its cluster_id
                 current_cluster_members[0].cluster_id = -1

    # Assign cluster_id to QCs
    for i, cluster in enumerate(clusters):
        for qc in cluster:
            qc.cluster_id = i
    
    # For any QC not in a significant cluster, set cluster_id to -1
    for qc in qcs:
        if not any(qc.id == member.id for sublist in clusters for member in sublist):
            qc.cluster_id = -1

    return clusters

# --- 4. Main Execution Block ---

print("--- Genesis Sim-Box: Launching Universe (Refined) ---")

# Initialize QCs
qcs = [QC(i, GRID_SIZE) for i in range(NUM_QCS)]

# Store historical data for analysis
df_history = [] # Will now store a list of Dfs per cluster
rho_k_dm_avg_history = []
avg_coherence_potential_history = []
avg_mass_history = []
num_clusters_history = []

# Bug fix in calculate_distance function for periodic boundaries
# For some reason, the previous fix was only for dx, not dy
def calculate_distance_fixed(pos1, pos2, grid_size):
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy) 
    return math.sqrt(dx**2 + dy**2)

# Monkey-patch the fixed function
globals()['calculate_distance'] = calculate_distance_fixed

for step in range(TOTAL_TIME_STEPS):
    # --- Interaction & Data Acquisition Phase ---
    for i, qc_i in enumerate(qcs):
        neighbors = []
        for j, qc_j in enumerate(qcs):
            if i == j: continue
            dist = calculate_distance(qc_i.position, qc_j.position, GRID_SIZE)
            if dist < R_INT:
                neighbors.append((qc_j, dist))
        
        if neighbors:
            avg_harmonicity = sum(calculate_harmonicity(qc_i, n[0]) for n in neighbors) / len(neighbors)
            qc_i.local_phase_coherence = (qc_i.local_phase_coherence + avg_harmonicity) / 2

            most_coherent_neighbor = None
            max_coherence_score = -1
            
            for neighbor_qc, dist in neighbors:
                current_coherence_score = calculate_harmonicity(qc_i, neighbor_qc) * (1 / (dist + 0.01))
                if current_coherence_score > max_coherence_score:
                    max_coherence_score = current_coherence_score
                    most_coherent_neighbor = neighbor_qc

            if most_coherent_neighbor and max_coherence_score > 0.1:
                w_i = qc_i.coherence_potential / (qc_i.coherence_potential + most_coherent_neighbor.coherence_potential + 1e-6)
                w_j = 1 - w_i

                blended_sig = {
                    'freq_mag': w_i * qc_i.fundamental_frequency_mag + w_j * most_coherent_neighbor.fundamental_frequency_mag,
                    'freq_phase': w_i * qc_i.fundamental_frequency_phase + w_j * most_coherent_neighbor.fundamental_frequency_phase,
                    'lpc': w_i * qc_i.local_phase_coherence + w_j * most_coherent_neighbor.local_phase_coherence,
                    'cp': w_i * qc_i.coherence_potential + w_j * most_coherent_neighbor.coherence_potential,
                    'ihm': w_i * qc_i.interaction_history_metric + w_j * most_coherent_neighbor.interaction_history_metric
                }
                
                qc_i.update_from_blend(blended_sig)
                qc_i.coherence_potential = min(1.0, qc_i.coherence_potential + 0.1)
                qc_i.interaction_history_metric += 1

    # --- Weight & Movement Evolution Phase ---
    for qc in qcs:
        qc.effective_mass = BASE_MASS + BETA1 * qc.interaction_history_metric + BETA2 * qc.coherence_potential

        target_direction = np.array([0.0, 0.0])
        total_coherence_influence = 0.0

        for other_qc in qcs:
            if qc.id == other_qc.id: continue
            dist = calculate_distance(qc.position, other_qc.position, GRID_SIZE)
            
            influence = qc.coherence_potential * other_qc.coherence_potential / (dist**2 + 1e-6)
            influence *= calculate_harmonicity(qc, other_qc)

            if influence > 0.05:
                target_direction += (other_qc.position - qc.position) * influence
                total_coherence_influence += influence
        
        if total_coherence_influence > 0:
            target_direction = target_direction / total_coherence_influence
            qc.velocity += ETA * qc.coherence_potential * target_direction * DT
        else:
            qc.velocity += np.array([random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]) * DT

    # --- Apply Movement ---
    for qc in qcs:
        qc.position += qc.velocity * DT
        qc.position = qc.position % GRID_SIZE

    # --- Measurement & Recording Phase (Refined) ---
    current_dfs = []
    
    # NEW: Detect clusters
    clusters = detect_clusters(qcs, CLUSTER_DIST_THRESHOLD, CLUSTER_CP_THRESHOLD, GRID_SIZE)
    num_clusters_history.append(len(clusters))

    if clusters:
        for cluster in clusters:
            cluster_positions = [qc.position for qc in cluster]
            # Calculate Df for each detected cluster
            df_cluster = calculate_fractal_dimension_box_counting(cluster_positions, BOX_SIZES_DF, GRID_SIZE)
            if df_cluster > 0: # Only record meaningful Df values
                current_dfs.append(df_cluster)
        
        if current_dfs:
            df_history.append(np.mean(current_dfs)) # Store average Df of all detected clusters
        else:
            df_history.append(0.0)
    else:
        df_history.append(0.0) # No clusters, no Df

    rho_k_dm_grid = calculate_rho_k_dm(qcs, GRID_SIZE, CELL_DIVISION_RHO_K_DM)
    rho_k_dm_avg_history.append(np.mean(rho_k_dm_grid))
    avg_coherence_potential_history.append(np.mean([qc.coherence_potential for qc in qcs]))
    avg_mass_history.append(np.mean([qc.effective_mass for qc in qcs]))

    # --- Progress Output (every 20 steps) ---
    if (step + 1) % 20 == 0 or step == 0:
        print(f"\n--- Time Step {step+1}/{TOTAL_TIME_STEPS} ---")
        print(f" Average Coherence Potential: {avg_coherence_potential_history[-1]:.3f}")
        print(f" Average Effective Mass: {avg_mass_history[-1]:.3f}")
        if current_dfs:
            print(f" Avg Fractal Dimension (Df) of Clusters: {df_history[-1]:.3f} (from {len(current_dfs)} clusters)")
        else:
            print(f" No significant clusters detected for Df calculation.")
        print(f" Average Kinetic Energy Density (ρ_K_DM): {rho_k_dm_avg_history[-1]:.4f}")
        print(" QC Positions (x, y), Mass, CP, Cluster_ID:")
        for qc in qcs:
            print(f" QC{qc.id}: ({qc.position[0]:.2f}, {qc.position[1]:.2f}), M:{qc.effective_mass:.2f}, CP:{qc.coherence_potential:.2f}, ClID:{qc.cluster_id}")

print("\n--- Simulation Complete ---")

# --- Final Analysis (Illustrative) ---
print("\n--- Final Summary ---")
print(f"Initial Average Df (Clusters): {df_history[0]:.3f}, Final Average Df (Clusters): {df_history[-1]:.3f}")
print(f"Initial Average ρ_K_DM: {rho_k_dm_avg_history[0]:.4f}, Final Average ρ_K_DM: {rho_k_dm_avg_history[-1]:.4f}")
print(f"Initial Avg Coherence Potential: {avg_coherence_potential_history[0]:.3f}, Final Avg Coherence Potential: {avg_coherence_potential_history[-1]:.3f}")
print(f"Initial Avg Effective Mass: {avg_mass_history[0]:.3f}, Final Avg Effective Mass: {avg_mass_history[-1]:.3f}")
print(f"Total Clusters Formed: {num_clusters_history[-1]} (at final step)")

final_qc_positions = np.array([qc.position for qc in qcs])
distances = []
for i in range(NUM_QCS):
    for j in range(i + 1, NUM_QCS):
        distances.append(calculate_distance(final_qc_positions[i], final_qc_positions[j], GRID_SIZE))

if np.mean(distances) < GRID_SIZE / 2:
    print(f"\nObservation: QCs appear to have clustered significantly (average pairwise distance: {np.mean(distances):.2f}).")
    print("This suggests emergent 'compressive gravity' at play.")
else:
    print(f"\nObservation: QCs did not show strong clustering (average pairwise distance: {np.mean(distances):.2f}).")
    print("Further parameter tuning or more time steps may be needed.")
