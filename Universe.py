import streamlit as st
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For color mapping
import time # To simulate time steps

# --- 1. QC Class Definition ---
class QC:
    def __init__(self, id, grid_size, base_cp=0.1, is_passive=False):
        self.id = id
        self.position = np.array([random.uniform(0, grid_size), random.uniform(0, grid_size)])
        self.velocity = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
        
        # Data Signature Components - REVISED FREQUENCY RANGE TO ALLOW HIGHER INTRINSIC DIMENSIONS
        # The range is now 1.0 to 8.0, allowing QCs to express higher dimensional energy
        self.fundamental_frequency_mag = random.uniform(1.0, 8.0) 
        self.fundamental_frequency_phase = random.uniform(0, 2 * math.pi) # 0 to 2pi
        self.local_phase_coherence = 0.5 # Normalized 0 to 1
        self.coherence_potential = base_cp # Dynamic scalar, starts low
        self.interaction_history_metric = 0 # Cumulative interactions

        # Derived Properties
        self.effective_mass = 1.0 # Starts at base mass
        self.cluster_id = -1 # To track which cluster it belongs to
        self.is_passive = is_passive # Flag for passive QCs

    def get_data_signature(self):
        return {
            'freq_mag': self.fundamental_frequency_mag,
            'freq_phase': self.fundamental_frequency_phase,
            'lpc': self.local_phase_coherence,
            'cp': self.coherence_potential,
            'ihm': self.interaction_history_metric
        }

    def update_from_blend(self, blended_signature):
        if not self.is_passive: # Passive QCs don't update their internal state actively
            self.fundamental_frequency_mag = blended_signature['freq_mag']
            self.fundamental_frequency_phase = blended_signature['freq_phase'] % (2 * math.pi) 
            self.local_phase_coherence = blended_signature['lpc']
            self.coherence_potential = blended_signature['cp']
            self.interaction_history_metric = blended_signature['ihm']

# --- 3. Measurement Functions (Moved here for better organization) ---

def calculate_distance(pos1, pos2, grid_size):
    # Calculate toroidal distance for periodic boundaries
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy) 
    return math.sqrt(dx**2 + dy**2)

def calculate_harmonicity(qc1, qc2, HARMONICITY_FREQ_TOL, HARMONICITY_PHASE_TOL):
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
    # Use max of grid_size and current bbox_dim for log_inv_s denominator
    effective_dim_for_log = max(grid_size, max(bbox_width, bbox_height))

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
            log_inv_s.append(math.log(effective_dim_for_log / s))

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

def detect_clusters(qcs, dist_threshold, cp_threshold, grid_size):
    all_qcs_map = {qc.id: qc for qc in qcs}
    
    significant_active_qcs = [qc for qc in qcs if not qc.is_passive and qc.coherence_potential >= cp_threshold]
    
    if len(significant_active_qcs) < 2:
        for qc in qcs: qc.cluster_id = -1
        return []

    adj = {qc.id: [] for qc in qcs}
    for i, qc1 in enumerate(qcs):
        for j, qc2 in enumerate(qcs):
            if i == j: continue 
            dist = calculate_distance(qc1.position, qc2.position, grid_size)
            if dist <= dist_threshold:
                adj[qc1.id].append(qc2.id)
                adj[qc2.id].append(qc1.id)

    visited_for_components = set()
    clusters = []

    for qc_start in significant_active_qcs:
        if qc_start.id not in visited_for_components:
            current_component_members = []
            queue = [qc_start.id]
            visited_for_components.add(qc_start.id)
            
            head_of_queue_idx = 0
            while head_of_queue_idx < len(queue):
                current_qc_id = queue[head_of_queue_idx]
                head_of_queue_idx += 1
                
                current_qc_obj = all_qcs_map[current_qc_id]
                current_component_members.append(current_qc_obj)
                
                for neighbor_id in adj[current_qc_id]:
                    if neighbor_id not in visited_for_components:
                        visited_for_components.add(neighbor_id)
                        queue.append(neighbor_id)
            
            cluster_significant_active_members = [
                qc for qc in current_component_members 
                if not qc.is_passive and qc.coherence_potential >= cp_threshold
            ]

            if len(cluster_significant_active_members) > 1:
                clusters.append(cluster_significant_active_members)

    assigned_qc_ids = {qc.id for cluster in clusters for qc in cluster}
    for qc in qcs:
        if qc.id not in assigned_qc_ids:
            qc.cluster_id = -1

    return clusters

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🌌 CompresSim: Exploring Coherence-Driven Cosmology with Frequency-Dimension Link")

st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("General Settings", expanded=True):
    NUM_QCS = st.slider("Number of QCs", 10, 1000, 500, 10) # Increased max QCs
    GRID_SIZE = st.slider("Grid Size", 50, 1000, 500, 25) # Increased max Grid Size
    TOTAL_TIME_STEPS = st.slider("Total Time Steps", 100, 2000, 1000, 50)
    DT = st.slider("Time Step Duration (DT)", 0.01, 0.5, 0.1, 0.01)

with st.sidebar.expander("Interaction & Movement", expanded=True):
    R_INT = st.slider("Interaction Radius (R_INT)", 5.0, 100.0, 50.0, 5.0) # Increased max R_INT
    HARMONICITY_FREQ_TOL = st.slider("Harmonicity Freq Tolerance", 0.05, 0.5, 0.4, 0.05)
    HARMONICITY_PHASE_TOL = st.slider("Harmonicity Phase Tolerance", 0.1, math.pi, math.pi * 0.9, 0.1)
    ETA = st.slider("Movement Learning Rate (ETA)", 0.1, 2.0, 2.0, 0.1)

with st.sidebar.expander("Cluster & Measurement", expanded=True):
    CLUSTER_DIST_THRESHOLD = st.slider("Cluster Distance Threshold", 1.0, 50.0, 30.0, 1.0) # Increased max CLUSTER_DIST_THRESHOLD
    CLUSTER_CP_THRESHOLD = st.slider("Coherence Potential Threshold for Df", 0.1, 1.0, 0.5, 0.1)
    PASSIVE_QC_PERCENTAGE = st.slider("Passive QC Percentage", 0.0, 0.5, 0.1, 0.05)

with st.sidebar.expander("Frequency Bands for Df Analysis", expanded=True):
    st.write("Define frequency bands (New range: 1.0 to 8.0 for fundamental_frequency_mag)")
    # IMPORTANT: User must manually adjust these for the new 1.0-8.0 range
    freq_band_low_max = st.slider("Max Freq Mag for LOW band (1.0 to X)", 1.5, 3.0, 2.0, 0.1) # Adjusted min/max for new range
    freq_band_mid_max = st.slider(f"Max Freq Mag for MID band ({freq_band_low_max:.1f} to X)", freq_band_low_max + 0.1, 7.5, 4.0, 0.1) # Adjusted min/max for new range
    # High band is automatically from freq_band_mid_max to 8.0

# Fixed parameters not exposed to UI (or less critical for quick tuning)
R_MERGE = 0.1 
ALPHA = 0.01
BASE_MASS = 1.0
BETA1 = 0.01
BETA2 = 0.1
CELL_DIVISION_RHO_K_DM = int(GRID_SIZE / 4) # Maintain ~4x4 unit cell for rho_k_dm
BOX_SIZES_DF = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]) # Wider range of box sizes for Df

if st.sidebar.button("Run Simulation", type="primary"):
    st.session_state.running_simulation = True
    st.session_state.current_step = 0
    st.session_state.progress_bar_value = 0
    st.session_state.df_overall_history = []
    st.session_state.df_low_freq_history = []
    st.session_state.df_mid_freq_history = []
    st.session_state.df_high_freq_history = []
    st.session_state.rho_k_dm_avg_history = []
    st.session_state.avg_coherence_potential_history = []
    st.session_state.avg_mass_history = []
    st.session_state.num_clusters_history = []

    # Initialize QCs
    qcs = []
    num_passive_qcs = int(NUM_QCS * PASSIVE_QC_PERCENTAGE)
    for i in range(NUM_QCS):
        if i < num_passive_qcs:
            qcs.append(QC(i, GRID_SIZE, base_cp=0.1, is_passive=True))
        else:
            qcs.append(QC(i, GRID_SIZE, base_cp=0.1, is_passive=False))
    st.session_state.qcs = qcs

    # --- Live Output Placeholders ---
    st.markdown("### Simulation Progress")
    progress_text = st.empty()
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()
    qc_details_placeholder = st.empty()
    plot_placeholder = st.empty()

    # --- Simulation Loop ---
    for step in range(TOTAL_TIME_STEPS):
        # Update progress bar and text
        st.session_state.current_step = step + 1
        st.session_state.progress_bar_value = (step + 1) / TOTAL_TIME_STEPS
        progress_bar.progress(st.session_state.progress_bar_value)
        progress_text.text(f"Running Time Step {st.session_state.current_step}/{TOTAL_TIME_STEPS}")

        # --- Interaction & Data Acquisition Phase ---
        for i, qc_i in enumerate(st.session_state.qcs):
            if qc_i.is_passive: continue

            neighbors = []
            for j, qc_j in enumerate(st.session_state.qcs):
                if i == j: continue
                dist = calculate_distance(qc_i.position, qc_j.position, GRID_SIZE)
                if dist < R_INT:
                    neighbors.append((qc_j, dist))
            
            if neighbors:
                avg_harmonicity = sum(calculate_harmonicity(qc_i, n[0], HARMONICITY_FREQ_TOL, HARMONICITY_PHASE_TOL) for n in neighbors) / len(neighbors)
                qc_i.local_phase_coherence = (qc_i.local_phase_coherence + avg_harmonicity) / 2

                most_coherent_neighbor = None
                max_coherence_score = -1
                
                for neighbor_qc, dist in neighbors:
                    current_coherence_score = calculate_harmonicity(qc_i, neighbor_qc, HARMONICITY_FREQ_TOL, HARMONICITY_PHASE_TOL) * (1 / (dist + 0.01)) 
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
        for qc in st.session_state.qcs:
            qc.effective_mass = BASE_MASS + BETA1 * qc.interaction_history_metric + BETA2 * qc.coherence_potential

            target_direction = np.array([0.0, 0.0])
            total_coherence_influence = 0.0

            for other_qc in st.session_state.qcs:
                if qc.id == other_qc.id: continue
                dist = calculate_distance(qc.position, other_qc.position, GRID_SIZE)
                
                influence = other_qc.coherence_potential / (dist**2 + 1e-6)
                influence *= calculate_harmonicity(qc, other_qc, HARMONICITY_FREQ_TOL, HARMONICITY_PHASE_TOL)

                if influence > 0.05:
                    target_direction += (other_qc.position - qc.position) * influence
                    total_coherence_influence += influence
            
            if total_coherence_influence > 0:
                target_direction = target_direction / total_coherence_influence
                if not qc.is_passive:
                    qc.velocity += ETA * qc.coherence_potential * target_direction * DT
                else:
                    qc.velocity += (ETA / 2.0) * target_direction * DT 
            else:
                qc.velocity += np.array([random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]) * DT

        # --- Apply Movement ---
        for qc in st.session_state.qcs:
            qc.position += qc.velocity * DT
            qc.position = qc.position % GRID_SIZE

        # --- Measurement & Recording Phase ---
        # 1. Overall Coherent Df (of all QCs with CP >= threshold)
        all_coherent_qcs_for_df = [qc for qc in st.session_state.qcs if not qc.is_passive and qc.coherence_potential >= CLUSTER_CP_THRESHOLD]
        if len(all_coherent_qcs_for_df) > 1:
            df_overall = calculate_fractal_dimension_box_counting([qc.position for qc in all_coherent_qcs_for_df], BOX_SIZES_DF, GRID_SIZE)
            st.session_state.df_overall_history.append(df_overall)
        else:
            st.session_state.df_overall_history.append(0.0)

        # 2. Df by Frequency Bands (now spanning 1.0 to 8.0)
        df_low, df_mid, df_high = 0.0, 0.0, 0.0

        # Low Frequency Band (from 1.0 to freq_band_low_max)
        low_freq_qcs = [qc for qc in all_coherent_qcs_for_df if qc.fundamental_frequency_mag >= 1.0 and qc.fundamental_frequency_mag <= freq_band_low_max]
        if len(low_freq_qcs) > 1:
            df_low = calculate_fractal_dimension_box_counting([qc.position for qc in low_freq_qcs], BOX_SIZES_DF, GRID_SIZE)
        st.session_state.df_low_freq_history.append(df_low)

        # Mid Frequency Band (from freq_band_low_max to freq_band_mid_max)
        mid_freq_qcs = [qc for qc in all_coherent_qcs_for_df if qc.fundamental_frequency_mag > freq_band_low_max and qc.fundamental_frequency_mag <= freq_band_mid_max]
        if len(mid_freq_qcs) > 1:
            df_mid = calculate_fractal_dimension_box_counting([qc.position for qc in mid_freq_qcs], BOX_SIZES_DF, GRID_SIZE)
        st.session_state.df_mid_freq_history.append(df_mid)

        # High Frequency Band (from freq_band_mid_max to 8.0)
        high_freq_qcs = [qc for qc in all_coherent_qcs_for_df if qc.fundamental_frequency_mag > freq_band_mid_max and qc.fundamental_frequency_mag <= 8.0]
        if len(high_freq_qcs) > 1:
            df_high = calculate_fractal_dimension_box_counting([qc.position for qc in high_freq_qcs], BOX_SIZES_DF, GRID_SIZE)
        st.session_state.df_high_freq_history.append(df_high)

        # 3. Traditional Cluster Detection (for counting)
        clusters = detect_clusters(st.session_state.qcs, CLUSTER_DIST_THRESHOLD, CLUSTER_CP_THRESHOLD, GRID_SIZE)
        st.session_state.num_clusters_history.append(len(clusters))

        rho_k_dm_grid = calculate_rho_k_dm(st.session_state.qcs, GRID_SIZE, CELL_DIVISION_RHO_K_DM)
        st.session_state.rho_k_dm_avg_history.append(np.mean(rho_k_dm_grid))
        st.session_state.avg_coherence_potential_history.append(np.mean([qc.coherence_potential for qc in st.session_state.qcs]))
        st.session_state.avg_mass_history.append(np.mean([qc.effective_mass for qc in st.session_state.qcs]))

        # --- Update Live Metrics & Plot ---
        if st.session_state.current_step % 10 == 0 or st.session_state.current_step == TOTAL_TIME_STEPS: # Update more frequently
            with metrics_placeholder.container():
                st.markdown(f"**Current Metrics (Step {st.session_state.current_step}/{TOTAL_TIME_STEPS})**")
                st.write(f" Overall Coherent Df: {st.session_state.df_overall_history[-1]:.3f}")
                st.write(f" Df (Low Freq Band): {df_low:.3f}")
                st.write(f" Df (Mid Freq Band): {df_mid:.3f}")
                st.write(f" Df (High Freq Band): {df_high:.3f}")
                st.write(f" Avg Coherence Potential: {st.session_state.avg_coherence_potential_history[-1]:.3f}")
                st.write(f" Avg Effective Mass: {st.session_state.avg_mass_history[-1]:.3f}")
                st.write(f" Avg Kinetic Energy Density (ρ_K_DM): {st.session_state.rho_k_dm_avg_history[-1]:.4f}")
                st.write(f" Number of Detected Clusters: {len(clusters)}")
            
            with qc_details_placeholder.container():
                st.markdown("**QC Details (subset)**")
                details_text = ""
                printed_count = 0
                sorted_qcs = sorted(st.session_state.qcs, key=lambda qc: (qc.cluster_id == -1, qc.is_passive, qc.id))
                for qc in sorted_qcs:
                    if qc.cluster_id!= -1 or qc.is_passive:
                        details_text += f"QC{qc.id}: ({qc.position[0]:.2f}, {qc.position[1]:.2f}), M:{qc.effective_mass:.2f}, CP:{qc.coherence_potential:.2f}, Freq:{qc.fundamental_frequency_mag:.2f}, ClID:{qc.cluster_id}, Passive:{qc.is_passive}\n"
                        printed_count += 1
                    if printed_count >= 15:
                        details_text += "... (truncated QCs)\n"
                        break
                st.text(details_text)

            # Plot QC positions, colored by frequency, sized by mass
            with plot_placeholder.container():
                fig, ax = plt.subplots(figsize=(10, 10))
                
                x = [qc.position[0] for qc in st.session_state.qcs]
                y = [qc.position[1] for qc in st.session_state.qcs]
                
                freq_mags = [qc.fundamental_frequency_mag for qc in st.session_state.qcs]
                min_freq, max_freq = 1.0, 8.0 # Updated min/max for color mapping to the new range
                
                # Adjust size based on effective_mass, with a minimum size
                sizes = [max(10, qc.effective_mass * 5) for qc in st.session_state.qcs]

                # Use plasma colormap for frequencies
                cmap_freq = cm.get_cmap('plasma')
                
                scatter = ax.scatter(x, y, c=freq_mags, cmap=cmap_freq, vmin=min_freq, vmax=max_freq, s=sizes, alpha=0.7, edgecolors='w', linewidth=0.5)
                
                ax.set_xlim(0, GRID_SIZE)
                ax.set_ylim(0, GRID_SIZE)
                ax.set_title(f"QC Distribution (Step {st.session_state.current_step}) - Color by Frequency, Size by Mass")
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
                ax.set_aspect('equal', adjustable='box')
                
                # Colorbar for frequencies
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label("Fundamental Frequency Magnitude")
                
                st.pyplot(fig)
                plt.close(fig) # Close the figure to free up memory

    st.success("Simulation Complete!")

    # --- Final Summary ---
    st.markdown("### Final Summary")
    st.write(f"Initial Overall Coherent Df: {st.session_state.df_overall_history[0]:.3f}, Final Overall Coherent Df: {st.session_state.df_overall_history[-1]:.3f}")
    st.write(f"Final Df (Low Freq Band): {st.session_state.df_low_freq_history[-1]:.3f}")
    st.write(f"Final Df (Mid Freq Band): {st.session_state.df_mid_freq_history[-1]:.3f}")
    st.write(f"Final Df (High Freq Band): {st.session_state.df_high_freq_history[-1]:.3f}")
    st.write(f"Initial Avg Kinetic Energy Density (ρ_K_DM): {st.session_state.rho_k_dm_avg_history[0]:.4f}, Final Avg ρ_K_DM: {st.session_state.rho_k_dm_avg_history[-1]:.4f}")
    st.write(f"Initial Avg Coherence Potential: {st.session_state.avg_coherence_potential_history[0]:.3f}, Final Avg Coherence Potential: {st.session_state.avg_coherence_potential_history[-1]:.3f}")
    st.write(f"Initial Avg Effective Mass: {st.session_state.avg_mass_history[0]:.3f}, Final Avg Effective Mass: {st.session_state.avg_mass_history[-1]:.3f}")
    st.write(f"Total Clusters Formed: {st.session_state.num_clusters_history[-1]} (at final step)")

    final_qc_positions = np.array([qc.position for qc in st.session_state.qcs])
    distances = []
    # Only calculate for active QCs for more meaningful average distance
    active_qc_positions = np.array([qc.position for qc in st.session_state.qcs if not qc.is_passive])
    if len(active_qc_positions) > 1:
        for i in range(len(active_qc_positions)):
            for j in range(i + 1, len(active_qc_positions)):
                distances.append(calculate_distance(active_qc_positions[i], active_qc_positions[j], GRID_SIZE))

    if distances and np.mean(distances) < GRID_SIZE / 2:
        st.write(f"Observation: Active QCs appear to have clustered significantly (average pairwise distance: {np.mean(distances):.2f}).")
        st.write("This suggests emergent 'compressive gravity' at play.")
    else:
        st.write(f"Observation: Active QCs did not show strong clustering (average pairwise distance: {np.mean(distances):.2f}).")
        st.write("Further parameter tuning or more time steps may be needed.")

    st.session_state.running_simulation = False
