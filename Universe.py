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

# Interaction & Movement Constants
R_INT = 2.0 # Interaction radius
R_MERGE = 0.1 # Merger distance
ALPHA = 0.01 # Interaction strength scaling
HARMONICITY_FREQ_TOL = 0.1 # Freq magnitude tolerance (10%)
HARMONICITY_PHASE_TOL = math.pi / 4 # Phase tolerance (45 degrees)
ETA = 0.05 # Movement learning rate (how aggressively QCs seek coherence)

# Mass Scaling Constants (m_i = m_0 + beta1 * h_i + beta2 * c_i)
BASE_MASS = 1.0
BETA1 = 0.01
BETA2 = 0.1

# Measurement Parameters
CELL_DIVISION_RHO_K_DM = 5 # For a 5x5 grid for rho_k_dm
BOX_SIZES_DF = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2]) # Box sizes for fractal dimension

# --- 3. Measurement Functions ---

def calculate_distance(pos1, pos2, grid_size):
    # Calculate toroidal distance for periodic boundaries
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy)
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

    for s in box_sizes:
        if s <= 0: continue
        occupied_boxes = set()
        
        # Use full grid bounds for box counting
        min_coord, max_coord = 0, grid_size

        for qc_pos in qc_positions:
            # Ensure qc_pos is within grid_size before calculating index
            box_x_index = math.floor((qc_pos[0] - min_coord) / s)
            box_y_index = math.floor((qc_pos[1] - min_coord) / s)
            occupied_boxes.add((box_x_index, box_y_index))

        N_s = len(occupied_boxes)
        if N_s > 0:
            log_N.append(math.log(N_s))
            log_inv_s.append(math.log(grid_size / s)) # Use grid_size for scaling 1/s

    if len(log_N) < 2:
        return 0.0

    # Perform linear regression
    # If using numpy for more robust fit
    try:
        slope, intercept = np.polyfit(log_inv_s, log_N, 1)
        return slope
    except Exception:
        return 0.0 # Fallback if polyfit fails (e.g., all points collinear or no variance)

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

# --- 4. Main Execution Block ---

print("--- Genesis Sim-Box: Launching Universe ---")

# Initialize QCs
qcs = [QC(i, GRID_SIZE) for i in range(NUM_QCS)]

# Store historical data for analysis
df_history = []
rho_k_dm_avg_history = []
avg_coherence_potential_history = []
avg_mass_history = []

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
            # Update local_phase_coherence based on neighbors
            # For simplicity, average harmonicity with all neighbors
            avg_harmonicity = sum(calculate_harmonicity(qc_i, n[0]) for n in neighbors) / len(neighbors)
            qc_i.local_phase_coherence = (qc_i.local_phase_coherence + avg_harmonicity) / 2 # Smoothed update

            # Check for Merger (simplified: strong interaction leads to blending)
            # Only blend with the closest/most coherent neighbor
            most_coherent_neighbor = None
            max_coherence_score = -1
            
            for neighbor_qc, dist in neighbors:
                current_coherence_score = calculate_harmonicity(qc_i, neighbor_qc) * (1 / (dist + 0.01)) # Factor in distance
                if current_coherence_score > max_coherence_score:
                    max_coherence_score = current_coherence_score
                    most_coherent_neighbor = neighbor_qc

            if most_coherent_neighbor and max_coherence_score > 0.5: # If there's a strong enough interaction
                # Data Signature Blending
                w_i = qc_i.coherence_potential / (qc_i.coherence_potential + most_coherent_neighbor.coherence_potential + 1e-6)
                w_j = 1 - w_i

                blended_sig = {
                    'freq_mag': w_i * qc_i.fundamental_frequency_mag + w_j * most_coherent_neighbor.fundamental_frequency_mag,
                    'freq_phase': w_i * qc_i.fundamental_frequency_phase + w_j * most_coherent_neighbor.fundamental_frequency_phase,
                    'lpc': w_i * qc_i.local_phase_coherence + w_j * most_coherent_neighbor.local_phase_coherence,
                    'cp': w_i * qc_i.coherence_potential + w_j * most_coherent_neighbor.coherence_potential,
                    'ihm': w_i * qc_i.interaction_history_metric + w_j * most_coherent_neighbor.interaction_history_metric
                }
                
                # Update QC's properties from blend
                qc_i.update_from_blend(blended_sig)
                
                # Increase coherence potential and interaction history
                qc_i.coherence_potential = min(1.0, qc_i.coherence_potential + 0.05) # Cap at 1.0
                qc_i.interaction_history_metric += 1

    # --- Weight & Movement Evolution Phase ---
    for qc in qcs:
        # Update effective_mass
        qc.effective_mass = BASE_MASS + BETA1 * qc.interaction_history_metric + BETA2 * qc.coherence_potential

        # Determine Coherence-Seeking Movement (simplified: move towards weighted average of coherent neighbors)
        target_direction = np.array([0.0, 0.0])
        total_coherence_influence = 0.0

        for other_qc in qcs:
            if qc.id == other_qc.id: continue
            dist = calculate_distance(qc.position, other_qc.position, GRID_SIZE)
            
            # Stronger influence from closer, more coherent QCs
            influence = qc.coherence_potential * other_qc.coherence_potential / (dist**2 + 1e-6) # Add small epsilon to avoid div by zero
            influence *= calculate_harmonicity(qc, other_qc) # Factor in harmonicity

            if influence > 0.1: # Only influenced by sufficiently coherent neighbors
                target_direction += (other_qc.position - qc.position) * influence
                total_coherence_influence += influence
        
        if total_coherence_influence > 0:
            target_direction = target_direction / total_coherence_influence
            # Update velocity towards target, scaled by learning rate and coherence potential
            qc.velocity += ETA * qc.coherence_potential * target_direction * DT # Scaling by DT for more stable integration
        else:
            # If no strong coherence influence, apply some random drift (minor Brownian motion)
            qc.velocity += np.array([random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)]) * DT

    # --- Apply Movement ---
    for qc in qcs:
        qc.position += qc.velocity * DT
        # Apply periodic boundary conditions
        qc.position = qc.position % GRID_SIZE

    # --- Measurement & Recording Phase ---
    qc_positions_for_df = [qc.position for qc in qcs if qc.coherence_potential > 0.2] # Only measure Df for somewhat coherent QCs
    df = calculate_fractal_dimension_box_counting(qc_positions_for_df, BOX_SIZES_DF, GRID_SIZE)
    rho_k_dm_grid = calculate_rho_k_dm(qcs, GRID_SIZE, CELL_DIVISION_RHO_K_DM)
    
    df_history.append(df)
    rho_k_dm_avg_history.append(np.mean(rho_k_dm_grid)) # Average rho_k_dm across the grid
    avg_coherence_potential_history.append(np.mean([qc.coherence_potential for qc in qcs]))
    avg_mass_history.append(np.mean([qc.effective_mass for qc in qcs]))

    # --- Progress Output (every 20 steps) ---
    if (step + 1) % 20 == 0 or step == 0:
        print(f"\n--- Time Step {step+1}/{TOTAL_TIME_STEPS} ---")
        print(f" Average Coherence Potential: {avg_coherence_potential_history[-1]:.3f}")
        print(f" Average Effective Mass: {avg_mass_history[-1]:.3f}")
        print(f" Calculated Fractal Dimension (Df): {df:.3f}")
        print(f" Average Kinetic Energy Density (ρ_K_DM): {rho_k_dm_avg_history[-1]:.4f}")
        print(" QC Positions (x, y), Mass, CP:")
        for qc in qcs:
            print(f" QC{qc.id}: ({qc.position[0]:.2f}, {qc.position[1]:.2f}), M:{qc.effective_mass:.2f}, CP:{qc.coherence_potential:.2f}")

print("\n--- Simulation Complete ---")

