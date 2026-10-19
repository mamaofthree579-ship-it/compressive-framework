import numpy as np

def parse_yantra_symbol(symbol_name):
    """Translates Sanskrit geometric algorithm intents into active scalar variables."""
    matrix_maps = {
        "Shri Yantra (Linear Vector Propagation)": {
            "mode": "Propulsion & Maneuvers", "asymmetry": 0.75, "eeg_target": "Beta (Focus)"
        },
        "Shiva-Shakti Kundalini (Vortex Coherence)": {
            "mode": "Propulsion & Maneuvers", "asymmetry": 1.20, "eeg_target": "Gamma (High Coherence)"
        },
        "Chakra Mandala (Isotropic Sphere Mask)": {
            "mode": "Invisibility System", "diffraction_mod": 1.5, "eeg_target": "Alpha (Flow State)"
        },
        "Trishula Vector (Directed Focus Point)": {
            "mode": "Defensive Beam", "beam_intensity": 2.5, "eeg_target": "Gamma (High Coherence)"
        }
    }
    return matrix_maps.get(symbol_name, {"mode": "Propulsion & Maneuvers", "asymmetry": 0.5, "eeg_target": "Alpha"})

def calculate_biometallurgy(alloy_type, temp):
    """
    Models solid-state atomic diffusion kinetics (D = D0 * exp(-Q / (R * T)))
    to analyze structural crystal lattice saturation points.
    """
    R = 8.314  # Universal Gas Constant
    if alloy_type == "Kundodara Loha (Engine Core)":
        D0, Q = 1.2e-4, 210000  # Tungsten-Carbide diffusion profiles
        matrix_name = "Tungsten-Carbide Ceramic Matrix"
    elif alloy_type == "Raajaloha (Airframe Hull)":
        D0, Q = 4.5e-5, 142000  # Aluminum-Scandium carbon matrix
        matrix_name = "Al-Sc Carbon-Fiber Lattice"
    else:  # Vishvodara Metal
        D0, Q = 8.1e-6, 85000   # Quartz-Gold-Mercury amalgam
        matrix_name = "Piezo-Resonant Quartz Substrate"
    
    # Calculate continuous diffusion coefficient across temperatures
    diffusion_coeff = D0 * np.exp(-Q / (R * temp))
    saturation_pct = min(100.0, 100.0 * (1 - np.exp(-diffusion_coeff * 500000)))
    return matrix_name, diffusion_coeff, saturation_pct

def calculate_aerodynamics(velocity_mach, density_kg_m3, spatial_well_depth):
    """
    Computes fluid-medium atmospheric drag factoring in spacetime curvature reduction.
    """
    Cd_base = 0.45 if velocity_mach < 1.0 else 0.85 * (velocity_mach ** 0.3)
    # The gravity well compresses local air, reducing effective drag metrics
    drag_attenuation = np.exp(-spatial_well_depth / 80.0)
    effective_Cd = Cd_base * drag_attenuation
    
    # Drag force equation: Fd = 0.5 * rho * v^2 * Cd * Area (Assuming uniform 15m^2 surface area)
    velocity_ms = velocity_mach * 343.0
    drag_force_kn = (0.5 * density_kg_m3 * (velocity_ms ** 2) * effective_Cd * 15.0) / 1000.0
    return effective_Cd, drag_force_kn
