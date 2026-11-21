import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance

st.set_page_config(
    page_title="UCM Unified Benchmark & Falsification Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 1. CORE OPERATOR ENGINE (Book Two, Ch. 14, 17, 19, 23)
# ==============================================================================

def build_1d_grid(n_points: int = 512, span: float = 4.0):
    x = np.linspace(-span / 2.0, span / 2.0, n_points, endpoint=False)
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n_points, d=dx)
    return x, dx, k


def compression_operator(psi: np.ndarray, dx: float, kappa: float, alpha: float) -> np.ndarray:
    """
    Conservative finite-difference discretization of Q[psi] = -kappa * div(rho^alpha * grad(psi))
    (Book Two, Ch. 17.4).
    """
    if kappa == 0.0:
        return np.zeros_like(psi, dtype=np.complex128)
    rho = np.abs(psi) ** 2
    A = np.power(rho + 1e-12, alpha)
    A_plus = 0.5 * (A + np.roll(A, -1))
    A_minus = 0.5 * (np.roll(A, 1) + A)
    grad_plus = np.roll(psi, -1) - psi
    grad_minus = psi - np.roll(psi, 1)
    div_flux = (A_plus * grad_plus - A_minus * grad_minus) / (dx ** 2)
    # Attractive compressive sign convention (+kappa drives inward clustering; Theorem 8)
    return kappa * div_flux


def information_operator(psi: np.ndarray, dx: float, gamma_I: float) -> np.ndarray:
    """
    Regularized information-gradient operator I[psi] = -gamma_I * grad(I(rho)) . grad(psi)
    (Book Two, Ch. 16.2 & 17.6).
    """
    if gamma_I == 0.0:
        return np.zeros_like(psi, dtype=np.complex128)
    rho = np.abs(psi) ** 2
    info_density = -rho * np.log(rho + 1e-12)
    grad_I = (np.roll(info_density, -1) - np.roll(info_density, 1)) / (2.0 * dx)
    grad_psi = (np.roll(psi, -1) - np.roll(psi, 1)) / (2.0 * dx)
    return -gamma_I * grad_I * grad_psi


def classical_gravity_potential(rho: np.ndarray, k: np.ndarray, G_eff: float = 1.0) -> np.ndarray:
    """Spectral Poisson solver for classical emergent gravity potential Phi_c (Ch. 14.5)."""
    delta_rho = rho - np.mean(rho)
    rho_k = np.fft.fft(delta_rho)
    k2 = k ** 2
    k2[0] = 1.0  # avoid division by zero at k=0
    phi_k = -4.0 * np.pi * G_eff * rho_k / k2
    phi_k[0] = 0.0
    return np.real(np.fft.ifft(phi_k))


def evolve_strang_1d(
    psi0: np.ndarray,
    dx: float,
    k: np.ndarray,
    dt: float,
    steps: int,
    kappa: float = 0.005,
    alpha: float = 2.0,
    gamma_I: float = 0.002,
    G_eff: float = 0.8,
):
    """
    Second-order Strang splitting integrator (Book Two, Ch. 23.2.1):
    psi(t+dt) = exp(-i H_SM dt/2) exp(-i (Q + I + Phi_c) dt) exp(-i H_SM dt/2) psi(t)
    """
    psi = psi0.astype(np.complex128).copy()
    norm0 = np.sum(np.abs(psi) ** 2) * dx
    half_kinetic_phase = np.exp(-0.5j * (0.5 * (k ** 2)) * dt)

    history_C = []
    history_norm = []

    for _ in range(steps):
        # Half-step spectral kinetic evolution
        psi = np.fft.ifft(np.fft.fft(psi) * half_kinetic_phase)

        # Full-step nonlinear + gravitational evolution
        rho = np.abs(psi) ** 2
        phi_c = classical_gravity_potential(rho, k, G_eff=G_eff)
        Q_term = compression_operator(psi, dx, kappa=kappa, alpha=alpha)
        I_term = information_operator(psi, dx, gamma_I=gamma_I)

        # Local potential rotation + explicit nonlinear update
        psi = psi * np.exp(-1j * phi_c * dt) - 1j * dt * (Q_term + I_term)

        # Preserve probability normalization (Ch. 4.5 & 23.5)
        current_norm = np.sum(np.abs(psi) ** 2) * dx
        if current_norm > 0:
            psi *= np.sqrt(norm0 / current_norm)

        # Second half-step spectral kinetic evolution
        psi = np.fft.ifft(np.fft.fft(psi) * half_kinetic_phase)

        # Log Compression Index C(t) = integral |grad(rho)| dx (Ch. 14.7)
        rho_now = np.abs(psi) ** 2
        grad_rho = np.abs(np.roll(rho_now, -1) - np.roll(rho_now, 1)) / (2.0 * dx)
        history_C.append(float(np.sum(grad_rho) * dx))
        history_norm.append(float(np.sum(rho_now) * dx))

    return psi, np.array(history_C), np.array(history_norm)


# ==============================================================================
# 2. MODULE A: 2026 MNRAS COSMIC FILAMENT INTERFERENCE BENCHMARK (Ch. 22 & 25)
# ==============================================================================

def analyze_mnras_filament(x: np.ndarray, dx: float, rho: np.ndarray):
    """
    Fits a smooth Gaussian spine profile to rho(x) and extracts radial periodic fringe
    contrast A and wavelength lambda_0 to compare against O'Kane et al. (MNRAS, 2026):
    3-sigma exclusion bound: A <= 0.16 * lambda_0 + 0.18 for 0.2 <= lambda_0 <= 2.0 Mpc.
    """
    def gaussian_spine(x_val, amp, sigma, bg):
        return amp * np.exp(-(x_val ** 2) / (2.0 * sigma ** 2)) + bg

    try:
        p0 = [np.max(rho) - np.min(rho), 0.7, np.min(rho)]
        popt, _ = curve_fit(gaussian_spine, x, rho, p0=p0, maxfev=5000)
        smooth_profile = gaussian_spine(x, *popt)
    except Exception:
        smooth_profile = ndimage.gaussian_filter1d(rho, sigma=12)

    residual = (rho - smooth_profile) / (np.max(smooth_profile) + 1e-12)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=dx)
    fft_mag = 2.0 * np.abs(np.fft.rfft(residual)) / n

    # Search wavelengths inside the MNRAS 0.2 - 2.0 Mpc window
    valid_mask = (freqs > 0) & ((1.0 / np.maximum(freqs, 1e-9)) >= 0.2) & ((1.0 / np.maximum(freqs, 1e-9)) <= 2.0)
    if np.any(valid_mask):
        idx_peak = np.argmax(fft_mag[valid_mask])
        peak_freq = freqs[valid_mask][idx_peak]
        lambda_0 = float(1.0 / peak_freq)
        fringe_A = float(fft_mag[valid_mask][idx_peak])
    else:
        lambda_0, fringe_A = 0.5, 0.0

    mnras_limit = 0.16 * lambda_0 + 0.18
    return smooth_profile, residual, lambda_0, fringe_A, mnras_limit


# ==============================================================================
# 3. MODULE B: BOOK ONE CF-A..CF-D GRAPH LAPLACIAN & THEOREM 2 (Ch. 4, 6, 22)
# ==============================================================================

def build_cf_topology_adjacency(topology: str, n_nodes: int = 81, jitter: float = 0.0, seed: int = 42):
    """Constructs weighted adjacency matrix K for CF-A through CF-D routing classes."""
    rng = np.random.default_rng(seed)
    K = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    if topology == "CF-A (Linear Cascade)":
        for i in range(n_nodes - 1):
            w = 1.0 + (rng.uniform(-jitter, jitter) if jitter > 0 else 0.0)
            K[i, i + 1] = K[i + 1, i] = max(w, 0.05)

    elif topology == "CF-B (Fold-Resonant)":
        for i in range(n_nodes - 1):
            K[i, i + 1] = K[i + 1, i] = 1.0
        fold_step = max(3, n_nodes // 9)
        for i in range(0, n_nodes - fold_step, fold_step):
            w = 1.414 + (rng.uniform(-jitter, jitter) if jitter > 0 else 0.0)
            K[i, i + fold_step] = K[i + fold_step, i] = max(w, 0.05)

    elif topology == "CF-C (Dual-Nested Loop)":
        for i in range(n_nodes):
            K[i, (i + 1) % n_nodes] = K[(i + 1) % n_nodes, i] = 1.0
            K[i, (i + 3) % n_nodes] = K[(i + 3) % n_nodes, i] = np.sqrt(3.0)

    elif topology == "CF-D (Fractal Tri-Harmonic 2:3:5)":
        # Recursive 3-scale tri-harmonic coupling with 2:3:5 ratio weights
        ratios = [(1, 2.0), (3, 3.0), (9, 5.0), (27, 2.5)]
        for stride, weight in ratios:
            if stride < n_nodes:
                for i in range(n_nodes):
                    j = (i + stride) % n_nodes
                    w = weight * (1.0 + (rng.uniform(-jitter, jitter) if jitter > 0 else 0.0))
                    K[i, j] += max(w, 0.05)
                    K[j, i] += max(w, 0.05)

    return K


def run_kuramoto_laplacian_sweep(K: np.ndarray, etas: np.ndarray, seed: int = 42):
    """Tests Theorem 2 (eta_c ~ Delta_omega / lambda_2) on H_coh = -eta * L."""
    rng = np.random.default_rng(seed)
    n = K.shape[0]
    D = np.diag(np.sum(K, axis=1))
    L = D - K
    eigvals = np.linalg.eigvalsh(L)
    lambda_2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0

    omega_nat = rng.normal(0.0, 0.5, size=n)
    delta_omega = float(np.max(omega_nat) - np.min(omega_nat))
    eta_bound = delta_omega / max(lambda_2, 1e-9)

    order_params = []
    dt = 0.03
    for eta in etas:
        theta = rng.uniform(0, 2 * np.pi, size=n)
        for _ in range(300):
            # Graph-coupled phase dynamics driven by H_coh = -eta * L
            coupling = np.sum(K * np.sin(theta[None, :] - theta[:, None]), axis=1) / max(np.mean(np.sum(K, axis=1)), 1.0)
            theta += dt * (omega_nat + eta * coupling)
        R = float(np.abs(np.mean(np.exp(1j * theta))))
        order_params.append(R)

    return lambda_2, eta_bound, np.array(order_params)


# ==============================================================================
# 4. MODULE C: COMPLEX PHASE-COUPLED CF-D vs. RIEMANN ZETA GUE SPECTRUM
# ==============================================================================

def unfold_eigenvalues(eigvals: np.ndarray, poly_deg: int = 5) -> np.ndarray:
    """Polynomial spectral unfolding to unit mean level spacing (Montgomery-Dyson GUE test)."""
    ev = np.sort(np.real(eigvals))
    # Trim extreme spectral edges (10% each side)
    trim = max(5, int(0.10 * len(ev)))
    ev = ev[trim:-trim]
    staircase = np.arange(1, len(ev) + 1, dtype=np.float64)
    coeffs = np.polyfit(ev, staircase, deg=poly_deg)
    unfolded = np.polyval(coeffs, ev)
    spacings = np.diff(unfolded)
    spacings = spacings[spacings > 0]
    return spacings / np.mean(spacings)


def compute_cfd_magnetic_laplacian_spacings(n_nodes: int = 243, phase_flux: float = 1.0, jitter: float = 0.08, seed: int = 42):
    """
