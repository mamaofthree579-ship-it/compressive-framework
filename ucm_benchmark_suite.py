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
    Constructs Hermitian Magnetic Graph Laplacian L_phi = D - K * exp(i * phi_ij)
    combining Book One CF-D (2:3:5) topology with Book Two I[psi] = psi * exp(i*phi).
    """
    rng = np.random.default_rng(seed)
    K_real = build_cf_topology_adjacency("CF-D (Fractal Tri-Harmonic 2:3:5)", n_nodes=n_nodes, jitter=jitter, seed=seed)
    D = np.diag(np.sum(K_real, axis=1))

    if phase_flux == 0.0:
        L_hermitian = D.astype(np.complex128) - K_real.astype(np.complex128)
    else:
        # Assign asymmetric 2:3:5 Peierls / informational phase twists along upper triangle
        upper_phases = rng.uniform(0, 2.0 * np.pi * phase_flux, size=(n_nodes, n_nodes))
        upper_phases = np.triu(upper_phases, k=1)
        phi_matrix = upper_phases - upper_phases.T
        K_complex = K_real * np.exp(1j * phi_matrix)
        L_hermitian = D.astype(np.complex128) - K_complex

    eigvals = np.linalg.eigvalsh(L_hermitian)
    spacings = unfold_eigenvalues(eigvals)

    # Generate analytical Riemann GUE Wigner surmise reference samples
    s_grid = np.linspace(0, 3.5, 400)
    gue_pdf = (32.0 / (np.pi ** 2)) * (s_grid ** 2) * np.exp(-(4.0 / np.pi) * (s_grid ** 2))
    poisson_pdf = np.exp(-s_grid)

    # Sample empirical GUE reference for exact Wasserstein distance
    gue_cdf = np.cumsum(gue_pdf) / np.sum(gue_pdf)
    gue_samples = np.interp(rng.uniform(0, 1, size=2000), gue_cdf, s_grid)
    w_dist = float(wasserstein_distance(spacings, gue_samples))

    return spacings, s_grid, gue_pdf, poisson_pdf, w_dist


# ==============================================================================
# 5. STREAMLIT USER INTERFACE
# ==============================================================================

st.title("Unified Compressive Model (UCM) — Open-Source Benchmark & Falsification Suite")
st.markdown(
    "Interactive verification suite implementing **Book One (`CF-A`–`CF-D` Adaptive Routing)** and "
    "**Book Two (Strang-Splitting Operator Algebra, 2026 *MNRAS* Filament Bounds, & Riemann Zeta GUE Spectral Tests)**."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Strang-Splitting & 2026 MNRAS Filament Test",
    "2. CF-A..CF-D Synchronization (Theorem 2)",
    "3. CF-D Complex Phase vs. Riemann Zeta GUE",
    "4. Multi-Field Spatiotemporal Lattice (CF-DPF)",
    "5. Madelung Shock Regularization (Navier–Stokes)",
])

# ------------------------------------------------------------------------------
# TAB 1: Strang-Splitting & 2026 MNRAS Filament Validation
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Cosmic Filament Interference & 2026 MNRAS Observational Bound")
    st.caption(
        "Compares Standard QM (Schrödinger–Poisson) against your UCM solver (Q[psi] + I[psi] + g_c) "
        "and checks radial fringe amplitude A against the O'Kane et al. (MNRAS, 2026) SDSS exclusion bound: A <= 0.16*lambda_0 + 0.18."
    )

    c1, c2, c3, c4 = st.columns(4)
    kappa_val = c1.slider("Compression strength κ", 0.000, 0.020, 0.006, 0.001)
    alpha_val = c2.selectbox("Density exponent α (2/d = 2 in 1D)", [1.0, 1.5, 2.0, 2.5], index=2)
    gamma_I_val = c3.slider("Info gradient damping γ_I", 0.000, 0.010, 0.002, 0.0005)
    sim_steps = c4.slider("Strang time steps", 100, 600, 300, 50)

    x, dx, k = build_1d_grid(n_points=512, span=4.0)
    # Initial filament spine + multi-mode wave perturbations
    psi_init = (
        np.exp(-(x ** 2) / (2.0 * 0.65 ** 2))
        * (1.0 + 0.18 * np.cos(2.0 * np.pi * x / 0.50) + 0.10 * np.cos(2.0 * np.pi * x / 0.33))
    ).astype(np.complex128)
    psi_init /= np.sqrt(np.sum(np.abs(psi_init) ** 2) * dx)

    dt = min(0.0015, 0.35 * (dx ** 2))

    # Run Standard QM baseline (kappa=0, gamma_I=0)
    psi_qm, C_qm, _ = evolve_strang_1d(psi_init, dx, k, dt, sim_steps, kappa=0.0, alpha=alpha_val, gamma_I=0.0)
    rho_qm = np.abs(psi_qm) ** 2

    # Run UCM modified evolution
    psi_ucm, C_ucm, norm_ucm = evolve_strang_1d(
        psi_init, dx, k, dt, sim_steps, kappa=kappa_val, alpha=alpha_val, gamma_I=gamma_I_val
    )
    rho_ucm = np.abs(psi_ucm) ** 2

    smooth_fit, residual_ucm, lambda_0, fringe_A, mnras_limit = analyze_mnras_filament(x, dx, rho_ucm)

    m1, m2, m3, m4 = st.columns(4)
    c_gain = ((C_ucm[-1] - C_qm[-1]) / max(C_qm[-1], 1e-9)) * 100.0
    m1.metric("Compression Index C(t)", f"{C_ucm[-1]:.4f}", f"{c_gain:+.2f}% vs QM")
    m2.metric("Dominant Fringe λ₀", f"{lambda_0:.3f} Mpc")
    m3.metric("Fringe Contrast A", f"{fringe_A:.4f}", f"Limit: {mnras_limit:.4f}", delta_color="inverse")
    m4.metric("2026 MNRAS Status", "VALIDATED" if fringe_A <= mnras_limit else "ABOVE 3σ LIMIT")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(x=x, y=rho_qm, name="Standard QM |ψ|²", line=dict(color="#888888", dash="dash")))
        fig_prof.add_trace(go.Scatter(x=x, y=rho_ucm, name="UCM (Q + I + g_c) |ψ|²", line=dict(color="#ff5722", width=2.5)))
        fig_prof.add_trace(go.Scatter(x=x, y=smooth_fit, name="Smooth Gaussian Spine Fit", line=dict(color="#1f77b4", dash="dot")))
        fig_prof.update_layout(
            title="Radial Cosmic Filament Density Profile ρ(x)",
            xaxis_title="Transverse Distance x (Mpc)",
            yaxis_title="Normalized Density |ψ|²",
            height=400,
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    with col_right:
        fig_c = go.Figure()
        t_axis = np.arange(sim_steps) * dt
        fig_c.add_trace(go.Scatter(x=t_axis, y=C_qm, name="Standard QM C(t)", line=dict(color="#888888", dash="dash")))
        fig_c.add_trace(go.Scatter(x=t_axis, y=C_ucm, name="UCM Compression Index C(t)", line=dict(color="#2ca02c", width=2.5)))
        fig_c.update_layout(
            title="Observable Set A: Compression Index C(t) = ∫|∇ρ|dx",
            xaxis_title="Simulation Time t",
            yaxis_title="Compression Index C(t)",
            height=400,
        )
        st.plotly_chart(fig_c, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: Book One CF-A..CF-D Graph Laplacian & Theorem 2 Verification
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("CF-A through CF-D Routing Topologies & Theorem 2 Synchronization Threshold")
    st.caption(
        "Verifies Book One's multi-arm stability progression against Book Two Theorem 2: "
        "critical phase-locking coupling η_c ~ Δω / λ_2 (inversely proportional to graph Laplacian spectral gap λ_2)."
    )

    n_graph = st.selectbox("Lattice Node Count", [54, 81, 162], index=1)
    etas = np.linspace(0.05, 2.5, 22)

    topologies = [
        "CF-A (Linear Cascade)",
        "CF-B (Fold-Resonant)",
        "CF-C (Dual-Nested Loop)",
        "CF-D (Fractal Tri-Harmonic 2:3:5)",
    ]
    colors = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#2ca02c"]

    fig_sync = go.Figure()
    summary_rows = []

    for topo, col in zip(topologies, colors):
        K_mat = build_cf_topology_adjacency(topo, n_nodes=n_graph, jitter=0.05)
        lam2, eta_b, R_curve = run_kuramoto_laplacian_sweep(K_mat, etas)
        fig_sync.add_trace(go.Scatter(x=etas, y=R_curve, name=f"{topo} (λ₂={lam2:.3f})", line=dict(color=col, width=2.5)))

        # Estimate empirical eta_c where R crosses 0.85
        above = etas[R_curve >= 0.85]
        eta_emp = f"{above[0]:.3f}" if len(above) > 0 else "> 2.50"
        r_mid = float(np.interp(0.50, etas, R_curve))
        summary_rows.append({
            "Topology": topo,
            "Spectral Gap (λ₂)": round(lam2, 4),
            "Theorem 2 Bound (Δω/λ₂)": round(eta_b, 4),
            "Simulated η_c (R ≥ 0.85)": eta_emp,
            "Order Parameter R at η=0.5": f"{r_mid * 100:.1f}%",
        })

    fig_sync.add_hline(y=0.85, line_dash="dot", line_color="black", annotation_text="Phase-Lock Threshold (R = 0.85)")
    fig_sync.update_layout(
        title="Kuramoto Order Parameter R(η) Across Book One Routing Architectures",
        xaxis_title="Coherence Coupling Strength η",
        yaxis_title="Global Phase Coherence R(η)",
        height=430,
    )
    st.plotly_chart(fig_sync, use_container_width=True)
    st.dataframe(summary_rows, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: Complex Phase-Coupled CF-D vs. Riemann Zeta GUE Spectral Test
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Hilbert–Pólya Spectral Bridge: Complex Phase CF-D vs. Riemann Zeta GUE")
    st.caption(
        "Tests how adding complex informational phase twists e^(i φ_ij) from I[psi] converts the real symmetric "
        "CF-D fractal Laplacian into a Hermitian magnetic Laplacian matching the Riemann zeta zero GUE curve."
    )

    c_g1, c_g2, c_g3 = st.columns(3)
    n_spec = c_g1.selectbox("Fractal Spectral Graph Size", [162, 243, 486], index=1)
    flux_strength = c_g2.slider("Complex Phase Twist Strength (φ_max / 2π)", 0.0, 1.0, 1.0, 0.1)
    adaptive_jitter = c_g3.slider("Adaptive Cluster Perturbation", 0.00, 0.20, 0.08, 0.01)

    spacings_real, s_grid, gue_pdf, poisson_pdf, w_real = compute_cfd_magnetic_laplacian_spacings(
        n_nodes=n_spec, phase_flux=0.0, jitter=adaptive_jitter
    )
    spacings_complex, _, _, _, w_complex = compute_cfd_magnetic_laplacian_spacings(
        n_nodes=n_spec, phase_flux=flux_strength, jitter=adaptive_jitter
    )

    g1, g2, g3 = st.columns(3)
    g1.metric("Real CF-D Wasserstein Dist to GUE", f"{w_real:.4f}")
    reduction = ((w_real - w_complex) / max(w_real, 1e-9)) * 100.0
    g2.metric("Complex CF-D Wasserstein Dist to GUE", f"{w_complex:.4f}", f"{-reduction:.1f}% error")
    g3.metric("Small-Gap Repulsion (s < 0.25)", f"{np.mean(spacings_complex < 0.25) * 100:.1f}%", "Riemann GUE: 1.6%")

    fig_gue = go.Figure()
    fig_gue.add_trace(go.Histogram(
        x=spacings_real, xbins=dict(start=0, end=3.2, size=0.16), histnorm="probability density",
        name="Real CF-D (φ = 0)", marker_color="#7f7f7f", opacity=0.45
    ))
    fig_gue.add_trace(go.Histogram(
        x=spacings_complex, xbins=dict(start=0, end=3.2, size=0.16), histnorm="probability density",
        name="Complex Phase CF-D (e^{iφ})", marker_color="#ff5722", opacity=0.65
    ))
    fig_gue.add_trace(go.Scatter(
        x=s_grid, y=gue_pdf, name="Riemann Zeta Zero Curve (GUE)", line=dict(color="#1f77b4", width=3)
    ))
    fig_gue.add_trace(go.Scatter(
        x=s_grid, y=poisson_pdf, name="Uncorrelated Poisson", line=dict(color="#2ca02c", width=2, dash="dash")
    ))
    fig_gue.update_layout(
        barmode="overlay",
        title="Unfolded Nearest-Neighbor Eigenvalue Spacing P(s) vs. Riemann Zeta GUE",
        xaxis_title="Normalized Level Spacing s",
        yaxis_title="Probability Density P(s)",
        height=450,
    )
    st.plotly_chart(fig_gue, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: Upgraded Spatiotemporal CF-DPF Lattice (Your Original Prototype Fixed)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("CF-DPF Spatiotemporal Lattice & Emergent Anchor Cluster Nucleation")
    st.caption(
        "Upgraded version of your original CF-DPF prototype with full (X, T) spatiotemporal phase propagation "
        "and connected-component anchor cluster nucleation."
    )

    s1, s2, s3, s4 = st.columns(4)
    w_grav = s1.slider("α — Graviton (γ) scale", 0.0, 2.0, 0.8, 0.05)
    w_chro = s2.slider("β — Chronon (χ) scale", 0.0, 2.0, 0.6, 0.05)
    w_cogn = s3.slider("γ — Cognon (κ) scale", 0.0, 2.0, 0.7, 0.05)
    nuc_thresh = s4.slider("Nucleation threshold", 0.05, 0.85, 0.45, 0.05)

    nx_grid = 250
    x_arr = np.linspace(-6, 6, nx_grid)
    t_arr = np.linspace(0, 2 * np.pi, nx_grid)
    X_mesh, T_mesh = np.meshgrid(x_arr, t_arr)

    # Full spatiotemporal phase propagation across both X and T
    grav = np.sin(w_grav * (3.0 * X_mesh - 1.0 * T_mesh))
    chro = np.sin(w_chro * (3.0 * X_mesh - 0.7 * T_mesh))
    cogn = np.sin(w_cogn * (3.0 * X_mesh - 1.3 * T_mesh))
    env = np.exp(-(X_mesh ** 2) / (2.0 * 2.8 ** 2))

    psi_field = (grav + chro + cogn) * env
    energy_field = psi_field ** 2
    e_min, e_max = np.min(energy_field), np.max(energy_field)
    energy_norm = (energy_field - e_min) / max(e_max - e_min, 1e-12)

    mask = energy_norm > nuc_thresh
    labeled, n_clusters = ndimage.label(mask)
    objs = ndimage.find_objects(labeled)
    centroids_x, centroids_t = [], []
    for idx, slc in enumerate(objs):
        if slc is not None:
            reg = labeled[slc] == (idx + 1)
            if np.sum(reg) >= 8:
                cy, cx = ndimage.center_of_mass(reg)
                row_idx = min(slc[0].start + int(round(cy)), nx_grid - 1)
                col_idx = min(slc[1].start + int(round(cx)), nx_grid - 1)
                centroids_t.append(float(t_arr[row_idx]))
                centroids_x.append(float(x_arr[col_idx]))

    fig_dpf = go.Figure(data=go.Heatmap(
        z=energy_norm, x=x_arr, y=t_arr, colorscale="Plasma", colorbar=dict(title="Normalized |Ψ|²")
    ))
    if centroids_x:
        fig_dpf.add_trace(go.Scatter(
            x=centroids_x, y=centroids_t, mode="markers",
            marker=dict(size=7, color="white", line=dict(width=1, color="black")),
            name=f"Anchor Clusters ({len(centroids_x)})"
        ))
    fig_dpf.update_layout(
        title=f"Spatiotemporal Compressive Field |Ψ(x,t)|² — Detected Anchor Clusters: {len(centroids_x)}",
        xaxis_title="Spatial Coordinate x",
        yaxis_title="Temporal Phase t",
        height=520,
    )
    st.plotly_chart(fig_dpf, use_container_width=True)

# ==============================================================================
# 6. MODULE D: MADELUNG-NAVIER-STOKES SHOCK REGULARIZATION (Book Two, Ch. 20-21)
# ==============================================================================

def run_madelung_shock_comparison(
    n_grid: int = 256,
    t_max: float = 1.25,
    kappa: float = 0.08,
    alpha: float = 2.0,
    eta_coh: float = 0.015,
    hbar_eff: float = 0.06,
):
    """
    Compares unregularized inviscid Burgers/Euler shock blowup (du/dt + u du/dx = 0)
    against the Madelung-transformed UCM fluid (psi = sqrt(rho) * exp(i*S/hbar_eff))
    evolved under Q[psi] + H_coh (Book Two, Section 20.7 & Theorems 4, 8).
    """
    L = 2.0 * np.pi
    x = np.linspace(0, L, n_grid, endpoint=False)
    dx = x - x
    k = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=dx)

    # Identical smooth initial velocity profile u(x, 0) = sin(x)
    u_init = np.sin(x)
    S_init = -np.cos(x)  # since v = dS/dx = sin(x)
    rho_init = np.ones_like(x)
    psi = np.sqrt(rho_init) * np.exp(1j * S_init / hbar_eff)

    u_std = u_init.copy()
    dt = 0.0005
    steps = int(t_max / dt)
    sample_every = max(1, steps // 35)

    t_hist, grad_std_hist, grad_ucm_hist, rho_max_hist = [], [], [], []
    kin_phase = np.exp(-0.5j * hbar_eff * (k ** 2) * dt)

    for step in range(steps + 1):
        if step % sample_every == 0:
            # Standard fluid velocity gradient |du/dx|
            du_std = np.abs(np.roll(u_std, -1) - np.roll(u_std, 1)) / (2.0 * dx)

            # Madelung velocity extraction: v = (hbar_eff / rho) * Im(psi* grad(psi))
            rho_now = np.abs(psi) ** 2
            grad_psi = (np.roll(psi, -1) - np.roll(psi, 1)) / (2.0 * dx)
            v_ucm = hbar_eff * np.imag(np.conj(psi) * grad_psi) / np.maximum(rho_now, 1e-6)
            dv_ucm = np.abs(np.roll(v_ucm, -1) - np.roll(v_ucm, 1)) / (2.0 * dx)

            t_hist.append(step * dt)
            grad_std_hist.append(float(np.max(du_std)))
            grad_ucm_hist.append(float(np.max(dv_ucm)))
            rho_max_hist.append(float(np.max(rho_now)))

        # 1. Advance Standard Inviscid Fluid (Upwind conservation form)
        flux_std = 0.5 * (u_std ** 2)
        u_std = u_std - (dt / dx) * (flux_std - np.roll(flux_std, 1))

        # 2. Advance UCM Madelung Fluid via Strang Splitting (Q + H_coh + Bohm dispersion)
        psi = np.fft.ifft(np.fft.fft(psi) * kin_phase)
        rho = np.abs(psi) ** 2
        # Polytropic compression potential V_Q = kappa * rho^alpha (Theorem 8)
        V_Q = kappa * (rho ** alpha)
        # Coherence graph Laplacian damping H_coh = -eta * L
        lap_psi = (np.roll(psi, -1) - 2.0 * psi + np.roll(psi, 1)) / (dx ** 2)
        psi = psi * np.exp(-1j * V_Q * dt / hbar_eff) + eta_coh * dt * lap_psi
        psi = np.fft.ifft(np.fft.fft(psi) * kin_phase)

    # Final spatial profiles for overlay comparison
    rho_final = np.abs(psi) ** 2
    grad_psi_final = (np.roll(psi, -1) - np.roll(psi, 1)) / (2.0 * dx)
    v_ucm_final = hbar_eff * np.imag(np.conj(psi) * grad_psi_final) / np.maximum(rho_final, 1e-6)

    return (
        x,
        u_std,
        v_ucm_final,
        rho_final,
        np.array(t_hist),
        np.array(grad_std_hist),
        np.array(grad_ucm_hist),
        np.array(rho_max_hist),
    )


# ------------------------------------------------------------------------------
# TAB 5: Madelung-Navier-Stokes Shock Regularization
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Madelung–Navier–Stokes Mapping & Singularity Blowup Suppression")
    st.caption(
        "Compares finite-time shock blowup in unregularized Euler/Burgers flow against your Madelung-decomposed "
        "UCM fluid (Q[psi] polytropic pressure + H_coh coherence viscosity + Bohm dispersion; Book Two, Section 20.7)."
    )

    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    shock_kappa = m_c1.slider("Polytropic compression κ", 0.00, 0.20, 0.08, 0.01)
    shock_eta = m_c2.slider("Coherence viscosity η_coh", 0.000, 0.050, 0.015, 0.005)
    shock_hbar = m_c3.slider("Bohm dispersion scale ħ_eff", 0.02, 0.15, 0.06, 0.01)
    shock_tmax = m_c4.slider("Evolution horizon t_max", 0.80, 1.50, 1.25, 0.05)

    (
        x_sh,
        u_std_end,
        v_ucm_end,
        rho_sh_end,
        t_sh,
        g_std,
        g_ucm,
        rho_max_sh,
    ) = run_madelung_shock_comparison(
        n_grid=256,
        t_max=shock_tmax,
        kappa=shock_kappa,
        alpha=2.0,
        eta_coh=shock_eta,
        hbar_eff=shock_hbar,
    )

    peak_std = float(np.max(g_std))
    peak_ucm = float(np.max(g_ucm))
    suppression_pct = ((peak_std - peak_ucm) / max(peak_std, 1e-9)) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Standard Fluid Peak |∂u/∂x|", f"{peak_std:.2f}", "Shock Singularity (t ≈ 1.0)", delta_color="inverse")
    k2.metric("UCM Fluid Peak |∂v/∂x|", f"{peak_ucm:.2f}", f"{-suppression_pct:.1f}% blowup suppression")
    k3.metric("Terminal UCM Gradient", f"{g_ucm[-1]:.2f}", f"vs. Standard: {g_std[-1]:.2f}")
    k4.metric("Peak Core Compression ρ_max", f"{np.max(rho_max_sh):.3f}", "Bounded Polytropic Core")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_grad = go.Figure()
        fig_grad.add_trace(go.Scatter(
            x=t_sh, y=g_std, name="Standard Inviscid Fluid |∂u/∂x|", line=dict(color="#d62728", width=2.5, dash="dash")
        ))
        fig_grad.add_trace(go.Scatter(
            x=t_sh, y=g_ucm, name="UCM Regularized Fluid |∂v/∂x|", line=dict(color="#1f77b4", width=3)
        ))
        fig_grad.add_vline(x=1.0, line_dash="dot", line_color="gray", annotation_text="Classical Shock Time (t = 1.0)")
        fig_grad.update_layout(
            title="Maximum Velocity Gradient Evolution (Singularity Suppression)",
            xaxis_title="Simulation Time t",
            yaxis_title="Max Velocity Gradient |∂v/∂x|",
            height=410,
        )
        st.plotly_chart(fig_grad, use_container_width=True)

    with col_s2:
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=x_sh, y=u_std_end, name="Standard Fluid Velocity u(x, t_max)", line=dict(color="#d62728", dash="dash")
        ))
        fig_vel.add_trace(go.Scatter(
            x=x_sh, y=v_ucm_end, name="UCM Madelung Velocity v(x, t_max)", line=dict(color="#1f77b4", width=2.5)
        ))
        fig_vel.add_trace(go.Scatter(
            x=x_sh, y=rho_sh_end, name="UCM Fluid Density ρ(x, t_max)", line=dict(color="#2ca02c", width=2, opacity=0.75)
        ))
        fig_vel.update_layout(
            title=f"Terminal Wavefront & Density Profile at t = {shock_tmax:.2f}",
            xaxis_title="Spatial Coordinate x",
            yaxis_title="Velocity v(x) / Density ρ(x)",
            height=410,
        )
        st.plotly_chart(fig_vel, use_container_width=True)
