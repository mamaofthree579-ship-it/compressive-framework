import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io, imageio.v2 as imageio

# --- Page Setup ---
st.set_page_config(page_title="Quantum Double-Slit Observer Simulator", layout="wide")
st.title("🧠 Quantum Double-Slit Observer Simulator")
st.markdown("""
This interactive simulator demonstrates how observation—by detectors, instruments, or humans—
influences interference in the **double-slit experiment**, visualized through the Compressive Framework.
""")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Simulation Controls")

observer_present = st.sidebar.checkbox("Enable Observation", value=True)
observer_type = st.sidebar.selectbox("Observer Type", ["detector", "instrument", "human"])
observer_strength = st.sidebar.slider("Observer Influence", 0.0, 2.0, 1.0, 0.1)
frames = st.sidebar.slider("Number of Frames", 10, 60, 25)
compare_mode = st.sidebar.checkbox("Side-by-side comparison mode", value=False)

# Human-specific harmonics
human_noise_amp = st.sidebar.slider("Human Noise Amplitude", 0.0, 0.5, 0.2)
human_bias_x = st.sidebar.slider("Human Spatial Bias (x)", -0.5, 0.5, 0.0)
detector_sigma = st.sidebar.slider("Detector Sigma", 0.2, 2.0, 1.0)

# --- Grid Setup ---
x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_grid, y_grid)
global_noise = 0.05

# --- Base Double Slit Field ---
def double_slit_wave(X, Y):
    slit_distance = 1.0
    slit1 = np.exp(-(Y - slit_distance) ** 2)
    slit2 = np.exp(-(Y + slit_distance) ** 2)
    psi = np.exp(1j * (X ** 2 + Y ** 2)) * (slit1 + slit2)
    return psi

psi_base = double_slit_wave(X, Y)

# --- Observer Field ---
def observer_field(X, Y, observer_type, strength, t, params=None):
    if params is None:
        params = {}
    if observer_type == "detector":
        detector_x, detector_sigma, detector_k = params.get("detector_x", 0), params.get("detector_sigma", 1.0), params.get("detector_k", 6.0)
        return strength * np.exp(-((X - detector_x) ** 2 + Y ** 2) / detector_sigma ** 2) * np.sin(detector_k * X + t)
    elif observer_type == "instrument":
        inst_freq = params.get("inst_freq", 0.9)
        return strength * np.cos(inst_freq * X + t) * np.exp(-0.2 * (X ** 2 + Y ** 2))
    elif observer_type == "human":
        noise_amp = params.get("human_noise_amp", 0.2)
        bias_x = params.get("human_bias_x", 0.0)
        rand_wave = np.sin(2 * X + 3 * Y + t) + np.cos(1.5 * X - 2 * Y + t / 2)
        bias_field = np.exp(-((X - bias_x) ** 2 + Y ** 2))
        noise = noise_amp * np.random.randn(*X.shape)
        return strength * (rand_wave * bias_field + noise)
    return np.zeros_like(X)

# --- Utilities ---
def normalize(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min + 1e-9)

plot_placeholder = st.empty()

# --- Renderers ---
def render_frame_and_stats(frame_index):
    tphase = 2 * np.pi * (frame_index / max(1, frames))
    psi_t = psi_base * np.cos(0.5 * tphase)
    obs = observer_field(X, Y, observer_type if observer_present else "none", observer_strength, tphase,
                         params={"detector_sigma": detector_sigma, "human_noise_amp": human_noise_amp, "human_bias_x": human_bias_x})
    psi_total = psi_t + obs + global_noise * np.random.randn(*psi_t.shape)
    energy = normalize(np.abs(psi_total) ** 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(energy, origin="lower", cmap="inferno", aspect="auto",
                   extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
    ax.set_title(f"Frame {frame_index+1}/{frames} — Observer: {observer_type if observer_present else 'None'}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="Intensity |ψ|²")
    plot_placeholder.pyplot(fig)
    plt.close(fig)

def render_side_by_side(frame_index):
    tphase = 2 * np.pi * (frame_index / max(1, frames))
    psi_base_t = psi_base * np.cos(0.5 * tphase)

    # Unobserved
    psi_no_obs = psi_base_t + global_noise * np.random.randn(*psi_base.shape)
    energy_no = normalize(np.abs(psi_no_obs)**2)

    # Observed
    obs = observer_field(X, Y, observer_type if observer_present else "none", observer_strength, tphase,
                         params={"detector_sigma": detector_sigma, "human_noise_amp": human_noise_amp, "human_bias_x": human_bias_x})
    psi_obs = psi_base_t + obs + global_noise * np.random.randn(*psi_base.shape)
    energy_obs = normalize(np.abs(psi_obs)**2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(energy_no, origin='lower', cmap='inferno', aspect='auto')
    axes[0].set_title("Unobserved (Baseline)")
    axes[1].imshow(energy_obs, origin='lower', cmap='inferno', aspect='auto')
    axes[1].set_title(f"Observed — {observer_type}")
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.suptitle(f"Frame {frame_index+1}/{frames} — Comparative View")
    plot_placeholder.pyplot(fig)
    plt.close(fig)

# --- Simulation Controls ---
frame_to_show = st.slider("Frame to View", 0, frames - 1, 0)
if compare_mode:
    render_side_by_side(frame_to_show)
else:
    render_frame_and_stats(frame_to_show)

# --- GIF Export ---
if st.button("🎞️ Export Animation"):
    st.info("Rendering frames — please wait...")
    frames_img = []
    for fi in range(frames):
        tphase = 2 * np.pi * fi / frames
        psi_t = psi_base * np.cos(0.5 * tphase)
        obs = observer_field(X, Y, observer_type if observer_present else "none", observer_strength, tphase,
                             params={"detector_sigma": detector_sigma, "human_noise_amp": human_noise_amp, "human_bias_x": human_bias_x})
        psi_total = psi_t + obs
        energy = normalize(np.abs(psi_total)**2)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(energy, origin='lower', cmap='inferno', aspect='auto')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        frames_img.append(img)
    out_gif = "/tmp/double_slit_observer.gif"
    imageio.mimsave(out_gif, frames_img, fps=15)
    with open(out_gif, "rb") as f:
        st.download_button("📥 Download Animation (GIF)", f, file_name="double_slit_observer.gif", mime="image/gif")

st.success("✅ Simulation ready. Adjust parameters to explore how observation influences quantum interference.")
