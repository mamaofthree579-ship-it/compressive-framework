import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import io, imageio.v2 as imageio

# --- Page Setup ---
st.set_page_config(page_title="Quantum Double-Slit Particle Simulator", layout="wide")
st.title("🧠 Quantum Double-Slit Simulator: Observer Collapse & Particle Registration")

st.markdown("""
This interactive simulator visualizes the **quantum double-slit experiment** with optional observers.  
When observation occurs, wave interference **collapses** into localized **particle detections**.
Now featuring:
- 🌀 Dynamic 3D field view of |ψ|²  
- 💡 Live particle registration counter  
- 👁️ Variable observer effects (detector, instrument, or human)
""")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Simulation Controls")

observer_present = st.sidebar.checkbox("Enable Observation", value=True)
observer_type = st.sidebar.selectbox("Observer Type", ["detector", "instrument", "human"])
observer_strength = st.sidebar.slider("Observer Influence", 0.0, 2.0, 1.0, 0.1)
frames = st.sidebar.slider("Number of Frames", 10, 60, 25)
enable_3d = st.sidebar.checkbox("Show 3D Field Visualization", value=True)
particle_sensitivity = st.sidebar.slider("Particle Detection Sensitivity", 0.5, 2.0, 1.0, 0.1)

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
    slit1 = np.exp(-(Y - slit_distance)**2)
    slit2 = np.exp(-(Y + slit_distance)**2)
    psi = np.exp(1j * (X**2 + Y**2)) * (slit1 + slit2)
    return psi

psi_base = double_slit_wave(X, Y)

# --- Observer Field ---
def observer_field(X, Y, observer_type, strength, t, params=None):
    if params is None:
        params = {}
    if observer_type == "detector":
        detector_x, detector_k = 0, 6.0
        return strength * np.exp(-((X - detector_x)**2 + Y**2)/params.get("sigma",1.0)**2) * np.sin(detector_k * X + t)
    elif observer_type == "instrument":
        inst_freq = 0.9
        return strength * np.cos(inst_freq * X + t) * np.exp(-0.2*(X**2 + Y**2))
    elif observer_type == "human":
        noise_amp = params.get("noise_amp", 0.2)
        bias_x = params.get("bias_x", 0.0)
        rand_wave = np.sin(2*X + 3*Y + t) + np.cos(1.5*X - 2*Y + t/2)
        bias_field = np.exp(-((X - bias_x)**2 + Y**2))
        noise = noise_amp * np.random.randn(*X.shape)
        return strength * (rand_wave * bias_field + noise)
    return np.zeros_like(X)

# --- Normalization ---
def normalize(arr):
    arr_min, arr_max = np.min(arr), np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min + 1e-9)

# --- Particle Detection Model ---
def detect_particles(energy_field, sensitivity=1.0):
    threshold = sensitivity * np.mean(energy_field) * 1.5
    detections = np.argwhere(energy_field > threshold)
    count = len(detections)
    return detections, count

# --- Main Simulation ---
plot_placeholder = st.empty()
particle_placeholder = st.empty()

def render_frame(frame_index, cumulative_particles):
    tphase = 2 * np.pi * (frame_index / max(1, frames))
    psi_t = psi_base * np.cos(0.5 * tphase)
    obs = observer_field(X, Y, observer_type if observer_present else "none",
                         observer_strength, tphase,
                         params={"sigma": detector_sigma, "noise_amp": human_noise_amp, "bias_x": human_bias_x})
    psi_total = psi_t + obs + global_noise * np.random.randn(*psi_t.shape)
    energy = normalize(np.abs(psi_total)**2)
    detections, count = detect_particles(energy, particle_sensitivity)
    cumulative_particles += count

    # --- Visualization ---
    if enable_3d:
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, energy, cmap='inferno', linewidth=0, antialiased=False)
        ax.set_title(f"3D Field View | Frame {frame_index+1}/{frames}")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("|ψ|²")
        fig.colorbar(surf, ax=ax, shrink=0.5)
    else:
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(energy, origin="lower", cmap="inferno", aspect="auto",
                       extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]])
        ax.scatter(X[detections[:,0], detections[:,1]],
                   Y[detections[:,0], detections[:,1]],
                   s=5, c="cyan", alpha=0.7, label="Particle hits")
        ax.set_title(f"Frame {frame_index+1}/{frames} — {observer_type if observer_present else 'No Observer'}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, label="Intensity |ψ|²")
        ax.legend(loc="upper right", fontsize="small")

    plot_placeholder.pyplot(fig)
    plt.close(fig)

    return cumulative_particles, count

# --- Controls ---
frame_to_show = st.slider("Frame to View", 0, frames - 1, 0)
particle_count_total, last_frame_hits = render_frame(frame_to_show, cumulative_particles=0)

particle_placeholder.metric(
    label="💡 Particle Detections (This Frame / Cumulative)",
    value=f"{last_frame_hits} / {particle_count_total}"
)

# --- GIF Export ---
if st.button("🎞️ Export Animation"):
    st.info("Rendering animation — please wait...")
    frames_img = []
    cumulative_particles = 0
    for fi in range(frames):
        tphase = 2 * np.pi * fi / frames
        psi_t = psi_base * np.cos(0.5 * tphase)
        obs = observer_field(X, Y, observer_type if observer_present else "none",
                             observer_strength, tphase,
                             params={"sigma": detector_sigma, "noise_amp": human_noise_amp, "bias_x": human_bias_x})
        psi_total = psi_t + obs
        energy = normalize(np.abs(psi_total)**2)
        detections, count = detect_particles(energy, particle_sensitivity)
        cumulative_particles += count
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(energy, origin='lower', cmap='inferno', aspect='auto')
        ax.scatter(X[detections[:,0], detections[:,1]],
                   Y[detections[:,0], detections[:,1]],
                   s=5, c='cyan', alpha=0.6)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frames_img.append(imageio.imread(buf))
    gif_path = "/tmp/double_slit_observer_particles.gif"
    imageio.mimsave(gif_path, frames_img, fps=15)
    with open(gif_path, "rb") as f:
        st.download_button("📥 Download Animation with Particle Hits (GIF)", f, file_name="double_slit_observer_particles.gif", mime="image/gif")

st.success("✅ Simulation ready. Observe wave collapse → particle detection dynamics.")
