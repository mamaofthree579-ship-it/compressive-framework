# shells_simulator.py
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
import time

st.set_page_config(page_title="Dimensional Shells Simulator", layout="wide")
st.title("Dimensional Shells — Layered Coupling Simulator")
st.markdown(
    "Model shells as coupled oscillator populations. Higher shells influence lower shells via resonance-filtered coupling; "
    "this demonstrates why layers inform each other without collapsing."
)

# ---------------------------
# Utilities & session init
# ---------------------------
def init_state():
    if "ready" not in st.session_state:
        st.session_state.ready = True
        st.session_state.shell_count = 5
        st.session_state.per_shell = 40
        st.session_state.K_intra = 1.0          # baseline intra-shell coupling
        st.session_state.C_inter = 0.25         # baseline inter-shell coupling
        st.session_state.resonance_width = 0.5  # controls resonance filter (sigma)
        st.session_state.damping = 0.05
        st.session_state.dt = 0.03
        st.session_state.time = 0.0
        st.session_state.run = False
        reset_sim(seed=1234)

def reset_sim(seed=None):
    rng = np.random.default_rng(seed if seed is not None else int(time.time()%1e9))
    S = st.session_state.shell_count
    N = st.session_state.per_shell
    # natural frequencies: higher shells biased to higher frequency band
    base = np.linspace(1.0, 6.0, S)  # center frequency per shell
    omegas = np.zeros((S, N))
    amps = np.zeros((S, N))
    phases = np.zeros((S, N))
    for s in range(S):
        omegas[s] = rng.normal(loc=base[s], scale=0.2, size=N)  # small spread
        amps[s] = rng.uniform(0.2, 1.0, size=N)
        phases[s] = rng.uniform(0, 2*np.pi, size=N)
    st.session_state.omegas = omegas
    st.session_state.amps = amps
    st.session_state.phases = phases
    # inter-shell coupling matrix (symmetric for now)
    C = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            if abs(i-j)==1:
                C[i,j] = st.session_state.C_inter
    st.session_state.C = C
    st.session_state.time = 0.0
    st.session_state.log = []

init_state()

# ---------------------------
# Simulation functions
# ---------------------------
def resonance_filter(dw, sigma):
    # Gaussian filter: returns weight in [0,1]
    return np.exp(-0.5 * (dw/sigma)**2)

def shell_mean_phase(s):
    vec = np.exp(1j * st.session_state.phases[s])
    return np.angle(vec.mean())

def compute_step():
    S, N = st.session_state.omegas.shape
    dt = st.session_state.dt
    sigma = st.session_state.resonance_width
    new_phases = st.session_state.phases.copy()
    new_amps = st.session_state.amps.copy()

    # precompute shell mean phases and mean frequencies
    shell_mean_phases = np.array([shell_mean_phase(s) for s in range(S)])
    shell_mean_freqs = np.array([st.session_state.omegas[s].mean() for s in range(S)])
    for s in range(S):
        # intra-shell coupling via Kuramoto
        phases_s = st.session_state.phases[s]
        omegas_s = st.session_state.omegas[s]
        # coupling term from shell's own oscillators
        # compute mean-field term R e^{i Phi}
        z = np.mean(np.exp(1j * phases_s))
        R = np.abs(z)
        Phi = np.angle(z)
        # use mean-field instead of O(N^2) pairwise for speed
        intra_term = st.session_state.K_intra * R * np.sin(Phi - phases_s)

        # inter-shell influence (sum over neighbor shells)
        inter_term = np.zeros_like(phases_s)
        amp_drive = np.zeros_like(phases_s)

        for s2 in range(S):
            if st.session_state.C[s,s2] == 0:
                continue
            # resonance weight based on mean freq difference
            dw = shell_mean_freqs[s2] - shell_mean_freqs[s]
            w = resonance_filter(dw, sigma)
            # drive from shell mean phase
            Phi2 = shell_mean_phases[s2]
            inter_term += st.session_state.C[s,s2] * w * np.sin(Phi2 - phases_s)
            # amplitude driving: if frequencies align, amplitude increases proportional to w
            amp_drive += w * (st.session_state.amps[s2].mean() - st.session_state.amps[s]) * 0.3

        # update phases
        dphi = omegas_s + intra_term + inter_term
        new_phases[s] = (phases_s + dphi * dt) % (2*np.pi)

        # amplitude dynamics: damping + driven growth from inter-shell drive and phase alignment
        gamma = st.session_state.damping
        # phase alignment factor (how well oscillator matches its shell mean)
        align = 0.5 * (1 + np.cos(phases_s - Phi))
        dA = -gamma * st.session_state.amps[s] + 0.5 * align + 0.2 * amp_drive
        new_amps[s] = np.clip(st.session_state.amps[s] + dA * dt, 0.0, 5.0)

    st.session_state.phases = new_phases
    st.session_state.amps = new_amps
    st.session_state.time += dt

def step_n(n=1):
    for _ in range(n):
        compute_step()
    # log summary
    S = st.session_state.omegas.shape[0]
    summary = {"t": float(st.session_state.time)}
    for s in range(S):
        summary[f"mean_amp_s{s}"] = float(st.session_state.amps[s].mean())
        summary[f"R_s{s}"] = float(np.abs(np.mean(np.exp(1j * st.session_state.phases[s]))))
    st.session_state.log.append(summary)
    return summary

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Simulator controls")
    st.session_state.shell_count = st.number_input("Number of shells", min_value=2, max_value=12, value=st.session_state.shell_count, step=1)
    st.session_state.per_shell = st.number_input("Oscillators per shell", min_value=6, max_value=200, value=st.session_state.per_shell, step=2)
    st.session_state.K_intra = st.slider("Intra-shell coupling K", 0.0, 4.0, float(st.session_state.K_intra), 0.01)
    st.session_state.C_inter = st.slider("Inter-shell coupling (nearest neighbor)", 0.0, 1.0, float(st.session_state.C_inter), 0.01)
    st.session_state.resonance_width = st.slider("Resonance width σ", 0.01, 2.0, float(st.session_state.resonance_width), 0.01)
    st.session_state.damping = st.slider("Amplitude damping γ", 0.0, 1.0, float(st.session_state.damping), 0.01)
    st.session_state.dt = st.number_input("Time step dt", value=float(st.session_state.dt), min_value=0.001, max_value=0.1, step=0.005, format="%.3f")
    st.markdown("---")
    colA, colB = st.columns(2)
    if colA.button("Reset / (new seed)"):
        reset_sim(seed=int(time.time())%1000000)
    if colB.button("Rebuild (apply shell count/per-shell)"):
        # rebuild with new dimensions keeping random seed
        reset_sim(seed=123)
    st.markdown("---")
    run_toggle = st.checkbox("Run continuously", value=st.session_state.run)
    st.session_state.run = run_toggle
    if st.button("Step 1"):
        summary = step_n(1)
        st.write("Step:", summary)
    if st.button("Step 10"):
        summary = step_n(10)
        st.write("Stepped 10: t =", st.session_state.time)
    st.markdown("---")
    st.subheader("Inter-shell coupling matrix")
    st.write("You can edit nearest-neighbor coupling constant; diagonal remains zero.")
    # show current C matrix and allow quick change
    if st.button("Zero inter-shell coupling"):
        st.session_state.C = np.zeros_like(st.session_state.C)
    if st.button("Reset standard nearest-neighbor coupling"):
        S = st.session_state.shell_count
        C = np.zeros((S, S))
        for i in range(S-1):
            C[i,i+1] = st.session_state.C_inter
            C[i+1,i] = st.session_state.C_inter
        st.session_state.C = C
    st.markdown("---")
    st.subheader("Export & logs")
    if st.button("Export last snapshot (JSON)"):
        snap = {
            "time": float(st.session_state.time),
            "shell_count": st.session_state.shell_count,
            "per_shell": st.session_state.per_shell,
            "amps": st.session_state.amps.tolist(),
            "phases": st.session_state.phases.tolist(),
            "omegas": st.session_state.omegas.tolist()
        }
        st.download_button("Download snapshot JSON", data=json.dumps(snap, indent=2), file_name=f"snapshot_t{st.session_state.time:.3f}.json")
    if st.button("Export metrics CSV"):
        if st.session_state.log:
            df = pd.DataFrame(st.session_state.log)
            st.download_button("Download metrics CSV", df.to_csv(index=False), file_name="shells_metrics.csv")
        else:
            st.warning("No metrics yet — Step to generate data.")

# ---------------------------
# Auto-run behavior
# ---------------------------
if st.session_state.run:
    # run a fixed number of internal steps per rerun to keep UI responsive
    for _ in range(4):
        compute_step()
    # log periodically
    if int(st.session_state.time / (st.session_state.dt * 6)) != 0 and (len(st.session_state.log) == 0 or st.session_state.log[-1]["t"] < st.session_state.time - 0.1):
        step_n(1)
    st.experimental_rerun()

# ---------------------------
# Build snapshot for visualization
# ---------------------------
def build_snapshot_for_viz():
    S, N = st.session_state.amps.shape
    # pack positions for concentric rings
    rings = []
    max_radius = 60
    for s in range(S):
        r = max_radius * (s+1) / (S+1)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        xs = (r * np.cos(theta)).tolist()
        ys = (r * np.sin(theta)).tolist()
        amps = st.session_state.amps[s].tolist()
        phases = st.session_state.phases[s].tolist()
        rings.append({
            "r": r,
            "x": xs,
            "y": ys,
            "amps": amps,
            "phases": phases,
            "shell_index": s,
            "mean_amp": float(np.mean(st.session_state.amps[s])),
            "mean_phase": float(np.angle(np.mean(np.exp(1j*st.session_state.phases[s]))))
        })
    return {
        "time": float(st.session_state.time),
        "shells": rings,
        "C": st.session_state.C.tolist(),
        "params": {
            "K_intra": float(st.session_state.K_intra),
            "C_inter": float(st.session_state.C_inter),
            "sigma": float(st.session_state.resonance_width)
        }
    }

snapshot = build_snapshot_for_viz()
snapshot_json = json.dumps(snapshot)

# ---------------------------
# Visualizer HTML (Three.js)
# ---------------------------
# note: double braces {{ }} to escape f-string braces for JS object literals
html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Dimensional shells viz</title>
<style>
  html,body {{ margin:0; padding:0; height:100%; background:#04121a; color:#dff7fa; overflow:hidden; }}
  #overlay {{ position:absolute; left:12px; top:12px; z-index:999; font-family:monospace; color:#bff; }}
</style>
</head>
<body>
<div id="overlay">t = {snapshot['time']:.3f} s</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.min.js"></script>

<script>
const snapshot = {snapshot_json};

// Setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x04121a);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 2000);
camera.position.set(0,0,220);
const renderer = new THREE.WebGLRenderer({{antialias:true, alpha:true}});
renderer.setSize(window.innerWidth*0.98, window.innerHeight*0.95);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enablePan = true;
controls.autoRotate = false;

// lighting
const ambient = new THREE.AmbientLight(0xffffff, 0.8);
scene.add(ambient);
const p = new THREE.PointLight(0x88ccff, 0.6);
p.position.set(200,200,200);
scene.add(p);

// helper to map amplitude -> size & color
function ampToVisual(a){{
  const minA = 0.1; const maxA = 2.0;
  const t = Math.min(1, Math.max(0, (a - minA) / (maxA - minA)));
  const size = 0.6 + 3.5 * t;
  // color: low = teal-blue, high = warm-cyan
  const c = new THREE.Color();
  c.setHSL(0.52 - 0.18*t, 0.85, 0.45 + 0.2*t);
  return {{ size, color: c }};
}}

// draw shells
const shellGroup = new THREE.Group();
scene.add(shellGroup);

for (let s=0; s<snapshot.shells.length; s++){{
  const shell = snapshot.shells[s];
  const ringGroup = new THREE.Group();
  ringGroup.userData.shellIndex = shell.shell_index;
  // ring line
  const ringGeom = new THREE.RingGeometry(shell.r - 1.0, shell.r + 1.0, 128);
  const ringMat = new THREE.MeshBasicMaterial({{ color: 0x08343a, side: THREE.DoubleSide, transparent:true, opacity:0.25 }});
  const ring = new THREE.Mesh(ringGeom, ringMat);
  ring.rotation.x = Math.PI/2;
  ringGroup.add(ring);

  // points
  for (let i=0; i<shell.x.length; i++){{
    const a = shell.amps[i];
    const vis = ampToVisual(a);
    const g = new THREE.SphereGeometry(vis.size, 10, 10);
    const m = new THREE.MeshStandardMaterial({{ color: vis.color, emissive: vis.color, emissiveIntensity: 0.4, metalness:0.2, roughness:0.3 }});
    const pnt = new THREE.Mesh(g, m);
    pnt.position.set(shell.x[i], shell.y[i], 0);
    ringGroup.add(pnt);
  }}

  // label
  const loader = new THREE.FontLoader();
  // skip font loading for remote; use simple circle marker
  shellGroup.add(ringGroup);
}

// draw inter-shell influence bands (visualize C with lines between mean radii)
const C = snapshot.C;
for (let i=0; i<C.length; i++){
  for (let j=0; j<C.length; j++){
    if (C[i][j] > 0.0001 && j>i){
      const ri = snapshot.shells[i].r;
      const rj = snapshot.shells[j].r;
      const geom = new THREE.TorusGeometry((ri+rj)/2, Math.abs(rj-ri)/8, 10, 64);
      const alpha = Math.min(0.9, C[i][j]*2.5);
      const mat = new THREE.MeshBasicMaterial({{ color: 0x4fe0da, transparent:true, opacity: alpha }});
      const tor = new THREE.Mesh(geom, mat);
      tor.rotation.x = Math.PI/2 * 0.98;
      scene.add(tor);
    }
  }
}

// subtle center core
const coreGeom = new THREE.CircleGeometry(8, 32);
const coreMat = new THREE.MeshBasicMaterial({{ color: 0x022a2c }});
const core = new THREE.Mesh(coreGeom, coreMat);
core.rotation.x = Math.PI/2;
scene.add(core);

// animate (gentle pulsing to show amplitude dynamics)
let t = 0;
function animate(){
  requestAnimationFrame(animate);
  t += 0.02;
  // gently pulse whole group for motion awareness
  shellGroup.rotation.z = 0.002 * t;
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', function(){{
  renderer.setSize(window.innerWidth*0.98, window.innerHeight*0.95);
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
}});
</script>
</body>
</html>
"""

components.html(html, height=820, scrolling=False)

# ---------------------------
# Right panel: metrics and charts
# ---------------------------
st.subheader("Metrics & diagnostics")
cols = st.columns(2)
with cols[0]:
    st.write("Time:", st.session_state.time)
    st.write("Shell count:", st.session_state.shell_count)
    st.write("Per shell:", st.session_state.per_shell)
    st.write("Last log entries (tail):")
    if st.session_state.log:
        st.dataframe(pd.DataFrame(st.session_state.log).tail(6))
    else:
        st.info("No logs yet. Use Step or Run to produce metrics.")
with cols[1]:
    # plot mean amplitude per shell (quick)
    S = st.session_state.amps.shape[0]
    means = [float(st.session_state.amps[s].mean()) for s in range(S)]
    df = pd.DataFrame({"shell": list(range(S)), "mean_amp": means})
    st.bar_chart(df.set_index("shell"))

st.markdown("---")
st.caption("Model notes: inter-shell coupling is filtered by frequency resonance; transfer requires frequency overlap and is time-delayed by dynamics. This prevents instantaneous collapse while enabling influence.")
