# shells_simulator.py
# Full-featured: mobile-optimized + reveal easing + timeline + pulse + extrude + inside/out + camera focus + energy fields + performance HUD
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime

st.set_page_config(page_title="Dimensional Shells Simulator — Full", layout="wide")
st.title("Dimensional Shells — Full Simulator Suite")
st.markdown("Layers simulator with timeline playback, reveal easing, pulse propagation, extruded shells, inside/outside view, energy fields, camera focus, and a performance HUD. Mobile optimized.")

# ---------------------------
# Initialization & state
# ---------------------------
def init_state():
    if "ready" not in st.session_state:
        st.session_state.ready = True
        # model params
        st.session_state.shell_count = 5
        st.session_state.per_shell = 40
        st.session_state.K_intra = 1.0
        st.session_state.C_inter = 0.25
        st.session_state.resonance_width = 0.5
        st.session_state.damping = 0.05
        st.session_state.dt = 0.03
        st.session_state.time = 0.0
        st.session_state.run = False
        # timeline & logs
        st.session_state.log = []   # list of summaries (dict)
        st.session_state.snapshots = []  # detailed snapshots
        st.session_state.record_timeline = False
        # visual options
        st.session_state.extruded = False
        st.session_state.inside_view = False
        st.session_state.reveal_easing = "easeOutCubic"
        st.session_state.reveal_delay = 0.35
        st.session_state.reveal_duration = 0.9
        st.session_state.pulse_enabled = True
        st.session_state.pulse_speed = 1.0
        st.session_state.energy_strength = 0.5
        st.session_state.show_hud = True
        reset_sim(seed=1234)

def reset_sim(seed=None):
    rng = np.random.default_rng(seed if seed is not None else int(time.time()%1e9))
    S = st.session_state.shell_count
    N = st.session_state.per_shell
    base = np.linspace(1.0, 6.0, S)
    omegas = np.zeros((S, N))
    amps = np.zeros((S, N))
    phases = np.zeros((S, N))
    for s in range(S):
        omegas[s] = rng.normal(loc=base[s], scale=0.2, size=N)
        amps[s] = rng.uniform(0.2, 1.0, size=N)
        phases[s] = rng.uniform(0, 2*np.pi, size=N)
    st.session_state.omegas = omegas
    st.session_state.amps = amps
    st.session_state.phases = phases
    C = np.zeros((S, S))
    for i in range(S-1):
        C[i,i+1] = st.session_state.C_inter
        C[i+1,i] = st.session_state.C_inter
    st.session_state.C = C
    st.session_state.time = 0.0
    st.session_state.log = []
    st.session_state.snapshots = []

init_state()

# ---------------------------
# Simulation core (unchanged)
# ---------------------------
def resonance_filter(dw, sigma):
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
    shell_mean_phases = np.array([shell_mean_phase(s) for s in range(S)])
    shell_mean_freqs = np.array([st.session_state.omegas[s].mean() for s in range(S)])
    for s in range(S):
        phases_s = st.session_state.phases[s]
        omegas_s = st.session_state.omegas[s]
        z = np.mean(np.exp(1j * phases_s))
        R = np.abs(z)
        Phi = np.angle(z)
        intra_term = st.session_state.K_intra * R * np.sin(Phi - phases_s)
        inter_term = np.zeros_like(phases_s)
        amp_drive = np.zeros_like(phases_s)
        for s2 in range(S):
            if st.session_state.C[s,s2] == 0:
                continue
            dw = shell_mean_freqs[s2] - shell_mean_freqs[s]
            w = resonance_filter(dw, sigma)
            Phi2 = shell_mean_phases[s2]
            inter_term += st.session_state.C[s,s2] * w * np.sin(Phi2 - phases_s)
            amp_drive += w * (st.session_state.amps[s2].mean() - st.session_state.amps[s]) * 0.3
        dphi = omegas_s + intra_term + inter_term
        new_phases[s] = (phases_s + dphi * dt) % (2*np.pi)
        gamma = st.session_state.damping
        align = 0.5 * (1 + np.cos(phases_s - Phi))
        dA = -gamma * st.session_state.amps[s] + 0.5 * align + 0.2 * amp_drive
        new_amps[s] = np.clip(st.session_state.amps[s] + dA * dt, 0.0, 5.0)
    st.session_state.phases = new_phases
    st.session_state.amps = new_amps
    st.session_state.time += dt

def step_n(n=1, record_snapshot=False):
    for _ in range(n):
        compute_step()
    S = st.session_state.omegas.shape[0]
    summary = {"t": float(st.session_state.time)}
    for s in range(S):
        summary[f"mean_amp_s{s}"] = float(st.session_state.amps[s].mean())
        summary[f"R_s{s}"] = float(np.abs(np.mean(np.exp(1j * st.session_state.phases[s]))))
    st.session_state.log.append(summary)
    if record_snapshot or st.session_state.record_timeline:
        st.session_state.snapshots.append(build_snapshot_for_viz())
    return summary

# ---------------------------
# Sidebar: controls & extras
# ---------------------------
with st.sidebar:
    st.header("Simulation controls / visual options")
    st.session_state.shell_count = st.number_input("Number of shells", min_value=2, max_value=12, value=st.session_state.shell_count, step=1)
    st.session_state.per_shell = st.number_input("Oscillators per shell", min_value=6, max_value=300, value=st.session_state.per_shell, step=2)
    st.session_state.K_intra = st.slider("Intra-shell coupling K", 0.0, 4.0, float(st.session_state.K_intra), 0.01)
    st.session_state.C_inter = st.slider("Inter-shell coupling (nearest neighbor)", 0.0, 1.0, float(st.session_state.C_inter), 0.01)
    st.session_state.resonance_width = st.slider("Resonance width σ", 0.01, 2.0, float(st.session_state.resonance_width), 0.01)
    st.session_state.damping = st.slider("Amplitude damping γ", 0.0, 1.0, float(st.session_state.damping), 0.01)
    st.session_state.dt = st.number_input("Time step dt", value=float(st.session_state.dt), min_value=0.001, max_value=0.1, step=0.001, format="%.3f")
    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("Step 1"):
        s = step_n(1, record_snapshot=False)
        st.write("Step:", s)
    if col2.button("Step 10"):
        st.write("Stepped 10: t =", step_n(10)["t"])
    if st.button("Reset (random seed)"):
        reset_sim(seed=int(time.time())%1000000)
    if st.button("Rebuild (apply new sizes)"):
        reset_sim(seed=123)
    st.markdown("---")
    st.subheader("Visualizer options")
    st.session_state.extruded = st.checkbox("Extruded shells (3D tubes)", value=st.session_state.extruded)
    st.session_state.inside_view = st.checkbox("Inside / Outside view (flip camera)", value=st.session_state.inside_view)
    st.session_state.pulse_enabled = st.checkbox("Enable pulse propagation view", value=st.session_state.pulse_enabled)
    st.session_state.pulse_speed = st.slider("Pulse speed", 0.1, 4.0, float(st.session_state.pulse_speed), 0.1)
    st.session_state.energy_strength = st.slider("Energy field strength", 0.0, 2.0, float(st.session_state.energy_strength), 0.05)
    st.session_state.show_hud = st.checkbox("Show performance HUD", value=st.session_state.show_hud)
    st.markdown("---")
    st.subheader("Reveal settings")
    st.session_state.reveal_easing = st.selectbox("Reveal easing", ["linear", "easeOutCubic", "elasticOut", "backOut"], index=["linear","easeOutCubic","elasticOut","backOut"].index(st.session_state.reveal_easing))
    st.session_state.reveal_delay = st.slider("Reveal delay (s)", 0.05, 1.5, float(st.session_state.reveal_delay), 0.05)
    st.session_state.reveal_duration = st.slider("Reveal duration (s)", 0.1, 2.5, float(st.session_state.reveal_duration), 0.05)
    st.markdown("---")
    st.subheader("Timeline & recording")
    st.session_state.record_timeline = st.checkbox("Record timeline automatically", value=st.session_state.record_timeline)
    if st.button("Capture snapshot now"):
        st.session_state.snapshots.append(build_snapshot_for_viz())
        st.success(f"Captured snapshot t={st.session_state.time:.3f}")
    if st.button("Clear timeline / snapshots"):
        st.session_state.snapshots = []
        st.session_state.log = []
        st.success("Cleared timeline and logs.")
    st.markdown("---")
    st.subheader("Export")
    if st.button("Export snapshots (JSON)"):
        if st.session_state.snapshots:
            fname = f"snapshots_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
            st.download_button("Download JSON", data=json.dumps(st.session_state.snapshots, indent=2), file_name=fname)
        else:
            st.warning("No snapshots recorded.")
    if st.button("Export metrics (CSV)"):
        if st.session_state.log:
            df = pd.DataFrame(st.session_state.log)
            st.download_button("Download CSV", df.to_csv(index=False), file_name="metrics.csv")
        else:
            st.warning("No metrics yet.")
    st.markdown("---")
    st.write("Quick: Click a shell in the visualizer to focus camera on it (client-side).")

# ---------------------------
# Auto-run: stepping + record
# ---------------------------
if st.session_state.run:
    for _ in range(4):
        compute_step()
    # periodically record summary snapshot
    if st.session_state.record_timeline:
        st.session_state.snapshots.append(build_snapshot_for_viz())
    st.rerun()

# ---------------------------
# Snapshot builder (for injection to JS)
# ---------------------------
def build_snapshot_for_viz():
    S, N = st.session_state.amps.shape
    rings = []
    max_radius = 60
    for s in range(S):
        r = max_radius * (s+1) / (S+1)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        xs = (r * np.cos(theta)).tolist()
        ys = (r * np.sin(theta)).tolist()
        rings.append({
            "r": r,
            "x": xs,
            "y": ys,
            "amps": st.session_state.amps[s].tolist(),
            "phases": st.session_state.phases[s].tolist(),
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
        },
        "visual": {
            "extruded": bool(st.session_state.extruded),
            "inside_view": bool(st.session_state.inside_view),
            "pulse_enabled": bool(st.session_state.pulse_enabled),
            "pulse_speed": float(st.session_state.pulse_speed),
            "energy_strength": float(st.session_state.energy_strength),
            "reveal": {
                "easing": st.session_state.reveal_easing,
                "delay": float(st.session_state.reveal_delay),
                "duration": float(st.session_state.reveal_duration)
            },
            "show_hud": bool(st.session_state.show_hud)
        }
    }

snapshot = build_snapshot_for_viz()
snapshot_json = json.dumps(snapshot)

# ---------------------------
# Visualizer HTML + Three.js (implements all features)
# ---------------------------
# Use double braces {{}} where JS uses braces to avoid f-string interpolation issues.
html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Dimensional Shells — Full Visualizer</title>
<style>
  html,body {{ margin:0; padding:0; height:100%; background:#04121a; color:#dff7fa; overflow:hidden; }}
  #overlay {{ position:absolute; left:12px; top:12px; z-index:999; font-family:monospace; color:#bff; }}
  #hud {{ position:absolute; right:12px; top:12px; z-index:999; font-family:monospace; color:#bff; text-align:right; }}
  #timelineControls {{ position:absolute; left:12px; bottom:12px; z-index:999; font-family:monospace; color:#bff; }}
  .btn {{ background: rgba(255,255,255,0.06); color:#bff; border:0; padding:6px 10px; margin-right:6px; border-radius:6px; cursor:pointer; }}
</style>
</head>
<body>
<div id="overlay">t = {snapshot['time']:.3f} s</div>
<div id="hud"></div>
<div id="timelineControls"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.min.js"></script>

<script>
// ===== Snapshot injected from Streamlit =====
const snapshot = {snapshot_json};

// ===== Mobile detection & camera presets =====
const isMobile = window.innerWidth < 700;
const cameraDist = snapshot.visual && snapshot.visual.inside_view ? (isMobile?40:70) : (isMobile?140:220);

// ===== Scene, camera, renderer =====
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x04121a);
const camera = new THREE.PerspectiveCamera(isMobile?70:60, window.innerWidth/window.innerHeight, 0.1, 2000);
camera.position.set(0,0,cameraDist);

const params = {{ alpha: false }};
if (!isMobile) params.antialias = true;
const renderer = new THREE.WebGLRenderer(params);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth*0.98, window.innerHeight*0.95);
renderer.setClearColor(0x071220, 1);
document.body.appendChild(renderer.domElement);

// Controls
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enablePan = true;
controls.autoRotate = false;
controls.rotateSpeed = isMobile?0.4:0.8;

// Lighting
const ambient = new THREE.AmbientLight(0xffffff, isMobile?0.8:1.0);
scene.add(ambient);
const pLight = new THREE.PointLight(0x88ccff, isMobile?0.6:1.4);
pLight.position.set(200,200,200);
scene.add(pLight);

// Utility easing functions
function easeLinear(t) {{ return t; }}
function easeOutCubic(t) {{ return 1 - Math.pow(1 - t, 3); }}
function elasticOut(t) {{
  const p = 0.3;
  return Math.pow(2, -10*t) * Math.sin((t - p/4)*(2*Math.PI)/p) + 1;
}}
function backOut(t) {{
  const s = 1.70158;
  return 1 + (--t)*t*((s+1)*t + s);
}}
function getEasing(name) {{
  if (name === "linear") return easeLinear;
  if (name === "easeOutCubic") return easeOutCubic;
  if (name === "elasticOut") return elasticOut;
  if (name === "backOut") return backOut;
  return easeOutCubic;
}}

// Visual groups & state
const shellGroup = new THREE.Group(); scene.add(shellGroup);
const ringGroups = [];
const ringMaterials = [];
const pointMaterials = [];
const torusList = [];
const energyFields = [];

// Build geometry (supports extruded option)
function buildVisuals(snap) {{
  // Clear previous
  while (shellGroup.children.length) shellGroup.remove(shellGroup.children[0]);
  ringGroups.length = 0; ringMaterials.length = 0; pointMaterials.length = 0;
  torusList.length = 0; energyFields.length = 0;

  const S = snap.shells.length;
  const extruded = snap.visual && snap.visual.extruded;
  const inside = snap.visual && snap.visual.inside_view;
  const baseRingOpacity = isMobile?0.55:0.32;

  for (let s=0; s<S; s++) {{
    const shell = snap.shells[s];
    const group = new THREE.Group();
    group.userData.shellIndex = s;

    // ring or tube geometry
    const inner = Math.max(0.8, shell.r - (isMobile?2.5:1.0));
    const outer = shell.r + (isMobile?2.5:1.0);

    if (extruded) {{
      // use Torus-like extruded tube as shell
      const rad = (inner + outer)/2;
      const thickness = Math.max(1.2, (outer - inner)/2);
      const geom = new THREE.TorusGeometry(rad, thickness, 8, Math.max(32, shell.x.length));
      const mat = new THREE.MeshBasicMaterial({{ color: 0x0aa9a0, transparent:true, opacity:0.0, side: THREE.DoubleSide }});
      const tor = new THREE.Mesh(geom, mat);
      tor.rotation.x = Math.PI/2;
      group.add(tor);
      ringMaterials.push(mat);
    }} else {{
      const ringGeom = new THREE.RingGeometry(inner, outer, Math.max(32, shell.x.length));
      const mat = new THREE.MeshBasicMaterial({{ color: 0x0aa9a0, side: THREE.DoubleSide, transparent:true, opacity:0.0 }});
      const mesh = new THREE.Mesh(ringGeom, mat);
      mesh.rotation.x = Math.PI/2;
      group.add(mesh);
      ringMaterials.push(mat);
    }}

    // points
    const mats = [];
    for (let i=0; i<shell.x.length; i++) {{
      const a = shell.amps[i];
      const tnorm = Math.min(1, Math.max(0, (a - 0.1) / (2.0 - 0.1)));
      const size = (isMobile?0.9:0.6) + (isMobile?4.5:3.5)*tnorm;
      const col = new THREE.Color(); col.setHSL(0.52 - 0.18*tnorm, 0.88, 0.45 + 0.28*tnorm);
      const geo = new THREE.SphereGeometry(size, isMobile?8:12, isMobile?8:12);
      const mat = isMobile ? new THREE.MeshBasicMaterial({{ color: col, transparent:true, opacity:0.0 }}) :
                            new THREE.MeshStandardMaterial({{ color: col, emissive: col, emissiveIntensity: 0.0, transparent:true, opacity:0.0, metalness:0.25, roughness:0.25 }});
      const pnt = new THREE.Mesh(geo, mat);
      pnt.position.set(shell.x[i], shell.y[i], 0);
      group.add(pnt);
      mats.push(mat);
    }}
    pointMaterials.push(mats);

    // initial scale small for reveal
    group.scale.set(0.4,0.4,0.4);
    shellGroup.add(group);
    ringGroups.push(group);
  }}

  // inter-shell tori (energy connectors)
  const C = snap.C;
  for (let i=0;i<C.length;i++) {{
    for (let j=i+1;j<C.length;j++) {{
      if (C[i][j] > 0.0001) {{
        const ri = snap.shells[i].r;
        const rj = snap.shells[j].r;
        const radius = (ri + rj) / 2;
        const thickness = Math.max(0.7, Math.abs(rj - ri) / 8);
        const geom = new THREE.TorusGeometry(radius, thickness, 8, 64);
        const mat = new THREE.MeshBasicMaterial({{ color: 0x4fe0da, transparent:true, opacity:0.0 }});
        const tor = new THREE.Mesh(geom, mat);
        tor.rotation.x = Math.PI/2 * 0.98;
        scene.add(tor);
        torusList.push({{mesh:tor, mat:mat, i:i, j:j}});
      }}
    }}
  }}

  // energy fields (glow tori) - optional later
  for (let s=0;s<snap.shells.length;s++) {{
    const ri = snap.shells[s].r;
    const geom = new THREE.TorusGeometry(ri, Math.max(0.6, ri*0.02), 8, 64);
    const mat = new THREE.MeshBasicMaterial({{ color: 0x22f0d8, transparent:true, opacity:0.0 }});
    const tor = new THREE.Mesh(geom, mat);
    tor.rotation.x = Math.PI/2 * 0.98;
    scene.add(tor);
    energyFields.push({{mesh:tor, mat:mat}});
  }}
}}
buildVisuals(snapshot);

// ===== Reveal Animation & Pulse propagation & Focus logic =====
const easingFunc = getEasing(snapshot.visual.reveal.easing || "easeOutCubic");
const revealDelay = snapshot.visual.reveal.delay || 0.35;
const revealDuration = snapshot.visual.reveal.duration || 0.9;
let clockT = 0;
let revealed = false;

// pulse propagation state
let pulseClock = 0;

// Performance HUD
let lastFrame = performance.now();
let fps = 0;
const hudEl = document.getElementById('hud');

function updateHUD() {{
  const device = navigator.userAgent;
  hudEl.innerHTML = (snapshot.visual.show_hud ? `FPS: ${fps.toFixed(0)}<br/>Device: ${device.split(')')[0]}<br/>Shells: ${snapshot.shells.length}` : '');
}}

// camera focus function (client-side only)
function focusShell(index) {{
  if (index < 0 || index >= ringGroups.length) return;
  const group = ringGroups[index];
  // compute world position center (0,0) for shells — zoom to radius
  const r = snapshot.shells[index].r;
  // set camera to radius * factor
  const z = snapshot.visual.inside_view ? r * 0.6 : r * 3.2;
  // smooth camera move (lerp)
  const start = camera.position.clone();
  const target = new THREE.Vector3(0, 0, z);
  let t0 = 0;
  const dur = 600;
  const anim = function(timestamp) {{
    if (!t0) t0 = timestamp;
    const dt = (timestamp - t0)/dur;
    const tt = Math.min(1, dt);
    camera.position.lerpVectors(start, target, tt);
    camera.lookAt(0,0,0);
    if (tt < 1) requestAnimationFrame(anim);
  }};
  requestAnimationFrame(anim);
}

// Click picking for focus
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
renderer.domElement.addEventListener('click', (ev) => {{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(shellGroup.children, true);
  if (intersects.length) {{
    // find group index
    let obj = intersects[0].object;
    while (obj && !obj.parent) obj = obj.parent;
    // try to find group index by traversing up
    let parent = intersects[0].object.parent;
    let groupIndex = -1;
    for (let i=0;i<ringGroups.length;i++) {{
      if (ringGroups[i].uuid === parent.uuid || ringGroups[i].uuid === parent.parent.uuid) {{ groupIndex = i; break; }}
    }}
    if (groupIndex >= 0) focusShell(groupIndex);
  }}
}});

// Timeline playback UI (client-side)
const timelineDiv = document.getElementById('timelineControls');
timelineDiv.innerHTML = `
  <button class="btn" id="playBtn">Play</button>
  <button class="btn" id="pauseBtn">Pause</button>
  <button class="btn" id="rewindBtn">⟲</button>
  <button class="btn" id="stepBackBtn">‹</button>
  <button class="btn" id="stepFwdBtn">›</button>
  <span id="timeLabel" style="margin-left:10px"></span>
`;
// timeline data injected from Streamlit snapshots (if any)
let timeline = [];
try {{
  timeline = window.parent && window.parent._st_raw_snapshots ? window.parent._st_raw_snapshots : [];
}} catch(e) {{ timeline = []; }}

// But we will request snapshots from Streamlit via the provided initial JSON if user captured snapshots
// For safety, we'll also allow the app to use the single current snapshot for playback as a static frame.
if (!timeline.length && window.__ST_SNAPSHOTS && Array.isArray(window.__ST_SNAPSHOTS)) {{
  timeline = window.__ST_SNAPSHOTS;
}}

// Helper to load a snapshot into the scene quickly (client-side)
function loadSnapshotLocal(snapObj) {{
  // rebuild visuals to reflect snapshot (cheap / fast)
  try {{
    buildVisuals(snapObj);
  }} catch(e) {{
    console.warn("Failed to load snapshot locally:", e);
  }}
}}

// Playback controls (if server-side snapshots exist, Streamlit will inject them via JSON download; for now use current single snapshot)
let playState = {{ playing:false, index:0, speed:1 }};
document.getElementById('playBtn').addEventListener('click', ()=>{{ playState.playing=true; }});
document.getElementById('pauseBtn').addEventListener('click', ()=>{{ playState.playing=false; }});
document.getElementById('rewindBtn').addEventListener('click', ()=>{{ playState.index=0; playState.playing=false; }});
document.getElementById('stepBackBtn').addEventListener('click', ()=>{{ playState.index = Math.max(0, playState.index-1); if (timeline.length) loadSnapshotLocal(timeline[playState.index]); }});
document.getElementById('stepFwdBtn').addEventListener('click', ()=>{{ playState.index = Math.min(timeline.length-1, playState.index+1); if (timeline.length) loadSnapshotLocal(timeline[playState.index]); }});

// If timeline empty, we allow a simple "pulse preview" using the current snapshot object
if (!timeline.length) {{
  // create a minimal timeline by sampling small animated states (client-side)
  timeline = [snapshot]; // single-frame fallback
}}

// ===== Animate loop (reveal + pulse + FPS HUD) =====
let last = performance.now();
function animate() {{
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = (now - last) / 1000;
  last = now;
  // FPS calc
  fps = 0.9*fps + 0.1*(1/dt);

  clockT += dt;
  // Reveal: sequential fade + scale
  const total = ringGroups.length;
  const easing = getEasing(snapshot.visual.reveal.easing || "easeOutCubic");
  for (let i=0;i<total;i++) {{
    const start = i * revealDelay;
    const raw = (clockT - start) / revealDuration;
    const p = Math.max(0, Math.min(1, raw));
    const eased = easing(p);
    const sVal = 0.4 + 0.6 * eased;
    ringGroups[i].scale.set(sVal, sVal, sVal);
    // ring material(s) opacity
    const mats = ringGroups[i].children.filter(c=>c.material).map(c=>c.material);
    for (let m of mats) {{
      if (m) m.opacity = (baseOpacity = (isMobile?0.55:0.32)) * eased;
    }}
    // points
    const pts = pointMaterials[i];
    if (pts) {{
      for (let pm of pts) {{
        if (pm) {{
          pm.opacity = 0.95 * eased;
          if ('emissiveIntensity' in pm) pm.emissiveIntensity = 0.6 * eased;
        }}
      }}
    }}
    // torus connectors
    if (i < torusList.length) {{
      torusList[i].mat.opacity = Math.max(0.06, torusList[i].mat.opacity + (0.9*eased - torusList[i].mat.opacity) * 0.15);
    }}
    // energy field
    if (energyFields[i]) {{
      energyFields[i].mat.opacity = Math.min(0.25, eased * (snapshot.visual.energy_strength || 0.5));
      energyFields[i].mesh.rotation.z = clockT * 0.05 * (1 + i*0.02);
    }}
  }}

  // Pulse propagation: radial brightness ripple
  if (snapshot.visual.pulse_enabled) {{
    pulseClock += dt * (snapshot.visual.pulse_speed || 1.0);
    const center = clockT * (snapshot.visual.pulse_speed || 1.0) * 0.5;
    for (let s=0;s<snapshot.shells.length;s++) {{
      const r = snapshot.shells[s].r;
      // wave function
      const wave = 0.5 + 0.5*Math.sin(center - r*0.06);
      // apply subtle emissive modulation
      const pts = pointMaterials[s];
      if (pts) {{
        for (let pm of pts) {{
          if ('emissiveIntensity' in pm) pm.emissiveIntensity = 0.2 + 0.6*wave*(snapshot.visual.energy_strength||0.5);
        }}
      }}
    }}
  }}

  // Timeline playback client-side if requested
  if (playState.playing && timeline.length) {{
    // simple frame-advance per second scaled by speed
    const framesPerSecond = 2 * (playState.speed || 1);
    if (!playState._acc) playState._acc = 0;
    playState._acc += dt * framesPerSecond;
    if (playState._acc >= 1) {{
      playState._acc = 0;
      playState.index = Math.min(timeline.length-1, playState.index+1);
      loadSnapshotLocal(timeline[playState.index]);
      document.getElementById('timeLabel').innerText = `frame ${playState.index+1}/${timeline.length}`;
    }}
  }}

  // update HUD
  if (snapshot.visual.show_hud) updateHUD();

  controls.update();
  renderer.render(scene, camera);
}}
animate();

// expose a simple API to parent for snapshot injection (Streamlit cannot call client directly easily)
window.__shells_viz = {{
  focusShell: focusShell,
  loadSnapshot: loadSnapshotLocal,
  setTimeline: function(t){{
    timeline = t || [];
  }},
  play: function(){ playState.playing = true; },
  pause: function(){ playState.playing = false; },
  goTo: function(idx){{ playState.index = idx; if (timeline.length) loadSnapshotLocal(timeline[idx]); }},
  currentTime: function(){{ return clockT; }}
}};

</script>
</body>
</html>
"""

# ---------------------------
# Render visualizer
# ---------------------------
components.html(html, height=860, scrolling=False)

# ---------------------------
# Right column: metrics, logs, timeline
# ---------------------------
st.subheader("Metrics, Timeline & Controls")
colA, colB = st.columns([1,1])
with colA:
    st.write("Time:", st.session_state.time)
    st.write("Shells:", st.session_state.shell_count, " | Per shell:", st.session_state.per_shell)
    st.write("Log tail:")
    if st.session_state.log:
        st.dataframe(pd.DataFrame(st.session_state.log).tail(8))
    else:
        st.info("No logs yet. Use Step or enable record timeline.")
with colB:
    st.write("Snapshots recorded:", len(st.session_state.snapshots))
    if st.session_state.snapshots:
        if st.button("Load last snapshot into visualizer (client-side)"):
            # inject via JS by rendering a small script that calls our client API
            last_snap = st.session_state.snapshots[-1]
            js = f"<script>if(window.__shells_viz) window.__shells_viz.loadSnapshot({json.dumps(last_snap)});</script>"
            components.html(js, height=10)
        if st.button("Focus camera on last snapshot shell 0"):
            js = "<script>if(window.__shells_viz) window.__shells_viz.focusShell(0);</script>"
            components.html(js, height=10)

st.markdown("---")
st.caption("Notes: Timeline playback is client-side using recorded snapshots. To build a timeline, toggle 'Record timeline automatically' in the sidebar or capture snapshots manually.")
