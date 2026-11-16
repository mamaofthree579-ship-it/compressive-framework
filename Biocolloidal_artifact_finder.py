# SPDX-License-Identifier: GPL-3.0-or-later
# Biocolloidal Vessel Geometry Detector & Medium Acoustic Activation Workbench

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import streamlit as st

st.set_page_config(
    page_title="Biocolloidal Geometry & Medium Activation Lab", layout="wide"
)
st.title("🧪 Biocolloidal Geometry & Medium Acoustic Activation Lab")

# =========================================================
# MODULE 1: VESSEL GEOMETRY & RESIDUE ANOMALY DETECTOR
# =========================================================
FEATURES = [
    "organic_diversity_score",
    "residue_uniformity_score",
    "salt_concentration_ratio",
    "vitrification_index",
    "geometric_symmetry_ratio",
]


def generate_vessel_dataset(seed=42):
  rng = np.random.default_rng(seed)
  n_norm = 200
  df_norm = pd.DataFrame({
      "artifact_id": [f"COOK-{i:03d}" for i in range(n_norm)],
      "classification": ["Domestic Cooking"] * n_norm,
      "organic_diversity_score": rng.uniform(0.60, 1.00, n_norm),
      "residue_uniformity_score": rng.uniform(0.10, 0.40, n_norm),
      "salt_concentration_ratio": rng.uniform(0.01, 0.05, n_norm),
      "vitrification_index": rng.uniform(0.00, 0.12, n_norm),
      "geometric_symmetry_ratio": rng.uniform(0.30, 0.65, n_norm),
  })

  n_anom = 12
  df_anom = pd.DataFrame({
      "artifact_id": [f"REACTOR-{i:03d}" for i in range(n_anom)],
      "classification": ["Unknown / Process Candidate"] * n_anom,
      "organic_diversity_score": rng.uniform(0.02, 0.20, n_anom),
      "residue_uniformity_score": rng.uniform(0.82, 0.99, n_anom),
      "salt_concentration_ratio": rng.uniform(0.25, 0.60, n_anom),
      "vitrification_index": rng.uniform(0.65, 0.98, n_anom),
      "geometric_symmetry_ratio": rng.uniform(0.88, 0.99, n_anom),
  })

  df_full = (
      pd.concat([df_norm, df_anom])
      .sample(frac=1, random_state=seed)
      .reset_index(drop=True)
  )
  return df_norm, df_full


# =========================================================
# MODULE 2: STANDALONE MEDIUM ACOUSTIC ACTIVATION ENGINE
# =========================================================
MEDIUM_PRESETS = {
    "Custom Manual Tuning": {
        "sal": 25.0,
        "clay": 0.08,
        "r": 2.5,
        "rho": 1000.0,
        "c": 1480.0,
        "desc": "User-defined medium parameters.",
    },
    "Maya Blue Sol (Palygorskite + Indigo)": {
        "sal": 22.0,
        "clay": 0.12,
        "r": 1.8,
        "rho": 1000.0,
        "c": 1480.0,
        "desc": (
            "Aqueous alkaline palygorskite clay + indigo precursor prior to"
            " thermal curing."
        ),
    },
    "Harappan Alkali-Silica (Carnelian Paste)": {
        "sal": 30.0,
        "clay": 0.10,
        "r": 1.5,
        "rho": 1000.0,
        "c": 1480.0,
        "desc": (
            "Wood-ash sodium carbonate alkali + micro-silica paste for etched"
            " carnelian beads."
        ),
    },
    "Blombos Ochre-Lipid Emulsion (100 ka)": {
        "sal": 12.0,
        "clay": 0.18,
        "r": 2.4,
        "rho": 930.0,
        "c": 1410.0,
        "desc": (
            "Finely ground hematite/goethite suspended in liquefied bone-marrow"
            " fat & mineral salts."
        ),
    },
    "Mesopotamian Bitumen-Brine Mortar": {
        "sal": 48.0,
        "clay": 0.05,
        "r": 3.5,
        "rho": 1015.0,
        "c": 1510.0,
        "desc": (
            "Natural asphalt droplets + clay filler dispersed in high-salinity"
            " brine."
        ),
    },
}


def simulate_medium_activation(
    freq_hz: float,
    salinity_ppt: float,
    clay_fraction: float,
    acoustic_pressure_pa: float,
    particle_radius_um: float = 2.5,
    base_rho: float = 1000.0,
    base_c: float = 1480.0,
):
  rho_medium = base_rho + (0.75 * salinity_ppt) + (650.0 * clay_fraction)
  c_medium = base_c + (1.2 * salinity_ppt) + (320.0 * clay_fraction)
  impedance_rayl = rho_medium * c_medium

  wavelength_m = c_medium / max(freq_hz, 1.0)
  k_wave = 2.0 * np.pi / wavelength_m
  r_p = particle_radius_um * 1e-6

  phi_contrast = 0.42 + 0.15 * (salinity_ppt / 50.0)
  e_ac = (acoustic_pressure_pa**2) / (4.0 * rho_medium * (c_medium**2))
  f_rad_n = 4.0 * np.pi * (r_p**3) * k_wave * e_ac * phi_contrast
  f_rad_un = f_rad_n * 1e6

  ionic_stability = np.exp(-(((salinity_ppt - 25.0) / 18.0) ** 2))
  harmonic_window = np.exp(
      -(((np.log10(max(f_rad_un, 1e-4)) - 0.35) / 0.85) ** 2)
  )
  activation_index = float(
      np.clip(0.55 * harmonic_window + 0.45 * ionic_stability, 0.0, 1.0)
  )

  if activation_index >= 0.80:
    regime = "Optimal Emulsification"
  elif activation_index >= 0.50:
    regime = "Active Suspension"
  elif f_rad_un > 5.0:
    regime = "Acoustic Flocculation"
  else:
    regime = "Gravitational Settling"

  return {
      "rho_medium": round(rho_medium, 1),
      "c_medium": round(c_medium, 1),
      "impedance_mrayl": round(impedance_rayl / 1e6, 3),
      "wavelength_m": round(wavelength_m, 2),
      "f_rad_un": round(f_rad_un, 4),
      "activation_index": round(activation_index, 3),
      "regime": regime,
  }


# =========================================================
# STREAMLIT INTERFACE
# =========================================================
tab1, tab2 = st.tabs([
    "🏺 1. Geometric & Chemical Anomaly Detector",
    "〰️ 2. Standalone Medium Acoustic Activation",
])

with tab1:
  st.subheader("Vessel Geometry + Chemical Residue Anomaly Detection")
  contamination = st.slider(
      "Baseline Contamination Tolerance", 0.005, 0.05, 0.01
  )
  df_train, df_all = generate_vessel_dataset()

  clf = IsolationForest(contamination=contamination, random_state=42)
  clf.fit(df_train[FEATURES])

  df_all["anomaly_score"] = clf.decision_function(df_all[FEATURES]).round(4)
  df_all["is_anomaly"] = clf.predict(df_all[FEATURES]) == -1

  flagged = df_all[df_all["is_anomaly"]].sort_values(by="anomaly_score")
  c1, c2 = st.columns(2)
  c1.metric("Total Artifacts Audited", len(df_all))
  c2.metric("High-Symmetry Process Vessels Flagged", len(flagged))

  st.dataframe(flagged, use_container_width=True)
  fig1 = px.scatter(
      df_all,
      x="geometric_symmetry_ratio",
      y="vitrification_index",
      color="anomaly_score",
      symbol="is_anomaly",
      hover_name="artifact_id",
      hover_data=FEATURES,
      color_continuous_scale="RdYlBu",
      title=(
          "Vessel Geometric Symmetry vs. Thermal Vitrification (Outlier"
          " Separation)"
      ),
  )
  st.plotly_chart(fig1, use_container_width=True)

with tab2:
  st.subheader("Standalone Medium Harmony & Colloidal Activation Engine")
  selected_medium = st.selectbox(
      "Select Ancient Colloidal Medium Preset", list(MEDIUM_PRESETS.keys())
  )
  preset = MEDIUM_PRESETS[selected_medium]
  st.caption(preset["desc"])

  col_a, col_b = st.columns([1, 2])
  with col_a:
    freq_hz = st.slider("Harmonic Excitation Frequency (Hz)", 20, 1200, 432)
    salinity_ppt = st.slider(
        "Medium Electrolyte Salinity (ppt)", 0.0, 60.0, float(preset["sal"])
    )
    clay_fraction = st.slider(
        "Mineral / Clay Colloid Fraction", 0.0, 0.30, float(preset["clay"])
    )
    particle_r = st.slider(
        "Colloid Particle Radius (μm)", 0.5, 5.0, float(preset["r"])
    )
    pressure_pa = st.slider(
        "Acoustic Driving Pressure (Pa)", 50.0, 600.0, 280.0
    )

  res = simulate_medium_activation(
      freq_hz,
      salinity_ppt,
      clay_fraction,
      pressure_pa,
      particle_radius_um=particle_r,
      base_rho=preset["rho"],
      base_c=preset["c"],
  )

  with col_b:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Acoustic Impedance", f"{res['impedance_mrayl']} MRayl")
    m2.metric("Radiation Force", f"{res['f_rad_un']} μN")
    m3.metric("Activation Index", f"{res['activation_index']}")
    m4.metric("Colloidal State", str(res["regime"]).split(" (")[0])

    freqs = np.linspace(20, 1200, 150)
    sweep_df = pd.DataFrame([
        {
            "Frequency (Hz)": f,
            "Activation Index": simulate_medium_activation(
                f,
                salinity_ppt,
                clay_fraction,
                pressure_pa,
                particle_radius_um=particle_r,
                base_rho=preset["rho"],
                base_c=preset["c"],
            )["activation_index"],
        }
        for f in freqs
    ])
    fig2 = px.line(
        sweep_df,
        x="Frequency (Hz)",
        y="Activation Index",
        title=(
            f"Colloidal Medium Activation Curve: {selected_medium}"
            f" (Salinity={salinity_ppt} ppt)"
        ),
    )
    fig2.add_vline(
        x=freq_hz,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Active: {freq_hz} Hz",
    )
    st.plotly_chart(fig2, use_container_width=True)

