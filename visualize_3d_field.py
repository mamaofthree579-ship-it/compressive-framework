#!/usr/bin/env python3
"""
visualize_3d_field.py

Interactive 3D visualization of a quantum compressive field using Plotly.

Usage:
    python visualize_3d_field.py
"""

import numpy as np
import plotly.graph_objects as go

# Grid
nx = 160
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, nx)
X, Y = np.meshgrid(x, y)

# Parameters (tweak these)
a = 0.5                # base curvature coefficient
grav_amp = 0.4         # graviton-style modulation amplitude
chron_amp = 0.3        # chronon-style modulation amplitude
cognon_amp = 0.2       # cognon-style modulation amplitude
fluct_amp = 0.06       # random fluctuation amplitude

# Base compressive field (radial envelope + harmonic content)
radial = np.exp(-0.15 * (X**2 + Y**2))
field_base = (np.sin(X**2 + Y**2) * radial * a
              + grav_amp * np.cos(2 * X) * radial
              + chron_amp * np.sin(1.5 * Y) * radial
              - cognon_amp * np.sin(3 * X) * np.cos(2 * Y) * radial)

# Add small stochastic quantum fluctuations (for realism)
np.random.seed(42)
noise = fluct_amp * np.random.randn(*field_base.shape)
field = field_base + noise

# Optional curvature measure (e.g. Laplacian) to show compressive hotspots
dx = x[1] - x[0]
lap = (np.roll(field, -1, axis=0) - 2*field + np.roll(field, 1, axis=0)) / dx**2 \
    + (np.roll(field, -1, axis=1) - 2*field + np.roll(field, 1, axis=1)) / dx**2

# Pick which to display: 'field' or 'lap'
display = "field"  # "field" or "lap"
Z = field if display == "field" else lap

# Normalize for nicer coloring
Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)

# Plotly surface
fig = go.Figure(data=[
    go.Surface(x=X, y=Y, z=Z_norm,
               colorscale='plasma',
               colorbar=dict(title="Intensity"),
               showscale=True,
               cmin=0, cmax=1)
])

fig.update_layout(
    title="3D Quantum Compressive Field (interactive)",
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='normalized intensity'
    ),
    width=900, height=700
)

# Show in browser / open in notebook
fig.show()
