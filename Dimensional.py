import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")

st.title("Dimensional Construction Model")

# ---------------------------
# Session state
# ---------------------------
if "dim" not in st.session_state:
    st.session_state.dim = 1

# ---------------------------
# Controls
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Previous Dimension"):
        st.session_state.dim = max(1, st.session_state.dim - 1)

with col2:
    if st.button("➡️ Next Dimension"):
        st.session_state.dim = min(5, st.session_state.dim + 1)

dim = st.session_state.dim

st.markdown(f"### Current Dimension: {dim}D")

# ---------------------------
# Geometry builders
# ---------------------------

def build_line():
    x = np.linspace(0, 1, 50)
    return x, np.zeros_like(x), np.zeros_like(x)

def build_square():
    pts = []
    for i in np.linspace(0,1,20):
        pts.append((i,0))
        pts.append((i,1))
        pts.append((0,i))
        pts.append((1,i))
    return pts

def build_cube():
    pts = []
    for x in [0,1]:
        for y in np.linspace(0,1,10):
            for z in np.linspace(0,1,10):
                pts.append((x,y,z))
    for y in [0,1]:
        for x in np.linspace(0,1,10):
            for z in np.linspace(0,1,10):
                pts.append((x,y,z))
    for z in [0,1]:
        for x in np.linspace(0,1,10):
            for y in np.linspace(0,1,10):
                pts.append((x,y,z))
    return pts

def build_tesseract_projection():
    pts = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    # project 4D → 3D
                    scale = 1 / (1 + w*0.8)
                    pts.append((x*scale, y*scale, z*scale))
    return pts

def build_5d_projection():
    pts = []
    for a in [0,1]:
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    for w in [0,1]:
                        scale = 1 / (1 + w*0.6 + a*0.6)
                        pts.append((x*scale, y*scale, z*scale))
    return pts

# ---------------------------
# Plotting
# ---------------------------

fig = go.Figure()

if dim == 1:
    x,y,z = build_line()
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode='lines'))

elif dim == 2:
    pts = build_square()
    x,y = zip(*pts)
    z = [0]*len(x)
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode='markers'))

elif dim == 3:
    pts = build_cube()
    x,y,z = zip(*pts)
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode='markers'))

elif dim == 4:
    pts = build_tesseract_projection()
    x,y,z = zip(*pts)
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode='markers'))

elif dim == 5:
    pts = build_5d_projection()
    x,y,z = zip(*pts)
    fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode='markers'))

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Concept Explanation
# ---------------------------

if dim == 4:
    st.info("4D: You are seeing a 3D projection of a tesseract (cube evolving across a new axis).")

if dim == 5:
    st.info("5D: This represents multiple tesseract states layered together — a higher-order structure of transformations.")
