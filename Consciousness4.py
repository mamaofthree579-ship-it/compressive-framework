import streamlit as st
import numpy as np

st.sidebar.header("Controls")
brain_mode = st.sidebar.selectbox("Brain mode", ["voice", "visual", "both"])
emotion = st.sidebar.selectbox("Emotion", ["calm", "fear", "joy"])
K = st.sidebar.slider("Coupling K", 0.1, 3.0, 0.8, 0.1)
gut_drive = st.sidebar.slider("Gut drive", 0.0, 1.0, 0.2, 0.05)

dt = 0.01
t = np.arange(0, 60, dt)
f_gut, f_heart = 0.05, 1.2
f_brain_voice, f_brain_visual = 5.0, 10.0

def heart_freq(em):
    return f_heart + {'calm':0, 'fear':0.3, 'joy':0.1}[em]

fb = [f_brain_voice] if brain_mode=='voice' else [f_brain_visual] if brain_mode=='visual' else [f_brain_voice, f_brain_visual]

phases = np.zeros((2+len(fb), len(t)))
for i in range(1, len(t)):
    drive = gut_drive*np.sin(2*np.pi*0.2*t[i]) if 19<t[i]<21 else 0
    phases[0,i] = phases[0,i-1] + dt*(2*np.pi*f_gut + K*np.sin(phases[1,i-1]-phases[0,i-1]) + drive)
    phases[1,i] = phases[1,i-1] + dt*(2*np.pi*heart_freq(emotion) + K*np.sin(phases[0,i-1]-phases[1,i-1]))
    for j, fbj in enumerate(fb):
        phases[2+j,i] = phases[2+j,i-1] + dt*(2*np.pi*fbj + K*np.sin(phases[1,i-1]-phases[2+j,i-1]))

for j in range(len(fb)):
    diff = np.angle(np.exp(1j*(phases[1]-phases[2+j])))
    win = int(3/heart_freq(emotion)/dt)
    locked = any(np.all(np.abs(diff[k:k+win])<0.5) for k in range(len(diff)-win))
    st.write(f'brain {j} locked: {locked}')
