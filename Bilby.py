import streamlit as st, numpy as np, bilby, tempfile, matplotlib.pyplot as plt
st.set_page_config(page_title="CGUP Demo",layout="centered")
st.title("Gravitational‑Wave Toy Model + CGUP Modulation")
st.markdown("Adjust alpha and lam to see how a CGUP correction deforms a sine‑Gaussian GW.")
alpha=st.slider("alpha",0.0,0.2,0.08,0.01)
lam=st.slider("lam",0.1,1.0,0.5,0.1)

duration,fs=4,2048; N=int(duration*fs)
t=np.arange(N)/fs; f0=30
def wave(p):
    base=p['A']*np.sin(2*np.pi*f0*t+p['phi'])*np.exp(-0.5*((t-2)/0.5)**2)
    mod=1+(p.get('alpha',0.08)**2)*p.get('lam',0.5)*np.sin(2*np.pi*5*t)
    return base*mod

# live waveform
p_demo=dict(A=1.0,phi=0.0,alpha=alpha,lam=lam)
fig,ax=plt.subplots()
ax.plot(t,wave({**p_demo,'alpha':0}),label='GR',color='steelblue')
ax.plot(t,wave(p_demo),label='CGUP',color='darkorange')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Strain'); ax.legend()
st.pyplot(fig)

class ToyLik(bilby.Likelihood):
    def __init__(self):
        super().__init__(dict(A=1,phi=0,alpha=0.08,lam=0.5)); self.d=np.zeros(N)
    def log_likelihood(self):
        r=self.d-wave(self.parameters); return -0.5*np.sum(r**2)

like=ToyLik()
priors=bilby.core.prior.PriorDict(dict(A=bilby.core.prior.Uniform(0,2),
    phi=bilby.core.prior.Uniform(0,np.pi),alpha=bilby.core.prior.DeltaFunction(0.08),
    lam=bilby.core.prior.DeltaFunction(0.5)))

if st.button("Run Inference"):
    with st.spinner("Sampling..."):
        with tempfile.TemporaryDirectory() as out:
            res=bilby.run_sampler(like,priors,sampler='dynesty',nlive=200,
                                  outdir=out,label='demo',verbose=False)
            st.pyplot(res.plot_corner(['A','phi']))
    st.success("Inference complete")

import streamlit as st
import numpy as np
from lal import SimInspiralFD # Ensure lalsuite is installed in your environment
import os

st.title("LALSimulation Waveform Generator")

# Cache data to prevent re-calculating on every rerun
@st.cache_data
def generate_waveform():
    # --- YOUR CODE ---
    # (Assuming h and h_lal.npy generation logic here)
    # Example placeholder:
    h_real = np.random.randn(1000) # Replace with h.real from lal
    np.save('h_lal.npy', h_real)
    return 'h_lal.npy'

if st.button("Generate Waveform"):
    with st.spinner("Generating..."):
        file_path = generate_waveform()
        st.success(f"Saved: {file_path}")
        
        # Optionally display the data
        data = np.load(file_path)
        st.line_chart(data)

# Create an in-memory buffer to store the file
buffer = io.BytesIO()
# Example data
h_real = np.random.randn(1000) 
np.save(buffer, h_real)
buffer.seek(0)
