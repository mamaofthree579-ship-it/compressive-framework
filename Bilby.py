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

import numpy as np
# placeholder waveform: a simple sine wave
h = np.sin(np.linspace(0, 2*np.pi, 100))
np.save("h_placeholder.npy", h.real)
# load it back to verify
loaded = np.load("h_placeholder.npy")
print(loaded[:5])
