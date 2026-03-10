import streamlit as st, numpy as np, bilby, tempfile, matplotlib.pyplot as plt
st.set_page_config(page_title="CGUP Demo",layout="centered")
st.title("Gravitational‑Wave + CGUP Modulation")
st.markdown("Alpha/lam sliders deform a real or fallback waveform with CGUP.")

alpha=st.slider("alpha",0.0,0.2,0.08,0.01)
lam=st.slider("lam",0.1,1.0,0.5,0.1)

duration,fs=4,2048; N=int(duration*fs)
t=np.arange(N)/fs
try:
    h_lal=np.load("h_lal.npy",allow_pickle=True)
except FileNotFoundError:
    h_lal=np.sin(2*np.pi*30*t)*np.exp(-0.5*((t-2)/0.5)**2) # fallback sine-Gaussian

def wave(alpha,lam):
    mod=1+alpha**2*lam*np.sin(2*np.pi*5*t)
    return h_lal*mod

fig,ax=plt.subplots()
ax.plot(t,h_lal,label='GR',color='steelblue')
ax.plot(t,wave(alpha,lam),label='CGUP',color='darkorange')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Strain'); ax.legend()
st.pyplot(fig)

class ToyLik(bilby.Likelihood):
    def __init__(self):
        super().__init__(dict(alpha=0.08,lam=0.5)); self.d=np.zeros(N)
    def log_likelihood(self):
        r=self.d-wave(self.parameters['alpha'],self.parameters['lam'])
        return -0.5*np.sum(r**2)

like=ToyLik()
priors=bilby.core.prior.PriorDict(dict(
    alpha=bilby.core.prior.Uniform(0,0.2),
    lam=bilby.core.prior.Uniform(0.1,1.0)
))

if st.button("Run Inference"):
    with st.spinner("Sampling..."):
        with tempfile.TemporaryDirectory() as out:
            res=bilby.run_sampler(like,priors,sampler='dynesty',nlive=200,
                                  outdir=out,label='demo',verbose=False)
            st.pyplot(res.plot_corner(['alpha','lam']))
    st.success("Inference complete")
