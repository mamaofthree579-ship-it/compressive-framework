import streamlit as st, numpy as np, bilby, tempfile, matplotlib.pyplot as plt

st.set_page_config(page_title="CGUP Demo",layout="centered")
st.title("Gravitational‑Wave + CGUP Modulation")
st.markdown("Alpha/lam sliders deform a **real LAL waveform** with a CGUP correction.")

alpha=st.slider("alpha",0.0,0.2,0.08,0.01)
lam=st.slider("lam",0.1,1.0,0.5,0.1)

# load real waveform (must match duration,fs)
h_lal=np.sin(2*np.pi*30*t)*np.exp(-0.5*((t-2)/0.5)**2)  # fallback
duration,fs=4,2048; N=int(duration*fs)
t=np.arange(N)/fs

def wave(p):
    mod=1+(p.get('alpha',0)**2)*p.get('lam',0.5)*np.sin(2*np.pi*5*t)
    return h_lal*mod

# live plot
fig,ax=plt.subplots()
ax.plot(t,h_lal,label='GR',color='steelblue')
ax.plot(t,wave(dict(alpha=alpha,lam=lam)),label='CGUP',color='darkorange')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Strain'); ax.legend()
st.pyplot(fig)

class ToyLik(bilby.Likelihood):
    def __init__(self):
        super().__init__(dict(alpha=0.08,lam=0.5)); self.d=np.zeros(N)
    def log_likelihood(self):
        r=self.d-wave(self.parameters); return -0.5*np.sum(r**2)

like=ToyLik()
priors=bilby.core.prior.PriorDict(dict(
    alpha=bilby.core.prior.DeltaFunction(0.08),
    lam=bilby.core.prior.DeltaFunction(0.5)
))

if st.button("Run Inference"):
    with st.spinner("Sampling..."):
        with tempfile.TemporaryDirectory() as out:
            res=bilby.run_sampler(like,priors,sampler='dynesty',nlive=200,
                                  outdir=out,label='demo',verbose=False)
            st.pyplot(res.plot_corner(['alpha','lam']))
    st.success("Inference complete")
