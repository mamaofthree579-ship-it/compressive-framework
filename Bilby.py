import streamlit as st, numpy as np, bilby, tempfile
st.title("Toy GW Inference (no lalsuite)")

duration=4; fs=2048; N=int(duration*fs)
t=np.arange(N)/fs; f0=30
def sine_waveform(p):
    return p['A']*np.sin(2*np.pi*f0*t+ p['phi'])*np.exp(-0.5*((t-2)/0.5)**2)
class ToyLik(bilby.Likelihood):
    def __init__(self):
        super().__init__(dict(A=1,phi=0))
        self.d=np.zeros(N)
    def log_likelihood(self):
        r=self.d-sine_waveform(self.parameters)
        return -0.5*np.sum(r**2)

like=ToyLik()
priors=bilby.core.prior.PriorDict(dict(A=bilby.core.prior.Uniform(0,2),
                                      phi=bilby.core.prior.Uniform(0,np.pi)))

if st.button("Run"):
    with tempfile.TemporaryDirectory() as out:
        res=bilby.run_sampler(like,priors,sampler='dynesty',nlive=200,
                              outdir=out,label='toy',verbose=False)
        st.write(f"Samples: {len(res.samples)}")
        st.pyplot(res.plot_corner(['A','phi']))
        st.success("Done")
