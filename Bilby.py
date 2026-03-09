import streamlit as st, numpy as np, bilby, tempfile, matplotlib.pyplot as plt
st.title("Toy GW + CGUP Visual Demo")

duration=4; fs=2048; N=int(duration*fs)
t=np.arange(N)/fs; f0=30
def sine_waveform(p):
    base=p['A']*np.sin(2*np.pi*f0*t+ p['phi'])*np.exp(-0.5*((t-2)/0.5)**2)
    alpha,lam=p.get('alpha',0.08),p.get('lam',0.5)
    mod=1 + (alpha**2)*lam*np.sin(2*np.pi*5*t)
    return base*mod

class ToyLik(bilby.Likelihood):
    def __init__(self):
        super().__init__(dict(A=1,phi=0,alpha=0.08,lam=0.5))
        self.d=np.zeros(N)
    def log_likelihood(self):
        r=self.d-sine_waveform(self.parameters)
        return -0.5*np.sum(r**2)

like=ToyLik()
priors=bilby.core.prior.PriorDict(dict(
    A=bilby.core.prior.Uniform(0,2),
    phi=bilby.core.prior.Uniform(0,np.pi),
    alpha=bilby.core.prior.DeltaFunction(0.08),
    lam=bilby.core.prior.DeltaFunction(0.5)
))

if st.button("Run"):
    with tempfile.TemporaryDirectory() as out:
        res=bilby.run_sampler(like,priors,sampler='dynesty',nlive=200,
                              outdir=out,label='demo',verbose=False)
        st.write(f"Samples: {len(res.posterior)}")
        st.pyplot(res.plot_corner(['A','phi']))
        p=res.posterior.iloc[0].to_dict()
        w_mod=sine_waveform(p)
        p0=p.copy(); p0['alpha']=0
        w_plain=sine_waveform(p0)
        fig,ax=plt.subplots()
        ax.plot(t,w_plain,label='GR')
        ax.plot(t,w_mod,label='CGUP')
        ax.set_xlabel('Time'); ax.legend()
        st.pyplot(fig)
        st.success("Demo complete!")
