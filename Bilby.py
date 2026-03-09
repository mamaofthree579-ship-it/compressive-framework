import streamlit as st
import bilby, os, tempfile

os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP Toy Explorer")

m1_lo, m1_hi = st.sidebar.slider("mass_1 prior", 25.0,45.0,(30.0,38.0))
m2_lo, m2_hi = st.sidebar.slider("mass_2 prior", 25.0,45.0,(28.0,36.0))
a_lo, a_hi = st.sidebar.slider("α* prior", 0.01,0.2,(0.05,0.12))
l_lo, l_hi = st.sidebar.slider("λ prior", 0.3,0.7,(0.4,0.6))

class ToyLike(bilby.core.likelihood.Likelihood):
    def __init__(self):
        super().__init__(parameters={'mass_1':None,'mass_2':None,
                                     'alpha':None,'lam':None})
    def log_likelihood(self):
        m1,m2,a,l = [self.parameters[k] for k in ('mass_1','mass_2','alpha','lam')]
        return -(((m1-33.6)/(1.2*a))**2 + ((m2-32.2)/(0.8*l))**2)

if st.button("Run"):
    with tempfile.TemporaryDirectory() as outdir:
        priors = bilby.core.prior.PriorDict({
            'mass_1': bilby.core.prior.Uniform(m1_lo,m1_hi),
            'mass_2': bilby.core.prior.Uniform(m2_lo,m2_hi),
            'alpha': bilby.core.prior.Uniform(a_lo,a_hi),
            'lam': bilby.core.prior.Uniform(l_lo,l_hi)
        })
        res = bilby.run_sampler(likelihood=ToyLike(), priors=priors,
                                sampler='dynesty', nlive=200,
                                outdir=outdir, label='ui', verbose=False)
        st.pyplot(res.plot_corner())
    st.success("Done!")
