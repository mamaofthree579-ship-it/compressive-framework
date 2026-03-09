import streamlit as st
import bilby, os, tempfile
import numpy as np

os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP Toy Inference (α*, λ sampled)")

class CGUPToyLike(bilby.core.likelihood.Likelihood):
    def __init__(self):
        super().__init__(parameters={'mass_1':None,'mass_2':None,
                                     'alpha':None,'lam':None})
    def log_likelihood(self):
        m1,m2,a,l = [self.parameters[k] for k in ('mass_1','mass_2','alpha','lam')]
        # toy CGUP penalty: pull toward GW250114 values with α*,λ modulation
        return -(((m1-33.6)/(1.2*a))**2 + ((m2-32.2)/(0.8*l))**2)

if st.button("Run CGUP Toy Sampler"):
    with tempfile.TemporaryDirectory() as outdir:
        like = CGUPToyLike()
        priors = bilby.core.prior.PriorDict({
            'mass_1': bilby.core.prior.Uniform(30,38),
            'mass_2': bilby.core.prior.Uniform(28,36),
            'alpha': bilby.core.prior.Uniform(0.05,0.12),
            'lam': bilby.core.prior.Uniform(0.4,0.6)
        })
        res = bilby.run_sampler(likelihood=like, priors=priors,
                                sampler='dynesty', nlive=300,
                                outdir=outdir, label='cgup_toy', verbose=False)
        st.pyplot(res.plot_corner())
    st.success("Done!")
