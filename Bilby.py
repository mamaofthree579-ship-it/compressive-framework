import streamlit as st
import bilby, os, tempfile

os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP GW250114 Analysis (working baseline)")

alpha = st.sidebar.slider("α*", 0.0, 0.2, 0.08, 0.005)
lam = st.sidebar.slider("λ", 0.3, 0.7, 0.5, 0.01)

class ToyLike(bilby.core.likelihood.Likelihood):
    def __init__(self):
        super().__init__(parameters={'mass_1':None, 'mass_2':None})
    def log_likelihood(self):
        m1, m2 = self.parameters['mass_1'], self.parameters['mass_2']
        return -(((m1-33.6)/1.2)**2 + ((m2-32.2)/0.8)**2)

if st.button("Run Toy Bilby"):
    with tempfile.TemporaryDirectory() as outdir:
        like = ToyLike()
        priors = bilby.core.prior.PriorDict({
            'mass_1': bilby.core.prior.Uniform(30,38),
            'mass_2': bilby.core.prior.Uniform(28,36)
        })
        st.write("Sampling…")
        res = bilby.run_sampler(likelihood=like, priors=priors,
                                sampler='dynesty', nlive=200,
                                outdir=outdir, label='toy', verbose=False)
        st.pyplot(res.plot_corner())
    st.success("Done!")
