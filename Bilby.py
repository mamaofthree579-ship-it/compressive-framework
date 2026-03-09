import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import bilby
import json, os, tempfile

st.title("CGUP Ladder + Bilby demo")

# --- sliders ---
alpha = st.sidebar.slider("α*", 0.0, 0.2, 0.08, 0.005)
lam = st.sidebar.slider("λ", 0.3, 0.7, 0.5, 0.01)
beta = 0.4
omega_GR = 0.3737

def cgup_freq(alpha, lam, n):
    return omega_GR + (alpha**2)*omega_GR*(lam**n + beta*lam**(n+1))

# ladder plot (same as before)
freqs = [cgup_freq(alpha, lam, n) for n in range(5)]
fig, ax = plt.subplots()
for f in freqs:
    ax.axvline(f, color='tab:blue')
ax.axvspan(omega_GR-0.06, omega_GR+0.06, color='gray', alpha=0.2)
ax.set_xlabel('Frequency'); ax.set_yticks([])
st.pyplot(fig)

# --- Bilby part ---
if st.button("Run tiny Bilby test"):
    st.write("Setting up Bilby... (this may take ~30s)")

    # fake data: GR frequency with tiny noise
    np.random.seed(42)
    data_freq = omega_GR + np.random.normal(0, 0.01)

    # likelihood: Gaussian around CGUP freq for n=0
    def loglike(params):
        model = cgup_freq(params['alpha'], params['lam'], 0)
        return -0.5*((data_freq - model)/0.01)**2

    priors = bilby.core.prior.PriorDict({
        'alpha': bilby.core.prior.Uniform(0, 0.2),
        'lam':   bilby.core.prior.Uniform(0.3, 0.7)
    })

    # run sampler in a temp dir
    with tempfile.TemporaryDirectory() as d:
        res = bilby.run_sampler(
            likelihood=bilby.likelihood.Likelihood(),
            priors=priors,
            log_likelihood=loglike,
            sampler='dynesty',
            nlive=200,
            outdir=d,
            label='cgup_test'
        )
        # show corner plot
        st.pyplot(res.plot_corner())
    
    st.success("Bilby run complete!")
