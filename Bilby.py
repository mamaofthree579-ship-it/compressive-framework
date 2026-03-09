import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import bilby, os, tempfile

# silence Bilby’s metadata reminder
os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP Ladder + Bilby (full)")

# --- sidebar controls ---
alpha = st.sidebar.slider("α*", 0.0, 0.2, 0.08, 0.005)
lam   = st.sidebar.slider("λ",   0.3, 0.7, 0.5, 0.01)
beta  = 0.4
omega_GR = 0.3737

def cgup_freq(alpha, lam, n):
    return omega_GR + (alpha**2)*omega_GR*(lam**n + beta*lam**(n+1))

# ladder plot
freqs = [cgup_freq(alpha, lam, n) for n in range(5)]
fig, ax = plt.subplots()
for f in freqs:
    ax.axvline(f, color='tab:blue', lw=2)
ax.axvspan(omega_GR-0.06, omega_GR+0.06, color='gray', alpha=0.2,
           label='GR 90% bound')
ax.set_xlabel("Frequency (arb.)")
ax.set_yticks([]); ax.legend()
st.pyplot(fig)

# --- Bilby test ---
if st.button("Run Bilby toy fit"):
    st.write("Running Bilby (toy Gaussian likelihood)…")
    data_freq = omega_GR + np.random.normal(0, 0.01)

    def loglike(params):
        model = cgup_freq(params['alpha'], params['lam'], 0)
        return -0.5*((data_freq - model)/0.01)**2

    priors = bilby.core.prior.PriorDict({
        'alpha': bilby.core.prior.Uniform(0, 0.2),
        'lam':   bilby.core.prior.Uniform(0.3, 0.7)
    })

    likelihood = bilby.likelihood.Likelihood()
likelihood.log_likelihood = loglike

res = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler='dynesty',
    nlive=200,
    outdir=d,
    label='cgup_demo',
    verbose=False
)
    st.pyplot(res.plot_corner())
    st.success("Done!")

# note
st.caption("Replace the toy loglike with Bilby’s GravitationalWaveTransient "
           "to use real GW data; the slider values feed α* and λ directly.")
