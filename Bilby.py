import streamlit as st
import bilby, os, tempfile
from bilby.gw import detector, likelihood
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw import utils

data = utils.get_event_data('GW250114', interferometers=['H1','L1'],
                             duration=4, sampling_frequency=2048)

os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP GW250114 Analysis")

# sliders for CGUP params
alpha = st.sidebar.slider("α*", 0.0, 0.2, 0.08, 0.005)
lam = st.sidebar.slider("λ", 0.3, 0.7, 0.5, 0.01)
beta = 0.4
omega_GR = 0.3737

# 1. download real data
if st.button("Load GW250114 data"):
    st.write("Fetching H1/L1 strain from GWOSC…")
    data = detector.get_event_data('GW250114', interferometers=['H1','L1'],
                                   duration=4, sampling_frequency=2048)
    st.success("Data loaded")

    # 2. CGUP waveform wrapper
    class CGUPWaveform(waveform.WaveformGenerator):
        def __init__(self, alpha, lam):
            super().__init__(
                duration=4, sampling_frequency=2048,
                waveform_function=waveform.get_td_waveform,
                waveform_arguments={'waveform_approximant':'pSEOBNRv5PHM'}
            )
            self.alpha, self.lam = alpha, lam
        def frequency_domain_strain(self, params):
            h = super().frequency_domain_strain(params)
            # toy shift: scale each mode by CGUP factor
            for i in range(len(h)):
                h[i] *= 1 + (self.alpha**2)*(self.lam**i)
            return h

    wf = CGUPWaveform(alpha, lam)
    like = likelihood.GravitationalWaveTransient(interferometers=data,
                                                  waveform_generator=wf)

    priors = bilby.core.prior.PriorDict({
        'mass_1': bilby.core.prior.Uniform(20, 80),
        'mass_2': bilby.core.prior.Uniform(20, 80),
        'alpha': bilby.core.prior.DeltaFunction(alpha),
        'lam': bilby.core.prior.DeltaFunction(lam),
        #... other needed params fixed or given priors...
    })

    st.write("Running Bilby on GW250114 (this will take minutes)…")
    with tempfile.TemporaryDirectory() as d:
        res = bilby.run_sampler(
            likelihood=like,
            priors=priors,
            sampler='dynesty',
            nlive=400,
            outdir=d,
            label='cgup_gw250114',
            verbose=False
        )
        st.pyplot(res.plot_corner(['mass_1','mass_2']))
    st.success("Done!")
