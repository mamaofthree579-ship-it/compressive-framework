import streamlit as st
import bilby, os, tempfile
from bilby.gw import likelihood
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.detector import InterferometerList

os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP GW250114 Analysis (PSD noise)")

alpha = st.sidebar.slider("α*", 0.0, 0.2, 0.08, 0.005)
lam = st.sidebar.slider("λ", 0.3, 0.7, 0.5, 0.01)

gps = 1420878141.2
duration = 4

if st.button("Run Bilby"):
    with tempfile.TemporaryDirectory() as outdir:
        ifos = InterferometerList(['H1','L1'])
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=2048, duration=duration, start_time=gps-2)

        class CGUPWaveform(WaveformGenerator):
    def __init__(self, alpha, lam):
        super().__init__(
            duration=duration,
            sampling_frequency=2048,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments={'waveform_approximant':'pSEOBNRv5PHM'}
        )
        self.alpha, self.lam = alpha, lam
            def frequency_domain_strain(self, params):
                h = super().frequency_domain_strain(params)
                for i in range(len(h)):
                    h[i] *= 1 + (self.alpha**2)*(self.lam**i)
                return h

        wf = CGUPWaveform(alpha, lam)
        like = likelihood.GravitationalWaveTransient(interferometers=ifos,
                                                     waveform_generator=wf)

        priors = bilby.core.prior.PriorDict({
            'mass_1': bilby.core.prior.Uniform(20,80),
            'mass_2': bilby.core.prior.Uniform(20,80),
            'alpha': bilby.core.prior.DeltaFunction(alpha),
            'lam': bilby.core.prior.DeltaFunction(lam)
        })

        st.write("Running Bilby on synthetic PSD noise…")
        res = bilby.run_sampler(likelihood=like, priors=priors,
                                sampler='dynesty', nlive=200,
                                outdir=outdir, label='cgup_gw', verbose=False)
        st.pyplot(res.plot_corner(['mass_1','mass_2']))
    st.success("Done!")
