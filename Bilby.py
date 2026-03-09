import streamlit as st, bilby, os, tempfile
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator

os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP Debug Run")

gps=1420878141.2; duration=4
try:
    from gwpy.timeseries import TimeSeries
    h1=TimeSeries.fetch_open_data('H1',gps-2,gps+2,cache=True)
    l1=TimeSeries.fetch_open_data('L1',gps-2,gps+2,cache=True)
    ifos=InterferometerList(['H1','L1'])
    ifos[0].strain_data.set_from_gwpy_timeseries(h1)
    ifos[1].strain_data.set_from_gwpy_timeseries(l1)
    st.write("Fetched real strain.")
except Exception as e:
    st.warning(f"Fetch failed ({e}), using PSD.")
    ifos=InterferometerList(['H1','L1'])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=2048,duration=duration,start_time=gps-2)

class CGUPWG(WaveformGenerator):
    def __init__(self):
        super().__init__(duration=duration,sampling_frequency=2048,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={'waveform_approximant':'IMRPhenomPv2','reference_frequency':50})
    def frequency_domain_strain(self,params):
    h=super().frequency_domain_strain(params)
    # a,l=params.get('alpha',0.08),params.get('lam',0.5)
    # return {k:h[k]*(1+(a**2)*(l**i)) for i,k in enumerate(h)}
    return h # <-- plain GR waveform

wf=CGUPWG()
like=bilby.gw.likelihood.GravitationalWaveTransient(ifos,wf)

priors=bilby.core.prior.PriorDict({
    'mass_1':bilby.core.prior.Uniform(30,38),
    'mass_2':bilby.core.prior.Uniform(28,36),
    'alpha':bilby.core.prior.DeltaFunction(0.08),
    'lam':bilby.core.prior.DeltaFunction(0.5),
    'theta_jn':bilby.core.prior.Uniform(0,3.14),
    'phase':bilby.core.prior.Uniform(0,3.14),
    'geocent_time':bilby.core.prior.Uniform(gps-0.1,gps+0.1),
    'psi':bilby.core.prior.Uniform(0,3.14),
    'ra':bilby.core.prior.Uniform(0,6.28),
    'dec':bilby.core.prior.Uniform(-1.57,1.57),
    'a_1':bilby.core.prior.DeltaFunction(0.0),
    'a_2':bilby.core.prior.DeltaFunction(0.0),
    'tilt_1':bilby.core.prior.DeltaFunction(0.0),
    'tilt_2':bilby.core.prior.DeltaFunction(0.0),
    'phi_12':bilby.core.prior.DeltaFunction(0.0),
    'phi_jl':bilby.core.prior.DeltaFunction(0.0),
    'luminosity_distance':bilby.core.prior.Uniform(100,1000)
})

if st.button("Run"):
    with tempfile.TemporaryDirectory() as outdir:
        try:
            res=bilby.run_sampler(likelihood=like,priors=priors,
                                  sampler='dynesty',nlive=200,
                                  outdir=outdir,label='debug',verbose=False)
            st.write(f"Samples: {len(res.samples)}")
            fig=res.plot_corner(['mass_1','mass_2'])
            st.pyplot(fig)
            st.success("Done!")
        except Exception as e:
            st.error(f"Sampler crashed: {e}")
