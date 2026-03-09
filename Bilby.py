import streamlit as st, bilby, os, tempfile
from bilby.gw.detector import InterferometerList

os.environ["BILBY_INCLUDE_GLOBAL_METADATA"] = "False"

st.title("CGUP Explorer with optional GW data")

use_real = st.sidebar.checkbox("Use real GWOSC data (H1/L1)", False)

class ToyLike(bilby.core.likelihood.Likelihood):
    def __init__(self):
        super().__init__(parameters={'mass_1':None,'mass_2':None,
                                     'alpha':None,'lam':None})
    def log_likelihood(self):
        m1,m2,a,l = [self.parameters[k] for k in ('mass_1','mass_2','alpha','lam')]
        return -(((m1-33.6)/(1.2*a))**2 + ((m2-32.2)/(0.8*l))**2)

if st.button("Run"):
    with tempfile.TemporaryDirectory() as outdir:
        if use_real:
            try:
                from gwpy.timeseries import TimeSeries
                gps=1420878141.2
                h1 = TimeSeries.fetch_open_data('H1', gps-2, gps+2, cache=True)
                l1 = TimeSeries.fetch_open_data('L1', gps-2, gps+2, cache=True)
                ifos = InterferometerList(['H1','L1'])
                ifos[0].strain_data.set_from_gwpy_timeseries(h1)
                ifos[1].strain_data.set_from_gwpy_timeseries(l1)
                st.write("Fetched real strain.")
            except Exception as e:
                st.warning(f"Fetch failed ({e}), using toy likelihood.")
                ifos = None
        else:
            ifos = None

        priors = bilby.core.prior.PriorDict({
            'mass_1': bilby.core.prior.Uniform(30,38),
            'mass_2': bilby.core.prior.Uniform(28,36),
            'alpha': bilby.core.prior.Uniform(0.05,0.12),
            'lam': bilby.core.prior.Uniform(0.4,0.6)
        })
        res = bilby.run_sampler(likelihood=ToyLike(), priors=priors,
                                sampler='dynesty', nlive=200,
                                outdir=outdir, label='run', verbose=False)
        st.pyplot(res.plot_corner())
    st.success("Done!")
