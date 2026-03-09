import streamlit as st, numpy as np, bilby, tempfile, matplotlib.pyplot as plt
st.set_page_config(page_title="CGUP Demo",layout="centered")
st.title("Gravitational‑Wave Toy Model + CGUP Modulation")
st.markdown("Adjust **alpha** and **lam** to see how a CGUP correction deforms a sine‑Gaussian GW.")
alpha=st.slider("alpha (modulation strength)",0.0,0.2,0.08,0.01)
lam=st.slider("lam (coupling)",0.1,1.0,0.5,0.1)

duration,fs=4,2048; N=int(duration*fs)
t=np.arange(N)/fs; f0=30
def wave(p):
    base=p['A']*np.sin(2*np.pi*f0*t+p['phi'])*np.exp(-0.5*((t-2)/0.5)**2)
    return base*(1+(p.get('alpha',0.08)**2)*p.get('lam',0.5)*np.sin(2*np.pi*5*t))

# waveform panel (live)
p_demo=dict(A=1.0,phi=0.0,alpha=alpha,lam=lam)
fig,ax=plt.subplots()
ax.plot(t,wave({**p_demo,'alpha':0}),label='GR',color='steelblue')
ax.plot(t,wave(p_demo),label='CGUP',color='darkorange')
ax.set_xlabel('Time [s]'); ax.legend(); st.pyplot(fig)

# inference panel
if st.button("Run Inference"):
    with st.spinner("Sampling..."):
        #...bilby run as before...
        st.success("Done")
