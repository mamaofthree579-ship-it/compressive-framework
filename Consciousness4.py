import streamlit as st 
import numpy as np
import pandas as pd
from math import atanh, sqrt

def fisher_z(r): return atanh(r)
def z_test(r1,n1,r2,n2):
    z1,z2 = fisher_z(r1), fisher_z(r2)
    se = sqrt(1/(n1-3)+1/(n2-3))
    return (z1-z2)/se

n = st.slider("Sample size per group", 10, 60, 30)
noise = st.slider("Noise level", 0.01, 0.2, 0.08)

modes = ["voice","visual","both"]
data=[]
for m in modes:
    gut = np.random.uniform(1,7,n)
    entropy = 0.9 - 0.1*gut + np.random.normal(0,noise,n)
    if m!="both": entropy += np.random.normal(0.3,0.05,n)
    data.append(pd.DataFrame({'mode':m,'gut':gut,'entropy':entropy}))
df=pd.concat(data)

rs={}
for m in modes:
    sub=df[df['mode']==m]
    r=sub['gut'].corr(sub['entropy'])
    rs[m]=r
    st.subheader(f"{m} (r={r:.2f})")
    st.scatter_chart(sub, x='gut', y='entropy')

z_voice = z_test(rs['both'],n,rs['voice'],n)
z_vis = z_test(rs['both'],n,rs['visual'],n)
st.write("Fisher z (both vs voice):", round(z_voice,2))
st.write("Fisher z (both vs visual):", round(z_vis,2))
