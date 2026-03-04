import streamlit as st
import numpy as np
import pandas as pd

st.header("Gut intuition ↔ HRV entropy by inner mode")
n = st.slider("Sample size per group", 10, 60, 30)
noise = st.slider("Noise level", 0.01, 0.2, 0.08)

modes = ["voice","visual","both"]
data = []
for m in modes:
    gut = np.random.uniform(1,7,n)
    entropy = 0.9 - 0.1*gut + np.random.normal(0,noise,n)
    if m != "both":
        entropy += np.random.normal(0.3,0.05,n)  # weaker link
    data.append(pd.DataFrame({'mode':m,'gut':gut,'entropy':entropy}))
df = pd.concat(data)

for m in modes:
    sub = df[df.mode==m]
    r = sub.gut.corr(sub.entropy)
    st.subheader(f"{m} (r={r:.2f})")
    st.scatter_chart(sub, x='gut', y='entropy')
