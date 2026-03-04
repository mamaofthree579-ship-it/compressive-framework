import streamlit as st
import numpy as np
import pandas as pd

st.header("Gut intuition ↔ HRV entropy demo")
n = st.slider("Sample size", 20, 100, 40)
noise = st.slider("Noise level", 0.01, 0.2, 0.08)

gut = np.random.uniform(1,7,n)
entropy = 0.9 - 0.1*gut + np.random.normal(0,noise,n)
df = pd.DataFrame({'gut':gut,'entropy':entropy})
r = df.gut.corr(df.entropy)
st.write("Correlation r:", round(r,3))
st.scatter_chart(df, x='gut', y='entropy')
