# Add this to your Streamlit logic to test the 89.89% Moon value
target_cohesion = 89.89
drift_rate = (97.9 - cohesion_score) / 100 # Higher gap = higher drift

st.metric("Lunar Cohesion", f"{cohesion_score:.2f}%", 
          delta=f"{cohesion_score - 89.89:.2f}% from Observed Moon")

if cohesion_score < 90.0:
    st.warning("🌙 LUNAR SLIPPAGE: The orbit is elliptical and slowly expanding.")
