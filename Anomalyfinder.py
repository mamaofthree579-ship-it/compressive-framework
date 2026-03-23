import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def generate_data():
    Generates a simulated dataset of archaeological residue analysis.
    # --- Generate "Normal" Data (e.g., cooking pots) ---
    num_normal = 200
    normal_data = {
        'artifact_id': [f'N-{i:03}' for i in range(num_normal)],
        'interpretation': ['Cooking'] * num_normal,
        # Normal pots have varied ingredients
        'organic_diversity_score': np.random.uniform(0.6, 1.0, num_normal),
        # Residue is mostly at the bottom
        'residue_uniformity_score': np.random.uniform(0.1, 0.4, num_normal),
        # Salt levels are low
        'salt_concentration_ratio': np.random.uniform(0.01, 0.05, num_normal),
        # No vitrification
        'vitrification_index': np.random.uniform(0.0, 0.1, num_normal)
    }

    # --- Generate "Anomalous" Data (e.g., process chambers) ---
    num_anomalous = 10
    anomalous_data = {
        'artifact_id': [f'A-{i:03}' for i in range(num_anomalous)],
        'interpretation': ['Unknown'] * num_anomalous,
        # Very pure, specific substance
        'organic_diversity_score': np.random.uniform(0.0, 0.2, num_anomalous),
        # Uniform coating
        'residue_uniformity_score': np.random.uniform(0.8, 1.0, num_anomalous),
        # High salt for chemical process
        'salt_concentration_ratio': np.random.uniform(0.2, 0.5, num_anomalous),
        # High heat has vitrified the surface
        'vitrification_index': np.random.uniform(0.7, 1.0, num_anomalous)
    }

    # Combine into pandas DataFrames
    df_normal = pd.DataFrame(normal_data)
    df_anomalous = pd.DataFrame(anomalous_data)

    # Return the "normal" training set and the full mixed dataset for testing
    return df_normal, pd.concat([df_normal, df_anomalous]).sample(frac=1).reset_index(drop=True)

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("🧠 Archaeological Anomaly Detector")
st.write("A prototype for identifying misclassified artifacts using machine learning, based on our conceptual framework.")

# --- Generate and Display Data ---
df_train_normal, df_full_dataset = generate_data()

st.header("Simulated Dataset")
st.write(f"Generated a dataset of {len(df_full_dataset)} artifacts. Most are 'normal' cooking pots, but a few have anomalous signatures.")
st.dataframe(df_full_dataset.head(10))

# --- Model Training and Prediction ---
st.header("Step 1: Train the Model")
st.write("We train an `IsolationForest` model. This model learns the statistical profile of 'normal' artifacts from the training data.")

features = ['organic_diversity_score', 'residue_uniformity_score', 'salt_concentration_ratio', 'vitrification_index']
X_train = df_train_normal[features]

# The "contamination" parameter tells the model the expected percentage of anomalies in the training data.
# Since we are training ONLY on normal data, we set it to a very small value.
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_train)

st.success("Model training complete.")

st.header("Step 2: Detect Anomalies")
st.write("Now, we use the trained model to score every artifact in the full dataset. A score of -1 indicates an anomaly.")

X_full = df_full_dataset[features]
predictions = model.predict(X_full)
scores = model.decision_function(X_full)

# Add results to the DataFrame
df_results = df_full_dataset.copy()
df_results['model_prediction'] = predictions
df_results['anomaly_score'] = scores
df_results['is_anomaly'] = df_results['model_prediction'] == -1

st.header("Results: Flagged Anomalies")
st.write("The model identified the following artifacts as being statistically different from the 'normal' cooking pots.")

# --- Display Results ---
anomalies_df = df_results[df_results['is_anomaly']].sort_values(by='anomaly_score')

# Add color formatting
def highlight_anomalies(s):
    return ['background-color: #f2a0a1' if v == 'Unknown' else '' for v in s]

st.dataframe(anomalies_df.style.apply(highlight_anomalies, subset=['interpretation']))

st.write("Notice how the flagged artifacts (originally labeled 'Unknown') have low diversity, high uniformity, high salt, and high vitrification—the exact signature we designed.")
