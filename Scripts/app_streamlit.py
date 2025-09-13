import streamlit as st
import joblib
import numpy as np
import os

# Define feature names and descriptions
feature_names = ['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness',
                 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
                 'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness', 'SE Compactness', 'SE Concavity',
                 'SE Concave Points', 'SE Symmetry', 'SE Fractal Dimension', 'Worst Radius', 'Worst Texture', 'Worst Perimeter',
                 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry',
                 'Worst Fractal Dimension']

# Get the directory of this script
current_dir = os.path.dirname(__file__)
scaler_path = os.path.join(current_dir, 'scaler.pkl')
model_path = os.path.join(current_dir, 'model.pkl')

def main():
    st.title('Breast Cancer Prediction App by Dr. Lee')
    st.write("Enter the values for the features to get a prediction.")

    # Load sample data to pre-load default values
    sample_data = joblib.load(scaler_path).inverse_transform(np.array([[0]*30]))  # Placeholder for the actual sample data

    # Define input fields for user to enter feature values with proper labels
    features = []
    for i, feature in enumerate(feature_names):
        feature_value = st.number_input(f'{feature}', min_value=0.0, value=float(sample_data[0, i]))
        features.append(feature_value)

    # Handle cases where input features may need scaling
    def preprocess_input(features):
        scaler = joblib.load(scaler_path)
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        return features_scaled

    # Button to make a prediction
    if st.button('Predict'):
        try:
            features_scaled = preprocess_input(features)
            model = joblib.load(model_path)
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0, 1]

            # Display the result
            st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
            st.write(f"Probability of Positive: {probability:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    main()
