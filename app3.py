import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import json
import importlib.util
from logistic_regression import run as logistic_run  # Import Logistic Regression function

# Load models
def load_models():
    try:
        return {
            "Logistic Regression": "Custom Logistic Regression",  # Placeholder (Handled separately)
            "XGBoost": joblib.load("xgboost.pkl"),
            "Decision Tree": joblib.load("decision_tree.pkl"),
            "Random Forest": joblib.load("random_forest.pkl"),
            "Artificial Neural Network": tf.keras.models.load_model("ann_model.h5")
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

models = load_models()

# Streamlit UI
st.title("Machine Learning Model Selector")

model_choice = st.selectbox("Choose a model:", list(models.keys()))

# File uploader for input_data.py
uploaded_file = st.file_uploader("Upload your Python file (input_data.py)", type="py")

if uploaded_file is not None:
    try:
        # Save uploaded file
        with open("uploaded_input.py", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Dynamically import uploaded input_data.py
        spec = importlib.util.spec_from_file_location("input_data", "uploaded_input.py")
        input_data_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(input_data_module)

        # Extract JSON data
        input_json = input_data_module.input_data
        input_values = json.loads(input_json)["data"]

        st.write("### Extracted Input Data:")
        st.json(input_values)

        if st.button("Predict"):
            if model_choice == "Logistic Regression":
                result = logistic_run(input_json)  # Call logistic_regression.py
                st.json(json.loads(result))  # Display the result
            else:
                model = models.get(model_choice, None)
                if model and hasattr(model, "predict"):
                    prediction = model.predict(np.array(input_values))
                    st.write("### Prediction:")
                    st.write(prediction.tolist())
                else:
                    st.error("The selected model is not a valid ML model.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
