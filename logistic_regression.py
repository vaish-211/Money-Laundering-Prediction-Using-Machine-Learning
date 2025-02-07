import pickle
import json
import numpy as np

# Load the model parameters
with open("logistic_model.pkl", "rb") as f:
    model_params = pickle.load(f)

coefficients = np.array(model_params["coefficients"])  # Convert list back to NumPy array

def run(data):
    try:
        input_data = np.array(json.loads(data)["data"])  # Convert to NumPy array

        # Add a constant term (intercept) to input data (for bias)
        input_data_with_intercept = np.c_[np.ones((input_data.shape[0], 1)), input_data]  # Prepend column of 1s

        # Compute logit (linear function)
        logit = np.dot(input_data_with_intercept, coefficients)

        # Apply sigmoid function to get probability
        probability = 1 / (1 + np.exp(-logit))

        # Convert to class (threshold = 0.5)
        prediction = (probability >= 0.5).astype(int)

        return json.dumps({"probability": probability.tolist(), "prediction": prediction.tolist()})
    
    except Exception as e:
        return json.dumps({"error": str(e)})
