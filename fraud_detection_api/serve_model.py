# serve_model.py
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
import logging
import numpy as np

app = Flask(__name__)

# Load models
models = {
    "fraud_model": joblib.load('models/fraud/random_forest_model.joblib'),
    # Add other models if needed
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    logger.info(f"Received request: {data}")

    # Preprocess data if needed
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = models["fraud_model"].predict(features)[0]

    response = {'prediction': int(prediction)}
    logger.info(f"Prediction: {response}")

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
