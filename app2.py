import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from datetime import datetime
from flasgger import Swagger
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
Swagger(app)

# Load model and supporting files
MODEL_PATH = os.path.join(os.getcwd(), 'hashtag_classification_Model_97ACC.keras')
TOKENIZER_CAPTION_PATH = os.path.join(os.getcwd(), 'caption_tokenizer2.pkl')
TOKENIZER_HASHTAG_PATH = os.path.join(os.getcwd(), 'hashtag_tokenizer2.pkl')
SCALER_PATH = os.path.join(os.getcwd(), 'scaler.pkl')

try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_CAPTION_PATH, 'rb') as f:
        caption_tokenizer = pickle.load(f)
    with open(TOKENIZER_HASHTAG_PATH, 'rb') as f:
        hashtag_tokenizer = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("[INFO] Model and tokenizers loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model or tokenizers: {e}")
    exit(1)  # Exit if loading fails

# Helper function for preprocessing
def preprocess_input(caption, hashtags):
    caption_seq = caption_tokenizer.texts_to_sequences([caption])
    caption_padded = tf.keras.preprocessing.sequence.pad_sequences(caption_seq, maxlen=30)
    hashtags_seq = hashtag_tokenizer.texts_to_sequences([hashtags])
    hashtags_padded = tf.keras.preprocessing.sequence.pad_sequences(hashtags_seq, maxlen=10)
    return caption_padded, hashtags_padded

# Prediction function
def predict_best_posting_time(caption, hashtags, engagement_rate, day_of_week):
    caption_padded, hashtags_padded = preprocess_input(caption, hashtags)
    features = np.array([[engagement_rate, day_of_week]])
    features_scaled = scaler.transform(features)
    input_data = [caption_padded, hashtags_padded, features_scaled]
    prediction = model.predict(input_data)
    recommended_hour = int(round(prediction[0][0]))
    return recommended_hour

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        caption = request.form.get('caption', '')
        hashtags = request.form.get('hashtags', '')
        engagement_rate = float(request.form.get('engagement_rate', 0))
        day_of_week = int(request.form.get('day_of_week', 0))
        recommended_hour = predict_best_posting_time(caption, hashtags, engagement_rate, day_of_week)
        result = {
            "recommended_posting_hour": recommended_hour,
            "status": "success"
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
