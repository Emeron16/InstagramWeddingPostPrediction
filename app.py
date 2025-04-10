#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os
import io
import base64
from tensorflow.keras.models import load_model
import getpass
import threading
from flasgger import Swagger
from flask_cors import CORS
from datetime import datetime
from tensorflow.keras.models import load_model
import datetime as dt
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[2]:


os.environ["FLASK_DEBUG"] = "development"

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)


# In[3]:


model = load_model('hashtag_classification_Model_97ACC.keras')


# In[4]:


def preprocess_and_predict(new_data, model, structured_columns, caption_tokenizer, hashtag_tokenizer, maxlen_caption=50, maxlen_hashtag=3):
    """
    Preprocesses new input data and predicts using the trained model.

    Parameters:
    - new_data (DataFrame): The new data to predict on.
    - model (keras.Model): The trained multi-input classification model.
    - structured_columns (list): List of columns for the structured input.
    - caption_tokenizer (Tokenizer): Tokenizer for caption text.
    - hashtag_tokenizer (Tokenizer): Tokenizer for hashtags.
    - maxlen_caption (int): Maximum sequence length for captions.
    - maxlen_hashtag (int): Maximum sequence length for hashtags.

    Returns:
    - predictions (numpy array): The predicted class probabilities for the new data.
    - Class bins are [-1, 100, 600, float('inf')]
    """

    # ---- Step 1: Structured Data Preprocessing ----
    # Ensure all necessary columns are present
    structured_data = new_data.drop(['caption', 'hashtags/0', 'hashtags/1', 'hashtags/2'], axis=1)
    structured_data = pd.get_dummies(structured_data, columns=['productType', 'type', 'day_of_week', 'season', 'month_of_year'])
    
    # Reorder the columns to match the training data
    for col in structured_columns:
        if col not in structured_data:
            structured_data[col] = 0  # Add missing columns as zeros (for one-hot encoding consistency)
    structured_data = structured_data[structured_columns]

    # Convert to NumPy array
    X_structured_new = structured_data.values.astype('float32')

    # ---- Step 2: Caption Text Preprocessing ----
    new_captions = new_data['caption'].astype(str).tolist()
    caption_sequences = caption_tokenizer.texts_to_sequences(new_captions)
    X_text_new = pad_sequences(caption_sequences, maxlen=maxlen_caption).astype('int32')

    # ---- Step 3: Hashtag Preprocessing ----
    # Combine hashtag columns into one string per sample
    hashtags_combined_new = new_data[['hashtags/0', 'hashtags/1', 'hashtags/2']].astype(str).agg(" ".join, axis=1)
    hashtag_sequences_new = hashtag_tokenizer.texts_to_sequences(hashtags_combined_new)
    X_hashtag_new = pad_sequences(hashtag_sequences_new, maxlen=maxlen_hashtag).astype('int32')

    # ---- Step 4: Model Prediction ----
    predictions = model.predict([X_structured_new, X_text_new, X_hashtag_new])
    predicted_class = np.argmax(predictions, axis=1)[0]  
    category_mapping = {0: 'Low', 1: 'Medium', 2: 'High'} 

    return category_mapping[predicted_class]


# In[5]:


# Load caption_tokenizer
with open("caption_tokenizer2.pkl", "rb") as caption_file:
    caption_tokenizer = pickle.load(caption_file)

# Load hashtag_tokenizer
with open("hashtag_tokenizer2.pkl", "rb") as hashtag_file:
    hashtag_tokenizer = pickle.load(hashtag_file)

# Load Scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)



@app.route('/', methods=['GET'])
def home():
        return render_template('post_prediction.html', caption='',
                               hashtags=[], formatted_date='',
                               productType='',
                               recommend_recommendation_dates=False,
                               type='', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    caption = request.form['caption']
    recommend_recommendation_dates = request.form.get('recommend_recommendation_dates', False)
    hashtags = [
        request.form.get('hashtags0', '').strip(),
        request.form.get('hashtags1', '').strip(),
        request.form.get('hashtags2', '').strip()
    ]

    # Filter out empty hashtags
    hashtags = [hashtag for hashtag in hashtags if hashtag]

    # Format the date to "January 1, 2025"
    post_date = request.form['post_date']
    date = dt.datetime.strptime(post_date, '%Y-%m-%d')
    formatted_date = date.strftime("%B %d, %Y")

    # Extract day of the week, month, and season
    day_of_week = date.strftime('%A')
    month_of_year = date.month
    season = get_season(month_of_year)

    form_data = {
        'caption': caption,
        'hashtags/0': request.form['hashtags0'],
        'hashtags/1': request.form['hashtags1'],
        'hashtags/2': request.form['hashtags2'],
        'productType': request.form['productType'],
        'type': request.form['type'],
        'day_of_week': day_of_week,
        'season': season,
        'month_of_year': month_of_year
    }

    structured_columns = ['productType_carousel_container', 'productType_clips',
       'productType_feed', 'productType_igtv', 'type_Image', 'type_Sidecar',
       'type_Video', 'day_of_week_Friday', 'day_of_week_Monday',
       'day_of_week_Saturday', 'day_of_week_Sunday', 'day_of_week_Thursday',
       'day_of_week_Tuesday', 'day_of_week_Wednesday', 'season_Fall',
       'season_Spring', 'season_Summer', 'season_Winter', 'month_of_year_1',
       'month_of_year_2', 'month_of_year_3', 'month_of_year_4',
       'month_of_year_5', 'month_of_year_6', 'month_of_year_7',
       'month_of_year_8', 'month_of_year_9', 'month_of_year_10',
       'month_of_year_11', 'month_of_year_12']

    new_df = pd.DataFrame([form_data])

    # Call the predict_popularity_category function
    prediction = preprocess_and_predict(new_df, model, structured_columns, caption_tokenizer, hashtag_tokenizer)
    
    recommended_dates = []
    if recommend_recommendation_dates:
        if prediction == "Low":
            # Step 3: Predict for the next 14 calendar days
            for day_offset in range(1, 15):
                future_date = date + dt.timedelta(days=day_offset)
                future_form_data = form_data.copy()
                future_form_data['day_of_week'] = future_date.strftime('%A')
                future_form_data['month_of_year'] = future_date.month
                future_form_data['season'] = get_season(future_form_data['month_of_year'])

                # Create DataFrame for future date
                future_df = pd.DataFrame([future_form_data])

                # Step 4: Predict for the future date
                future_prediction = preprocess_and_predict(
                    future_df, model, structured_columns, caption_tokenizer, hashtag_tokenizer
                )

                # Step 5: Collect future date if prediction is "High"
                if future_prediction in ["Medium", "High"]:
                    recommended_dates.append(future_date.strftime("%B %d, %Y"))

        if prediction in ["Medium", "High"]:
            # Step 3: Predict for the next 14 calendar days
            for day_offset in range(1, 15):
                future_date = date + dt.timedelta(days=day_offset)
                future_form_data = form_data.copy()
                future_form_data['day_of_week'] = future_date.strftime('%A')
                future_form_data['month_of_year'] = future_date.month
                future_form_data['season'] = get_season(future_form_data['month_of_year'])

                # Create DataFrame for future date
                future_df = pd.DataFrame([future_form_data])

                # Step 4: Predict for the future date
                future_prediction = preprocess_and_predict(
                    future_df, model, structured_columns, caption_tokenizer, hashtag_tokenizer
                )

                # Step 5: Collect future date if prediction is "High"
                if future_prediction == "High":
                    recommended_dates.append(future_date.strftime("%B %d, %Y"))


    # Render the form with the prediction
    return render_template(
        'post_prediction.html',
        prediction=prediction,
        caption=caption,
        hashtags=hashtags,
        formatted_date=formatted_date,
        recommended_dates=recommended_dates,
        recommend_recommendation_dates=recommend_recommendation_dates,
        productType=request.form['productType'],
        type=request.form['type']
    )


def get_season(month):
    """Returns the season based on the month of the year."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))






