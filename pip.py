import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import string

# Load trained model
model = tf.keras.models.load_model("sentiment_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Padding length (same as used during training)
MAX_LEN = 200

# Prediction function
def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]

    # Threshold logic (adjustable)
    if prediction >= 0.6:
        return "Positive 😊"
    elif prediction <= 0.4:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.markdown("Enter a movie review below and the model will predict if it's *Positive, **Negative, or **Neutral*.")

user_input = st.text_area("Your Review", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_sentiment(user_input)
            st.success(f"Sentiment: *{result}*")
