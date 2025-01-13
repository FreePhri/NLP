import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import torch

# Load your sentiment analysis model and vectorizer
with open("sentiment_model.pkl", "rb") as sentiment_model_file:
    sentiment_model = pickle.load(sentiment_model_file)

with open("sentiment_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the Hugging Face model for rating prediction
rating_model_name = "MahmoudMohamed/Amazon_rating_review_model"
rating_tokenizer = AutoTokenizer.from_pretrained(rating_model_name)
rating_model = AutoModelForSequenceClassification.from_pretrained(rating_model_name)

# Streamlit app
st.title("Review Analysis Application")
st.markdown("""
### Features:
1. **Sentiment Analysis**: Predict if a review is positive, neutral, or negative (using your custom model).
2. **Rating Prediction**: Predict the star rating (1–5) for a review using a Hugging Face model.
""")

# Feature selection
feature = st.radio("Select a feature:", ["Sentiment Analysis", "Rating Prediction"])

# Input review text
review_text = st.text_area("Enter your review:")

# Predict button
if st.button("Analyze Review"):
    if review_text.strip():
        if feature == "Sentiment Analysis":
            # Use your custom sentiment analysis model
            review_vectorized = vectorizer.transform([review_text])
            predicted_sentiment = sentiment_model.predict(review_vectorized)[0]
            st.subheader("Predicted Sentiment:")
            st.write(predicted_sentiment.capitalize())

        elif feature == "Rating Prediction":
            # Use Hugging Face model for rating prediction
            inputs = rating_tokenizer(review_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = rating_model(**inputs)
            predicted_rating = torch.argmax(outputs.logits, dim=1).item() + 1  # Assuming ratings are 0-indexed
            st.subheader("Predicted Rating:")
            st.write(f"⭐ {predicted_rating}")
    else:
        st.error("Please enter a review!")

# Footer
st.markdown("""
---
**Review Analysis Application** | Built with Streamlit
""")
