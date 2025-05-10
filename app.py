import streamlit as st
import joblib
import re
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stop_words = set(stopwords.words('arabic'))

# Text preprocessing function
def clean_text(text):
    text = re.sub(r"[^\u0600-\u06FF]", " ", str(text))  # Keep only Arabic
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def preprocess(text):
    cleaned = clean_text(text)
    no_stopwords = remove_stopwords(cleaned)
    return no_stopwords

# Streamlit UI
st.title("Arabic Sentiment Analysis")
user_input = st.text_area("أدخل تعليقًا باللغة العربية:")

if st.button("تحليل"):
    if user_input.strip() == "":
        st.warning("يرجى إدخال تعليق.")
    else:
        processed = preprocess(user_input)
        features = vectorizer.transform([processed]).toarray()
        prediction = model.predict(features)[0]
        
        st.subheader("النتيجة:")
        if prediction == 'positive':
            st.success("👍 تعليق إيجابي")
        elif prediction == 'negative':
            st.error("👎 تعليق سلبي")
        else:
            st.info("😐 تعليق محايد")
