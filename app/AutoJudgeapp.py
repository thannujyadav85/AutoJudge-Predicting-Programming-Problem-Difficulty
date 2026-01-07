import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
import re
import nltk
import spacy
from nltk.corpus import stopwords

# text cleaning

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.text not in stop_words and len(token.text) > 2:
            tokens.append(token.lemma_)
    return " ".join(tokens)

# extra features(match training)

MATH_SYMBOLS_REGEX = r'[=+\-*/%^<>]'

KEYWORDS = [
    'graph', 'tree', 'dp', 'dynamic', 'recursion',
    'greedy', 'binary', 'search', 'sort',
    'array', 'string', 'matrix'
]

def count_math_symbols(text):
    return len(re.findall(MATH_SYMBOLS_REGEX, text))

def keyword_frequency(text):
    text = text.lower()
    return sum(text.count(k) for k in KEYWORDS)

# load models

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# streamlit UI

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("AutoJudge")
st.write("Predicts programming problem difficulty using text only")

st.subheader("Enter problem details")

problem_desc = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

# prediction

if st.button("Predict Difficulty"):
    if not problem_desc or not input_desc or not output_desc:
        st.warning("Please fill all fields")
    else:
        # Combine text EXACTLY like training
        full_text = problem_desc + " " + input_desc + " " + output_desc

        # Clean text
        cleaned_text = clean_text(full_text)

        # TF-IDF
        X_text = vectorizer.transform([cleaned_text])

        # Numeric features (ORDER MUST MATCH TRAINING)
        text_length = len(cleaned_text.split())
        math_symbol_count = count_math_symbols(full_text)
        keyword_count = keyword_frequency(full_text)

        X_numeric = np.array([[text_length, math_symbol_count, keyword_count]])

        # Final feature vector
        X = hstack([X_text, X_numeric])

        # Predictions
        pred_class_encoded = clf.predict(X)[0]
        pred_class = label_encoder.inverse_transform([pred_class_encoded])[0]
        pred_score = reg.predict(X)[0]

        # Output
        st.success(f"Predicted Difficulty Class: {pred_class}")
        st.success(f"Estimated Difficulty Score: {pred_score:.2f}")
