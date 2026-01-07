import pandas as pd
import numpy as np
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

from preprocess import clean_text

# load data

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "problems.jsonl")

df = pd.read_json(data_path, lines=True)

df['full_text'] = (
    df['title'].astype(str) + " " +
    df['description'].astype(str) + " " +
    df['input_description'].astype(str) + " " +
    df['output_description'].astype(str)
)

df['clean_text'] = df['full_text'].apply(clean_text)

# extra features

df['text_length'] = df['clean_text'].apply(lambda x: len(x.split()))

MATH_SYMBOLS_REGEX = r'[=+\-*/%^<>]'
df['math_symbol_count'] = df['full_text'].apply(
    lambda x: len(re.findall(MATH_SYMBOLS_REGEX, x))
)

KEYWORDS = [
    'graph','tree','dp','dynamic','recursion',
    'greedy','binary','search','sort',
    'array','string','matrix'
]

df['keyword_count'] = df['full_text'].apply(
    lambda x: sum(x.lower().count(k) for k in KEYWORDS)
)

# TF-IDF

vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1,2),
    stop_words='english'
)

X_text = vectorizer.fit_transform(df['clean_text'])

X = hstack([
    X_text,
    df[['text_length','math_symbol_count','keyword_count']].values
])

# targets

label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(df['problem_class'])
y_score = df['problem_score']

# split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

# classification

print("\nClassification Models")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest": RandomForestClassifier(
        n_estimators=600, max_depth=50, class_weight='balanced'
    ),
    "SVM": LinearSVC()
}

best_acc = 0
best_clf = None
best_clf_name = ""

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_clf = model
        best_clf_name = name

print(f"\nBest Classifier: {best_clf_name} ({best_acc:.4f})")
print("Confusion Matrix:\n", confusion_matrix(y_test, best_clf.predict(X_test)))

# regression

print("\nRegression Models")

regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=400),
    "Gradient Boosting": GradientBoostingRegressor()
}

best_rmse = float("inf")
best_reg = None
best_reg_name = ""

for name, model in regressors.items():
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_test_r)

    rmse = np.sqrt(mean_squared_error(y_test_r, preds))
    mae = mean_absolute_error(y_test_r, preds)

    print(f"{name} → RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_reg = model
        best_reg_name = name

print(f"\n Best Regressor: {best_reg_name} (RMSE: {best_rmse:.4f})")

# best models

joblib.dump(best_clf, "models/classifier.pkl")
joblib.dump(best_reg, "models/regressor.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")



