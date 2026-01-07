# AutoJudge: Predicting Programming Problem Difficulty

# Project Overview
AutoJudge is an intelligent system that automatically predicts the difficulty of programming problems using only their textual descriptions.  
The project performs two tasks:

1. # Classification – Predicts the difficulty class (Easy / Medium / Hard)
2. # Regression – Predicts a numerical difficulty score

The goal is to reduce reliance on manual judgment when assigning difficulty levels on competitive programming platforms.


# Dataset Used
The project uses a provided dataset containing programming problems and their difficulty annotations.

Each data sample includes:
- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numerical value)

The dataset is stored locally in: data/problems.jsonl


# Total samples: 4112

# Approach and Models Used

# 1. Text Preprocessing
- Lowercasing
- Removal of special characters
- Stopword removal using NLTK
- Lemmatization using SpaCy
- Merging all text fields into a single text input

# 2. Feature Engineering
- TF-IDF vectorization of cleaned text
- Text length
- Count of mathematical symbols
- Frequency of important keywords (e.g., graph, dp, recursion, search)

# 3. Models
# Classification Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

# Regression Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The bestperforming models were selected and saved for use in the web interface.

# Evaluation Metrics

# Classification Results
The following models were evaluated for difficulty classification:

- Logistic Regression: Accuracy = 0.53
- Support Vector Machine (SVM): Accuracy = 0.50
- Random Forest Classifier: Accuracy = **0.56**

The **Random Forest Classifier** achieved the best performance with an accuracy of **56.38%**.

**Confusion Matrix (Random Forest):**
[[ 50 66 20]
[ 24 372 29]
[ 17 203 42]]


### Regression Results
The following models were evaluated for difficulty score prediction:

- Linear Regression: RMSE = 2.45, MAE = 1.98
- Random Forest Regressor: RMSE = **2.03**, MAE = **1.69**
- Gradient Boosting Regressor: RMSE = 2.05, MAE = 1.72

The **Random Forest Regressor** achieved the lowest error and was selected as the final regression model.


## Web Interface Explanation
A simple Streamlit-based web interface is provided where users can:

1. Paste a programming problem description
2. Click the Predict button
3. View:
   - Predicted difficulty class
   - Predicted difficulty score

The interface runs locally and does not require authentication or a database.

## Dependencies

The project uses the following Python libraries:

import pandas as pd
import numpy as np
import os
import json
import re
import joblib
import streamlit as st
import nltk
import spacy
from nltk.corpus import stopwords
from scipy.sparse import hstack

Make sure these dependencies are installed before running the project.

# Project Structure

AutoJudge_OpenProject/
├── app/
│   └── AutoJudgeapp.py
├── data/
│   └── problems.jsonl
├── models/
│   ├── classifier.pkl
│   ├── label_encoder.pkl
│   ├── regressor.pkl
│   └── vectorizer.pkl
├── src/
│   ├── preprocess.py
│   └── train.py
├── load_data.py
└── README.md

# Step 1: Load and Inspect the Dataset
python load_data.py

# Step 2: Run Data Preprocessing
python src/preprocess.py

# Step 3: Train the Models
python src/train.py

# Step 4: Run the Web Application
streamlit run app/AutoJudgeapp.py




# Name: Thannuj Gorla
# Institution: IIT Roorkee


