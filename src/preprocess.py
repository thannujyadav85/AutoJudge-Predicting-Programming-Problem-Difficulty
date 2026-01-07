import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import spacy

# setup

# download only once
nltk.download('stopwords')

# load SpaCy model
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))

# text cleaning

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove special characters

    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.text not in stop_words and len(token.text) > 2:
            tokens.append(token.lemma_)   # lemmatisation

    return " ".join(tokens)

# extra features

MATH_SYMBOLS_REGEX = r'[=+\-*/%^<>]'

KEYWORDS = [
    'graph', 'tree', 'dp', 'dynamic', 'recursion',
    'greedy', 'binary', 'search', 'sort',
    'array', 'string', 'matrix'
]

def text_length(text):
    return len(str(text).split())

def count_math_symbols(text):
    return len(re.findall(MATH_SYMBOLS_REGEX, str(text)))

def keyword_frequency(text):
    text = str(text).lower()
    return sum(text.count(k) for k in KEYWORDS)

# loading data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "problems.jsonl")

df = pd.read_json(data_path, lines=True)

# merging text fields

df['full_text'] = (
    df['title'].astype(str) + " " +
    df['description'].astype(str) + " " +
    df['input_description'].astype(str) + " " +
    df['output_description'].astype(str)
)

# applying preprocessing

df['clean_text'] = df['full_text'].apply(clean_text)

# Extra features from hints
df['text_length'] = df['full_text'].apply(text_length)
df['math_symbol_count'] = df['full_text'].apply(count_math_symbols)
df['keyword_count'] = df['full_text'].apply(keyword_frequency)

# debug output

print(
    df[
        [
            'clean_text',
            'text_length',
            'math_symbol_count',
            'keyword_count',
            'problem_class',
            'problem_score'
        ]
    ].head()
)

print("Total samples:", len(df))


