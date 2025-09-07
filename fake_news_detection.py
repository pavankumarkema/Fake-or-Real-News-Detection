import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("news_dataset.csv") # columns: ['title', 'text', 'label']
print(df.columns)  # Check what columns you actually have

# Suppose columns are ['text', 'label'] only:
df['content'] = df['text']


# Clean data
df = df[['content', 'label']].dropna()
df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})

# Feature Extraction
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['content'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Streamlit interface
st.title('Fake News Detector')
st.write(f"Test accuracy: {acc:.2f}")

input_text = st.text_area('Enter news content for prediction:')
if st.button('Predict'):
    input_vec = tfidf.transform([input_text])
    prediction = model.predict(input_vec)
    result = 'REAL' if prediction == 1 else 'FAKE'

    st.write(f'This news seems to be {result}.')
