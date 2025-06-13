import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Download NLTK stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Title
st.title("💬 Sentiment Analysis Dashboard")

# Select input type
input_type = st.radio("Choose input type:", ["Single Text", "Multiple Texts", "Upload File"])

# Get input
texts = []

if input_type == "Single Text":
    user_input = st.text_area("Enter a sentence:")
    if user_input:
        texts = [user_input]

elif input_type == "Multiple Texts":
    user_input = st.text_area("Enter multiple sentences (one per line):")
    if user_input:
        texts = user_input.strip().split('\n')

elif input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        texts = content.strip().splitlines()

# Analyze
if texts and st.button("Analyze Sentiment"):
    results = []
    all_labels = []
    for text in texts:
        if text.strip() == "":
            continue
        result = classifier(text)[0]
        label = result["label"].lower()
        score = round(result["score"], 4)

        # Convert Hugging Face label to "neutral" if confidence is low
        if label in ["positive", "negative"] and score < 0.6:
            label = "neutral"

        results.append({
            "Text": text,
            "Sentiment": label.capitalize(),
            "Confidence": score
        })
        all_labels.append(label)

    df = pd.DataFrame(results)

    # Show table
    st.subheader("🔍 Sentiment Results")
    st.dataframe(df, use_container_width=True)

    # Show chart
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot.pie(autopct="%1.1f%%", startangle=90, ax=ax)
    ax.set_ylabel("")
    ax.set_title("Sentiment Share")
    st.pyplot(fig)

    # Keyword Extraction
    st.subheader("🗝️ Top Keywords (TF-IDF)")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(df["Text"])
    keywords = vectorizer.get_feature_names_out()
    st.write(", ".join(keywords))

    # Export Options
    st.subheader("📁 Export Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

    json = df.to_json(orient="records", force_ascii=False)
    st.download_button("Download JSON", json, "sentiment_results.json", "application/json")
