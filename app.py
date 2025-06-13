import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Load model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function to get sentiment and confidence
def analyze_sentiment(texts):
    results = sentiment_pipeline(texts, truncation=True)
    sentiments, scores = [], []
    for r in results:
        sentiments.append(r['label'].lower())
        scores.append(round(r['score'], 3))
    return sentiments, scores

# Function to extract keywords using TF-IDF
def extract_keywords(texts, top_n=3):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        keywords = []
        for row in X:
            scores = row.toarray().flatten()
            top_indices = scores.argsort()[-top_n:][::-1]
            top_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            keywords.append(", ".join(top_keywords))
        return keywords
    except Exception as e:
        return [""] * len(texts)

# Streamlit UI
st.title("🧠 Sentiment Analysis Dashboard")
st.markdown("Upload one or two `.txt` files. Each line will be treated as a separate text entry.")

# File uploads
file1 = st.file_uploader("Upload File A (.txt)", type=["txt"])
file2 = st.file_uploader("Upload File B (.txt) - Optional", type=["txt"])

def process_file(uploaded_file):
    raw_text = uploaded_file.read().decode("utf-8").splitlines()
    raw_text = [line.strip() for line in raw_text if line.strip()]
    sentiments, scores = analyze_sentiment(raw_text)
    keywords = extract_keywords(raw_text)
    df = pd.DataFrame({
        "text": raw_text,
        "sentiment": sentiments,
        "confidence": scores,
        "keywords": keywords
    })
    return df

# Process and display files
if file1:
    st.subheader("📂 File A Results")
    df1 = process_file(file1)
    st.dataframe(df1)

if file2:
    st.subheader("📂 File B Results")
    df2 = process_file(file2)
    st.dataframe(df2)

    # Comparative Sentiment Distribution
    st.subheader("📊 Comparative Sentiment Distribution")
    sentiment_counts = pd.DataFrame({
        "File A": df1["sentiment"].value_counts(),
        "File B": df2["sentiment"].value_counts()
    }).fillna(0)
    st.bar_chart(sentiment_counts)

# Export options
if file1:
    st.subheader("📁 Export Results")
    file_format = st.selectbox("Choose format", ["CSV", "JSON"])
    if file_format == "CSV":
        st.download_button("Download File A Results", df1.to_csv(index=False), file_name="fileA_sentiments.csv", mime="text/csv")
        if file2:
            st.download_button("Download File B Results", df2.to_csv(index=False), file_name="fileB_sentiments.csv", mime="text/csv")
    elif file_format == "JSON":
        st.download_button("Download File A Results", df1.to_json(orient="records"), file_name="fileA_sentiments.json", mime="application/json")
        if file2:
            st.download_button("Download File B Results", df2.to_json(orient="records"), file_name="fileB_sentiments.json", mime="application/json")
