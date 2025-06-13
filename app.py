import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import torch
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF
import io
import tempfile
import json

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load sentiment model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("📊 Sentiment Analysis Dashboard")

st.markdown("""
This app analyzes sentiment from text (positive, neutral, negative), shows confidence scores, highlights keywords,
and allows you to upload, type, or paste text.
""")

input_mode = st.radio("Choose Input Method:", ("Single Text", "Multiple Texts", "Upload CSV"))

texts = []
if input_mode == "Single Text":
    text_input = st.text_area("Enter your text here")
    if text_input:
        texts = [text_input]
elif input_mode == "Multiple Texts":
    multi_input = st.text_area("Enter multiple texts (one per line)")
    if multi_input:
        texts = [line.strip() for line in multi_input.split('\n') if line.strip()]
else:
    uploaded_file = st.file_uploader("Upload CSV file with a 'text' column", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            texts = df['text'].dropna().astype(str).tolist()
        else:
            st.error("CSV must contain a 'text' column")

if texts:
    with st.spinner("Analyzing..."):
        results = []
        for text in texts:
            try:
                sentiment = sentiment_pipeline(text)[0]
                label = sentiment['label'].capitalize()
                score = round(sentiment['score'], 4)

                # Adjust to 3-class manually
                if score < 0.5:
                    label = "Neutral"

                # Extract keywords
                words = [word for word in text.split() if word.lower() not in stop_words]
                vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
                tfidf_matrix = vectorizer.fit_transform([text])
                keywords = vectorizer.get_feature_names_out()

                results.append({
                    "Text": text,
                    "Sentiment": label,
                    "Confidence": score,
                    "Keywords": ", ".join(keywords)
                })
            except Exception as e:
                results.append({
                    "Text": text,
                    "Sentiment": "Error",
                    "Confidence": 0.0,
                    "Keywords": str(e)
                })

        results_df = pd.DataFrame(results)
        st.subheader("Sentiment Results")
        st.dataframe(results_df, use_container_width=True)

        # Visualization
        sentiment_counts = results_df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "gray", "red"])
        ax.set_title("Sentiment Distribution")
        ax.set_ylabel("Number of Texts")
        st.pyplot(fig)

        # Export options
        st.subheader("Export Results")
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_data, file_name="sentiment_results.csv", mime="text/csv")

        # JSON Export
        json_data = results_df.to_json(orient="records", lines=True).encode("utf-8")
        st.download_button("Download JSON", json_data, file_name="sentiment_results.json", mime="application/json")

        # PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align='C')
        for idx, row in results_df.iterrows():
            pdf.multi_cell(0, 10, txt=f"Text: {row['Text']}\nSentiment: {row['Sentiment']}\nConfidence: {row['Confidence']}\nKeywords: {row['Keywords']}\n---")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            with open(tmp_pdf.name, "rb") as f:
                st.download_button("Download PDF", f.read(), file_name="sentiment_report.pdf", mime="application/pdf")
