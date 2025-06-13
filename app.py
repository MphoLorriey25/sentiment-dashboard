import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF
import io
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLTK stopwords once
nltk.download('stopwords')

# ----- Page config and style -----
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f4f8;
    }
    .stButton>button {
        background-color: #007ACC;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 25px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005A9E;
        cursor: pointer;
    }
    .reportview-container .markdown-text-container {
        font-size: 18px;
        color: #333;
    }
    .stTextArea textarea, .stTextInput input {
        border-radius: 8px;
        border: 1.5px solid #007ACC;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 Sentiment Analysis Dashboard")
st.markdown("Analyze the sentiment of texts (positive, neutral, negative) with confidence scores, keyword extraction, and export options.")

# Load Hugging Face sentiment model for multi-class sentiment analysis
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
    return nlp

sentiment_analyzer = load_model()

# NLTK stopwords for keyword extraction
stop_words = set(stopwords.words('english'))

# Function to classify with neutral detection
def classify_text(text):
    results = sentiment_analyzer(text)[0]  # List of dicts with label and score
    # Convert to dict {label: score}
    scores = {res['label']: res['score'] for res in results}
    # Threshold for neutral: if max confidence < 0.6, consider neutral
    max_label = max(scores, key=scores.get)
    max_score = scores[max_label]
    if max_score < 0.6:
        sentiment = "neutral"
        confidence = max_score
    else:
        # Map model labels to simple positive/negative/neutral
        if max_label.lower() == 'positive':
            sentiment = "positive"
        elif max_label.lower() == 'negative':
            sentiment = "negative"
        else:
            sentiment = "neutral"
        confidence = max_score
    return sentiment, confidence

# Keyword extraction using TF-IDF on input texts
def extract_keywords(texts, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    keywords_scores = [(word, sums[0, idx]) for idx, word in enumerate(feature_names)]
    keywords_scores.sort(key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in keywords_scores]

# PDF Export helper
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Sentiment Analysis Results', ln=True, align='C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf(df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Add each row
    for idx, row in df.iterrows():
        pdf.cell(0, 10, f"Text: {row['Text']}", ln=True)
        pdf.cell(0, 10, f"Sentiment: {row['Sentiment']} (Confidence: {row['Confidence']:.2f})", ln=True)
        pdf.cell(0, 10, f"Keywords: {row['Keywords']}", ln=True)
        pdf.ln(5)
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output

# --- Sidebar: Input selection ---
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Select Input Mode", ["Single Text", "Multiple Texts (paste)", "Upload Text File (.txt)"])

texts = []

if input_mode == "Single Text":
    single_text = st.text_area("Enter your text here", height=150)
    if single_text:
        texts = [single_text]

elif input_mode == "Multiple Texts (paste)":
    multi_texts = st.text_area("Enter multiple texts separated by new lines", height=250)
    if multi_texts:
        texts = [line.strip() for line in multi_texts.strip().split("\n") if line.strip()]

elif input_mode == "Upload Text File (.txt)":
    uploaded_file = st.file_uploader("Upload a plain text file (.txt)", type=["txt"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        texts = [line.strip() for line in content.strip().split("\n") if line.strip()]

if texts:
    # Process texts batch
    results = []
    for txt in texts:
        sentiment, confidence = classify_text(txt)
        results.append({"Text": txt, "Sentiment": sentiment, "Confidence": confidence})

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Extract keywords globally from all texts
    keywords = extract_keywords(texts)
    keywords_str = ", ".join(keywords)
    df['Keywords'] = keywords_str

    st.markdown("### Sentiment Analysis Results")
    st.dataframe(df.style.format({"Confidence": "{:.2f}"}))

    # Visualization: Sentiment distribution
    fig, ax = plt.subplots()
    df['Sentiment'].value_counts().plot(kind='bar', color=['#2ECC71', '#F1C40F', '#E74C3C'], ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Texts")
    st.pyplot(fig)

    # Wordcloud for keywords
    wordcloud = WordCloud(width=600, height=200, background_color='white').generate(keywords_str)
    st.markdown("### Keywords Wordcloud")
    st.image(wordcloud.to_array())

    # Export buttons
    st.markdown("### Export Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name='sentiment_results.csv', mime='text/csv')

    json_data = df.to_json(orient='records')
    st.download_button(label="Download JSON", data=json_data, file_name='sentiment_results.json', mime='application/json')

    pdf_data = generate_pdf(df)
    st.download_button(label="Download PDF", data=pdf_data, file_name="sentiment_results.pdf", mime='application/pdf')

else:
    st.info("Please enter or upload text data to analyze.")

# Footer
st.markdown(
    """
    <div style='text-align:center; margin-top: 40px; font-size:14px; color:#888;'>
    Developed with ❤️ using Streamlit & Hugging Face Transformers
    </div>
    """,
    unsafe_allow_html=True
)
