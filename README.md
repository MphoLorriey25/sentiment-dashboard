# Sentiment Analysis Dashboard

## Overview
This is an interactive sentiment analysis dashboard built with Python and Streamlit. It allows users to analyze text sentiment (positive, neutral, negative) with confidence scores, keyword extraction, batch processing, visualizations, and export results as CSV, JSON, or PDF.

## 📊 Sentiment Analysis Dashboard
https://sentiment-dashboard-jmzufaa88feuna8v6ujtx4.streamlit.app/ 


## Features
- Input text via direct entry, multiple texts, or file upload (CSV, TXT)
- Multi-class sentiment classification (positive, neutral, negative)
- Confidence scoring for predictions
- Keyword extraction highlighting sentiment drivers
- Batch processing for multiple texts
- Visualizations showing sentiment distribution
- Export results to CSV, JSON, and PDF formats
- Error handling for invalid inputs or API failures
- Explanation of sentiment score rationale

## Usage Guide

1. Choose input mode:

   * Single Text: Enter one text.
   * Multiple Texts: Enter several texts separated by lines.
   * Upload File: Upload a `.csv` or `.txt` file with texts.
2. Click **Analyze**.
3. View results:

   * Sentiment classification with confidence scores.
   * Extracted keywords that influence sentiment.
   * Visualizations of sentiment distribution.
4. Export results to CSV, JSON, or PDF formats.

---

## Notes

* Uses Hugging Face model `cardiffnlp/twitter-roberta-base-sentiment-latest` for multi-class sentiment.
* Neutral sentiment detected when confidence is below threshold.
* TF-IDF used for keyword extraction.
* PDF export includes sentiment summary and keywords.
* Large batch sizes may be limited by API or hardware.

---

## License

This project is licensed under the MIT License.

