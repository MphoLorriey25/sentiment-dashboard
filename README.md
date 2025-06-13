# Sentiment Analysis Dashboard

## Overview
This is an interactive sentiment analysis dashboard built with Python and Streamlit. It allows users to analyze text sentiment (positive, neutral, negative) with confidence scores, keyword extraction, batch processing, visualizations, and export results as CSV, JSON, or PDF.


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


## Setup Instructions

### 1. Clone the repository (or download the source code)

```bash
https://github.com/MphoLorriey25/sentiment-dashboard.git
cd sentiment-dashboard
````

### 2. Create and activate a Python virtual environment (optional but recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Upgrade pip and install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the App Locally

```bash
https://sentiment-dashboard-jmzufaa88feuna8v6ujtx4.streamlit.app/ 
```

This command starts the dashboard and opens it in your default web browser.

---

## Deployment

* Make sure your `requirements.txt` includes all necessary packages, such as:

```
streamlit
transformers
torch
nltk
scikit-learn
pandas
matplotlib
fpdf
```

* Push your code to GitHub.
* Connect your GitHub repository to [Streamlit Cloud](https://streamlit.io/cloud) or another hosting service.
* Deploy the app and let the platform install dependencies automatically.

---

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

## Troubleshooting

* If you get errors like `ModuleNotFoundError`, run:

  ```bash
  pip install -r requirements.txt
  ```
* For TensorFlow or PyTorch errors, install one of them:

  * TensorFlow: [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/)
  * PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
* Restart the app after installing packages.
* Check your internet connection for downloading models.

---


## License

This project is licensed under the MIT License.

```
```
# sentiment-dashboard
