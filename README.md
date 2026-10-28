# Rupiah Sentiment Analysis

A Streamlit application for analyzing Indonesian and English text about the
Indonesian rupiah. The application uses a trained Logistic Regression pipeline
to classify text sentiment and provides dataset visualizations and K-Means
clustering analysis.

## Features

- Predict the sentiment of user-entered text as positive, negative, or neutral.
- Show sentiment predictions for individual words (token sentiment).
- Display the prepared rupiah dataset and sentiment distribution.
- Generate a word cloud and show the most frequently used words.
- Group text into clusters using TF-IDF and K-Means.
- Display the clustering result using a two-dimensional PCA visualization.

## Technologies

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- NLTK
- Matplotlib and WordCloud
- Joblib

## Project Structure

```text
week5/
├── main.py
├── model/
│   └── model_lr.joblib
├── rupiah_clean.csv
├── rupiah_predicted.csv
├── Cleaning.ipynb
├── model.ipynb
├── Predict.ipynb
├── Clustering.ipynb
├── requirement.txt
└── README.md
```

### Dataset and model files

- `rupiah_clean.csv` contains cleaned text used for clustering.
- `rupiah_predicted.csv` contains text with sentiment labels used in the
  analysis page.
- `model/model_lr.joblib` contains the trained sentiment classification
  pipeline.

## Installation

Create a virtual environment,

```bash
python -m venv .venv
```

and activate it:

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirement.txt
```

NLTK stopword data is downloaded automatically the first time the application
starts. An internet connection may be required for this download.

## Running the Application

Run:

```bash
streamlit run main.py
```

Use the sidebar to select one of these pages:

1. **Home** - enter text and view its sentiment prediction.
2. **Dataset & Analysis** - explore sentiment labels, word counts, and the word
   cloud.
3. **KMeans Clustering** - inspect text clusters and their PCA visualization.