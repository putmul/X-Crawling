# Rupiah Sentiment Analysis

A Streamlit app for sentiment analysis and topic exploration of Indonesian
Rupiah-related posts. It provides sentiment prediction, word-frequency views,
word clouds, and TF-IDF/K-Means clustering with PCA visualization.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install -r requirement.txt
streamlit run main.py
```

## Dataset

The dataset was collected from public posts on **X (formerly Twitter)** using
Tweet Harvest in `colab/Crawling.ipynb`. The notebook searched the latest posts
for `rupiah` and requested up to 1,000 posts. `Cleaning.ipynb` then removed
links, mentions, hashtags, numbers, and special characters.

The checked-in CSVs are generated outputs:

- `dataset/rupiah_clean.csv` — cleaned text used for clustering.
- `dataset/rupiah_predicted.csv` — cleaned text with generated sentiment labels.

Labels were generated automatically and were not manually verified.

## Colab models

- `Predict.ipynb` applies a pre-trained Bag-of-Words + **Naive Bayes** model to
  create the initial positive/negative/neutral labels.
- `model.ipynb` trains the app model: **TF-IDF + Logistic Regression**, saved as
  `model/model_lr.joblib`.
- `Clustering.ipynb` represents the text with TF-IDF and groups it with
  **K-Means**. The Streamlit app recomputes the clustering and displays it with
  PCA.

The notebooks are in `colab/` and reproduce the data preparation and modeling
workflow.
