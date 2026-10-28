import streamlit as st
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords as stopwords_scratch
nltk.download('stopwords')
list_stopwords = stopwords_scratch.words('indonesian') # panggil stopword id
list_stopwords_en = stopwords_scratch.words('english') # panggil stopword en
list_stopwords.extend(list_stopwords_en) # gabungkan id & en
list_stopwords.extend(['ya', 'yg', 'ga', 'yuk', 'dah', 'ngga', 'engga', 'ygy']) # tambah daftar stopword jika perlu
stopwords = list_stopwords

# Word Cloud function
def generate_wordcloud(data):
    from wordcloud import WordCloud
    data_text = ' '.join(data['full_text'].astype(str).tolist()) # Combine the full_text column into one large string
    wc = WordCloud(stopwords=stopwords, background_color='black', max_words=500, width=800, height=400).generate(data_text) # Generate word cloud
    # Display the word cloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to plot top frequent words
def plot_top_words(data):
    from collections import Counter
    text = ' '.join(data['full_text'].astype(str).tolist())
    words = text.split()
    words = [word for word in words if word not in stopwords]
    word_counts = Counter(words)
    top_words = word_counts.most_common(12)
    words, counts = zip(*top_words)

    colors = plt.cm.Paired(range(len(words)))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(words, counts, color=colors)
    plt.xlabel('Kata')
    plt.ylabel('Frekuensi')
    plt.title('Kata yang sering muncul')
    plt.xticks(rotation=45)

    for bar, num in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=12, color='black', ha='center')

    st.pyplot(plt)

def analyze_token_sentiment(docx, pipeline):
    pos_list = []
    neg_list = []
    neu_list = []

    for word in docx.split():
        prediction = pipeline.predict([word])[0] # Predict sentiment for each token using the trained pipeline

        if hasattr(pipeline, 'predict_proba'): # Check if the pipeline supports predict_proba
            proba = pipeline.predict_proba([word])
            proba_max = proba.max()  # Get the highest probability
        else:
            proba_max = None  # If predict_proba is not available, return None

        if prediction == 'positive':
            pos_list.append([word, proba_max])
        elif prediction == 'negative':
            neg_list.append([word, proba_max])
        else:
            neu_list.append(word)

    result = {'positives': pos_list,'negatives': neg_list,'neutral': neu_list,}
    return result


# Load the trained pipeline
import joblib
pipeline = joblib.load('model/model_lr.joblib')

def main():
    import pandas as pd
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Projects")

    menu = ["Home", "Dataset & Analysis", "KMeans Clustering"] # Add sidebar with 3 menu options
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form("nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # Layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                sentiment = pipeline.predict([raw_text])[0] # Predict sentiment of the entire text using the pipeline
                st.write(f"Predicted Sentiment:")

                # Emoji based on prediction
                if sentiment == "positive":
                    st.markdown("# Positive 😄")
                elif sentiment == "negative":
                    st.markdown("# Negative 😡")
                else:
                    st.markdown("# Neutral 😐")

            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text, pipeline) # Analyze sentiment for individual tokens
                st.write(token_sentiments)

    elif choice == "Dataset & Analysis":
        st.subheader("Dataset & Sentiment Analysis")

        # Load dataset
        dataset_path = 'dataset/rupiah_predicted.csv'
        data = pd.read_csv(dataset_path)

        # Display dataset
        st.write("### Dataset")
        st.dataframe(data)

        # Count sentiment distribution
        sentiment_counts = data['sentiment'].value_counts()
        st.write("### Distribusi Sentimen")
        st.bar_chart(sentiment_counts)

        # Display number of positive, negative, and neutral sentiments
        st.write("#### Jumlah Sentimen")
        st.write(sentiment_counts)

        # Generate Word Cloud
        st.write("### Word Cloud")
        generate_wordcloud(data)

        # Plot top frequent words
        st.write("### Most Frequent Words")
        plot_top_words(data)

        # Display dataset with sentiment labels
        st.write("### Dataset with Sentiment Labels")
        st.dataframe(data[['full_text', 'sentiment']])

    else:
        st.subheader("KMeans Clustering Analysis")
        
        from sklearn.cluster import KMeans
        # Load dataset
        data = pd.read_csv('dataset/rupiah_clean.csv')
        # Convert text data to list
        text_data = data['full_text'].to_list()

        # Vectorization
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(text_data)

        # Silhouette Score
        range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
        high = 0
        true_k = 0

        for n_clusters in range_n_clusters:
            # Initialize KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(X)

            # Calculate score
            from sklearn.metrics import silhouette_score  
            score = silhouette_score(X, cluster_labels)
            if score > high:
                high = score
                true_k = n_clusters

        # KMeans clustering
        model = KMeans(n_clusters=true_k, random_state=42, n_init='auto')
        cluster_labels = model.fit_predict(X)

        # Display clusters
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        cluster_info = ""
        for i in range(true_k):
            cluster_info += f"Cluster {i}: " + ", ".join([terms[ind] for ind in order_centroids[i, :10]]) + ", ...\n"
        st.write("### Cluster Terms")
        st.text(cluster_info)

        # PCA for 2D visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=0)
        reduced_features = pca.fit_transform(X.toarray())
        reduced_cluster_centers = pca.transform(model.cluster_centers_)

        st.write(f"### Silhouette Score: {high:.2f} with {true_k} clusters")

        # Plotting the clusters
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
        plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='red', label='Centroids')
        plt.title('KMeans Clustering Visualization')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(scatter)
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()