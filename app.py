from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
import pandas as pd
import math
from collections import defaultdict

app = FastAPI()

class Tweet(BaseModel):
    text: str

# Placeholder for class probabilities, word probabilities, IDF, and number of documents
class_probs = {}
class_word_probs = {}
idf = {}
num_documents = 1  # Replace with actual number of documents

# TF-IDF Calculation
def calculate_tfidf(sentence: str, idf: Dict[str, float], num_documents: int) -> Dict[str, float]:
    words = sentence.split()
    tf = defaultdict(int)
    for word in words:
        tf[word] += 1
    tfidf_scores = {}
    for word, count in tf.items():
        tf_score = count / len(words)
        idf_score = idf.get(word, math.log(num_documents + 1))
        tfidf_scores[word] = tf_score * idf_score
    return tfidf_scores

# IDF Calculation
def calculate_idf(corpus: List[str]) -> Tuple[Dict[str, float], int]:
    num_documents = len(corpus)
    df = defaultdict(int)
    for document in corpus:
        words = set(document.split())
        for word in words:
            df[word] += 1
    idf = {word: math.log(num_documents / (count + 1)) for word, count in df.items()}
    return idf, num_documents

# Naive Bayes Training
def train_naive_bayes(corpus: List[str], labels: List[str]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float], int]:
    class_docs = defaultdict(list)
    for document, label in zip(corpus, labels):
        class_docs[label].append(document)
    class_probs = {label: len(docs) / len(corpus) for label, docs in class_docs.items()}
    word_counts = {label: defaultdict(int) for label in class_docs}
    for label, docs in class_docs.items():
        for doc in docs:
            for word in doc.split():
                word_counts[label][word] += 1
    class_word_probs = {}
    for label, counts in word_counts.items():
        total_words = sum(counts.values())
        class_word_probs[label] = {word: (count / total_words) for word, count in counts.items()}
    idf, num_documents = calculate_idf(corpus)
    return class_probs, class_word_probs, idf, num_documents

# Predict Function
def predict_naive_bayes(sentence: str, class_probs: Dict[str, float], class_word_probs: Dict[str, Dict[str, float]], idf: Dict[str, float], num_documents: int) -> str:
    tfidf_scores = calculate_tfidf(sentence, idf, num_documents)
    class_scores = {label: math.log(prob) for label, prob in class_probs.items()}
    for label, word_probs in class_word_probs.items():
        for word, score in tfidf_scores.items():
            word_prob = word_probs.get(word, 1e-7 / (len(word_probs) + len(tfidf_scores)))
            class_scores[label] += math.log(word_prob * (score + 1e-7))
    predicted_class = max(class_scores, key=class_scores.get)
    return predicted_class

# Sentiment Encoding
encodingSentiment = {"positive": 2, "negative": 0, "neutral": 1}

def encoding_sentiments(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset.replace(encodingSentiment, inplace=True)
    return dataset

@app.post("/predict")
def predict(tweet: Tweet):
    sentiment = predict_naive_bayes(tweet.text, class_probs, class_word_probs, idf, num_documents)
    return {"sentiment": sentiment}

# Load your training data
train_data_path = r"C:\Users\CHAITANYA\Documents\SENTIMENTAL_ANALYSIS_NLP_PROJECT\Train__Data.csv"
train_data = pd.read_csv(train_data_path)
test_data_path = r"C:\Users\CHAITANYA\Documents\SENTIMENTAL_ANALYSIS_NLP_PROJECT\Test_Data.csv"
test_data = pd.read_csv(test_data_path)

# Encode sentiments
train_data = encoding_sentiments(train_data)
test_data = encoding_sentiments(test_data)

# Ensure all text entries are strings and fill missing values
train_data['text'] = train_data['text'].astype(str).fillna("")
test_data['text'] = test_data['text'].astype(str).fillna("")

# Assuming the columns are 'text' and 'sentiment' (adjust as necessary)
corpus = train_data['text'].tolist()
labels = train_data['sentiment'].tolist()

# Preprocess and train the model with the actual data
class_probs, class_word_probs, idf, num_documents = train_naive_bayes(corpus, labels)
