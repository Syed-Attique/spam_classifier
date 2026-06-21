from dataset_loader import load_new_sms_dataset
import os
import random
import math
from naive_bayes_model import preprocess
from tfidf_vectorizer import (
    filter_words,
    documents_to_tfidf
)

SEED = 42
TRAIN_RATIO = 0.75
MIN_FREQ = 2

dataset = load_new_sms_dataset("Dataset_5971.csv")

random.seed(SEED)
random.shuffle(dataset)

split_index = int(TRAIN_RATIO * len(dataset))

train_data = dataset[:split_index]
test_data = dataset[split_index:]


def prepare_text_data(train_data):
    word_frequency = {}

    for message, label in train_data:
        words = preprocess(message)

        for word in words:
            word_frequency[word] = word_frequency.get(word, 0) + 1

    vocabulary = {
        word
        for word, freq in word_frequency.items()
        if freq >= MIN_FREQ
    }

    vocabulary = sorted(vocabulary)
    vocabulary_set = set(vocabulary)

    processed_train_docs = []
    train_labels = []

    for message, label in train_data:
        words = preprocess(message)

        filtered_words = filter_words(
            words,
            vocabulary_set
        )

        processed_train_docs.append(filtered_words)
        train_labels.append(1 if label == "spam" else 0)

    return processed_train_docs, train_labels, vocabulary

def compute_idf(processed_docs, vocabulary):
    total_docs = len(processed_docs)

    df = {
        word: 0
        for word in vocabulary
    }

    for doc in processed_docs:
        unique_words = set(doc)

        for word in unique_words:
            if word in df:
                df[word] += 1

    idf = {}

    for word in vocabulary:
        idf[word] = math.log(total_docs / df[word])

    return idf
        

processed_train_docs, train_labels, vocabulary = prepare_text_data(train_data)

idf = compute_idf(processed_train_docs, vocabulary)

print("Vocabulary Size:", len(vocabulary))
print("Sample IDF:", list(idf.items())[:10])


X_train = documents_to_tfidf(
    processed_train_docs,
    vocabulary,
    idf
)

from logistic_regression_model import predict, predict_probability

def train(X_train, y_train, learning_rate=0.1, epochs=100):
    num_features = len(X_train[0])

    weights = [0.0] * num_features
    bias = 0.0

    for epoch in range(epochs):

        total_loss = 0

        for features, label in zip(X_train, y_train):

            probability = predict_probability(features, weights, bias)

            error = probability - label

            for i in range(num_features):
                gradient = error * features[i]
                weights[i] = weights[i] - learning_rate * gradient

            bias = bias - learning_rate * error

            probability = min(max(probability, 1e-15), 1 - 1e-15)

            loss = -(
                label * math.log(probability)
                + (1 - label) * math.log(1 - probability)
            )

            total_loss += loss

        if epoch % 10 == 0:
            average_loss = total_loss / len(X_train)
            print("Epoch:", epoch, "Loss:", average_loss)

    return weights, bias


weights, bias = train(
    X_train,
    train_labels,
    learning_rate = 0.1,
    epochs = 70
)

print("Weights learned:", len(weights))
print("Bias:", bias)



def prepare_test_data(test_data, vocabulary):
    vocabulary_set = set(vocabulary)

    processed_test_docs = []
    test_labels = []

    for message, label in test_data:
        words = preprocess(message)

        filtered_words = filter_words(
            words,
            vocabulary_set
        )

        processed_test_docs.append(filtered_words)
        test_labels.append(1 if label == "spam" else 0)

    return processed_test_docs, test_labels


processed_test_docs, test_labels = prepare_test_data(test_data, vocabulary)

X_test = documents_to_tfidf(
    processed_test_docs,
    vocabulary,
    idf
)

tp = tn = fp = fn = 0

for features, label in zip(X_test, test_labels):
    prediction = predict(features, weights, bias)

    if prediction == 1 and label == 1:
        tp += 1
    elif prediction == 0 and label == 0:
        tn += 1
    elif prediction == 1 and label == 0:
        fp += 1
    elif prediction == 0 and label == 1:
        fn += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) != 0
    else 0
)

print("\n--- Logistic Regression Confusion Matrix ---")
print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

print("\n--- Logistic Regression Metrics ---")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1_score)

import pickle

logistic_model = {
    "weights": weights,
    "bias": bias,
    "vocabulary": vocabulary,
    "idf": idf,
    "threshold": 0.5,
    "min_freq": MIN_FREQ
}

os.makedirs("models", exist_ok=True)

with open("models/logistic_regression_v2.pkl", "wb") as file:
    pickle.dump(logistic_model, file)
