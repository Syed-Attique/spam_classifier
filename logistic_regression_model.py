import math

def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)

def dot_product(weights, features):
    total = 0

    for w, x in zip(weights, features):
        total += w * x

    return total

def predict_probability(features, weights, bias):
    z = dot_product(weights, features) + bias
    return sigmoid(z)

def predict(features, weights, bias, threshold=0.5):
    probability = predict_probability(features, weights, bias)

    if probability >= threshold:
        return 1
    else:
        return 0

def prediction_label_from_probability(
    probability,
    spam_threshold=0.75,
    ham_threshold=0.35
):
    if probability >= spam_threshold:
        return "spam"
    elif probability <= ham_threshold:
        return "ham"
    else:
        return "uncertain"

from naive_bayes_model import preprocess
from tfidf_vectorizer import filter_words, document_to_tfidf


def predict_text(message, model):
    words = preprocess(message)

    vocabulary = model["vocabulary"]
    vocabulary_set = set(vocabulary)

    used_words = filter_words(words, vocabulary_set)

    features = document_to_tfidf(
        used_words,
        vocabulary,
        model["idf"]
    )

    probability = predict_probability(
        features,
        model["weights"],
        model["bias"]
    )

    spam_threshold = model.get("spam_threshold", 0.75)
    ham_threshold = model.get("ham_threshold", 0.35)

    prediction = prediction_label_from_probability(
        probability,
        spam_threshold,
        ham_threshold
    )

    ignored_words = [
        word
        for word in words
        if word not in vocabulary_set
    ]

    return {
        "prediction": prediction,
        "spam_probability": probability,
        "ham_probability": 1 - probability,
        "processed_words": words,
        "used_words": used_words,
        "ignored_words": ignored_words
    }
    