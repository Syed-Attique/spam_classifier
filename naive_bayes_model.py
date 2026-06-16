import math
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stopwords = {
    "a", "the", "is", "and", "to",
    "in", "of", "for", "on", "at",
    "i", "you", "me", "my"
}


def preprocess(message):
    message = message.lower()

    cleaned = ""

    for char in message:
        if char not in string.punctuation:
            cleaned += char

    words = []

    for word in cleaned.split():
        if word not in stopwords:
            stemmed_word = stemmer.stem(word)
            words.append(stemmed_word)

    return words


def calculate_word_probability(word, word_counts, total_words, vocab_size):
    count = word_counts.get(word, 0)
    return (count + 1) / (total_words + vocab_size)


def predict(message, model):
    words = preprocess(message)

    spam_score = math.log(model["p_spam"])
    ham_score = math.log(model["p_ham"])

    used_words = []
    ignored_words = []

    for word in words:
        if word not in model["vocabulary"]:
            ignored_words.append(word)
            continue

        used_words.append(word)

        spam_score += math.log(
            calculate_word_probability(
                word,
                model["spam_word_counts"],
                model["total_spam_words"],
                model["vocab_size"]
            )
        )

        ham_score += math.log(
            calculate_word_probability(
                word,
                model["ham_word_counts"],
                model["total_ham_words"],
                model["vocab_size"]
            )
        )

    if spam_score > ham_score:
        prediction = "spam"
    else:
        prediction = "ham"

    return {
        "prediction": prediction,
        "spam_score": spam_score,
        "ham_score": ham_score,
        "processed_words": words,
        "used_words": used_words,
        "ignored_words": ignored_words
    }