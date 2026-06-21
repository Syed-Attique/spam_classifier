import math
import string
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stopwords = {
    "a", "the", "is", "and", "to",
    "in", "of", "for", "on", "at",
    "i", "you", "me", "my"
}

def preprocess(message):
    message = message.lower()

    email_pattern = r"\b[\w.-]+@[\w.-]+\.\w+\b"
    url_pattern = r"(https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|net|org|io|co|pk|uk|ly|info|biz|ru|cn|xyz)\S*)"
    phone_pattern = r"\+?\d[\d\s\-()]{7,}\d"
    currency_pattern = r"(\$|£|€|\brs\.?\b|\bpkr\b|\busd\b|\bdollars?\b|\brupees?\b|\bpounds?\b)"
    number_suffix_pattern = r"\b\d+(?:s|k|p|rs|pm|am)?\b"
    number_pattern = r"\b\d+\b"

    message = re.sub(email_pattern, " emailtoken ", message)
    message = re.sub(url_pattern, " urltoken ", message)
    message = re.sub(phone_pattern, " phonetoken ", message)
    message = re.sub(currency_pattern, " currencytoken ", message)
    message = re.sub(number_suffix_pattern, " numbertoken ", message)

    cleaned = ""

    for char in message:
        if char not in string.punctuation:
            cleaned += char

    words = []

    special_tokens = {
        "urltoken",
        "emailtoken",
        "phonetoken",
        "currencytoken",
        "numbertoken"
    }

    for word in cleaned.split():
        if word not in stopwords:
            if word in special_tokens:
                words.append(word)
            else:
                stemmed_word = stemmer.stem(word)
                words.append(stemmed_word)

    bigrams = []

    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i + 1]
        bigrams.append(bigram)

    return words + bigrams


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