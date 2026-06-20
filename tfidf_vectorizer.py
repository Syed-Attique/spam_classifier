def filter_words(words, vocabulary_set):
    return [
        word
        for word in words
        if word in vocabulary_set
    ]


def document_to_tfidf(doc, vocabulary, idf):
    total_words = len(doc)

    if total_words == 0:
        return [0.0] * len(vocabulary)

    word_counts = {}

    for word in doc:
        word_counts[word] = word_counts.get(word, 0) + 1

    vector = []

    for word in vocabulary:
        tf = word_counts.get(word, 0) / total_words
        tfidf = tf * idf[word]
        vector.append(tfidf)

    return vector


def documents_to_tfidf(processed_docs, vocabulary, idf):
    vectors = []

    for doc in processed_docs:
        vector = document_to_tfidf(
            doc,
            vocabulary,
            idf
        )

        vectors.append(vector)

    return vectors