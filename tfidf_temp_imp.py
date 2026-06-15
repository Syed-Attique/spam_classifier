import string
import math

docs = [
    "free prize winner",
    "free free claim prize",
    "ok see you later"
]

total_docs = len(docs)


def preprocess(msg):
    msg = msg.lower()

    cleaned = ""

    for char in msg:
        if char not in string.punctuation:
            cleaned += char

    return cleaned.split()


processed_docs = [preprocess(msg) for msg in docs]

# Build vocabulary
vocab_set = set()

for words in processed_docs:
    for word in words:
        vocab_set.add(word)

# Fixed column order for the matrix
vocab = sorted(vocab_set)

# Calculate TF for each document
all_tf = []

for words in processed_docs:
    word_freq = {}

    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    tf = {}

    for word, count in word_freq.items():
        tf[word] = count / len(words)

    all_tf.append(tf)

# Calculate DF
df = {}

for word in vocab:
    df[word] = 0

    for words in processed_docs:
        if word in words:
            df[word] += 1

# Calculate IDF
idf = {}

for word in vocab:
    idf[word] = math.log(total_docs / df[word])

# Calculate TF-IDF dictionaries
all_tfidf = []

for tf in all_tf:
    tfidf = {}

    for word in vocab:
        tfidf[word] = tf.get(word, 0) * idf[word]

    all_tfidf.append(tfidf)

# Build TF-IDF matrix
tfidf_matrix = []

for tfidf in all_tfidf:
    vector = []

    for word in vocab:
        vector.append(tfidf[word])

    tfidf_matrix.append(vector)

print("Vocabulary:")
print(vocab)

print("\nIDF:")
print(idf)

print("\nTF-IDF dictionaries:")
for tfidf in all_tfidf:
    print(tfidf)

print("\nTF-IDF Matrix:")
for vector in tfidf_matrix:
    print(vector)