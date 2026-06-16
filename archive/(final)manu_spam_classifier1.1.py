import math
import random
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

dataset = []

with open("SMSSpamCollection", "r", encoding="utf-8") as file:
    
    for line in file:
        
        label, message = line.strip().split("\t")
        
        dataset.append((message, label))

print("Total Dataset Size:", len(dataset))

random.seed(42)
random.shuffle(dataset)

split_index = int(0.75 * len(dataset))

train_data = dataset[:split_index]
test_data = dataset[split_index:]

print("Training Size:", len(train_data))
print("Testing Size:", len(test_data))


total_messages = len(train_data)
spam_word_counts = {}
ham_word_counts = {}
spam_messages = 0
ham_messages = 0

vocabulary = set()

#---Frequency Filtering Words---
word_frequency = {}

for message, label in train_data:

    words = preprocess(message)

    for word in words:
        word_frequency[word] = word_frequency.get(word, 0) + 1

MIN_FREQ = 2

valid_words = {
    word
    for word, freq in word_frequency.items()
    if freq >= MIN_FREQ
}

#---Training---
for message, label in train_data:
    if label == "spam":
        spam_messages+=1
    else:
        ham_messages+=1
    
    words = preprocess(message)
    
    for word in words:
        if word not in valid_words:
            continue
        
        vocabulary.add(word)

        if label == "spam":
            spam_word_counts[word] = spam_word_counts.get(word, 0) + 1
        
        else:
            ham_word_counts[word] = ham_word_counts.get(word, 0) + 1   

vocab_size = len(vocabulary)
p_spam = spam_messages / total_messages
p_ham = ham_messages / total_messages

# print("Spam Counts:")
# print(spam_word_counts)

# print("\nHam Counts:")
# print(ham_word_counts)


print("P(Spam):", p_spam)
print("P(Ham):", p_ham)

total_spam_words = sum(spam_word_counts.values())
total_ham_words = sum(ham_word_counts.values())

print("Total Spam Words:", total_spam_words)
print("Total Ham Words:", total_ham_words)


# print("\nLikelihoods for Spam:")

# for word, count in spam_word_counts.items():
    
#     probability = count / total_spam_words
    
#     print(word, ":", probability)

# print("\nLikelihoods for Ham:")

# for word, count in ham_word_counts.items():
    
#     probability = count / total_ham_words
    
#     print(word, ":", probability)


def calculate_word_probability(word, word_counts, total_words, vocab_size):
    
    count = word_counts.get(word, 0)
    
    probability = (count + 1) / (total_words + vocab_size)
    # print(f"p({word}) = {probability}")
    
    return probability

word_scores = {}

for word in vocabulary:

    p_word_given_spam = calculate_word_probability(
        word,
        spam_word_counts,
        total_spam_words,
        vocab_size
    )

    p_word_given_ham = calculate_word_probability(
        word,
        ham_word_counts,
        total_ham_words,
        vocab_size
    )

    score = math.log(
        p_word_given_spam / p_word_given_ham
    )

    word_scores[word] = score

sorted_words = sorted(
    word_scores.items(),
    key=lambda x: x[1],
    reverse=True
)



def predict(message):
    
    words = preprocess(message)
    
    # ---- SPAM SCORE ----
    spam_score = math.log(p_spam)
    # ---- HAM SCORE ----
    ham_score = math.log(p_ham)
    
    # print("\nProbabilities of word being spam:")

    for word in words:
        if word not in vocabulary:
            continue
        
        spam_score += math.log(
            calculate_word_probability(
                word,
                spam_word_counts,
                total_spam_words,
                vocab_size
            )
        )
    
        # print("\nProbabilities of word being ham:")
        ham_score += math.log(
            calculate_word_probability(
                word,
                ham_word_counts,
                total_ham_words,
                vocab_size
            )
        )
    
    # ---- DECISION ----
    if spam_score > ham_score:
        return "spam"
    else:
        return "ham"


correct = 0
tp = 0
tn = 0
fp = 0
fn = 0

for message, label in test_data:
    
    prediction = predict(message)
    
    # True Positive
    if prediction == "spam" and label == "spam":
        tp += 1

    # True Negative
    elif prediction == "ham" and label == "ham":
        tn += 1

    # False Positive
    elif prediction == "spam" and label == "ham":
        fp += 1

    # False Negative
    elif prediction == "ham" and label == "spam":
        fn += 1
        
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print("\n--- Confusion Matrix ---")
print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

print("\n--- Metrics ---")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1_score)

# print("\n--- Top 20 Spam Words ---")
# for word, score in sorted_words[:20]:
#     print(word, ":", score)

# print("\n--- Top 20 Ham Words ---")
# for word, score in sorted_words[-20:]:
#     print(word, score)

print(vocabulary)