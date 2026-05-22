import math
import random
import string

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
            words.append(word)
    
    return words

dataset = []

with open("SMSSpamCollection", "r", encoding="utf-8") as file:
    
    for line in file:
        
        label, message = line.strip().split("\t")
        
        dataset.append((message, label))

print("Total Dataset Size:", len(dataset))

random.shuffle(dataset)

split_index = int(0.8 * len(dataset))

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

#---Training---
for message, label in train_data:
    if label == "spam":
        spam_messages+=1
    else:
        ham_messages+=1
    
    words = preprocess(message)
    
    for word in words:
        
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

def predict(message):
    
    words = preprocess(message)
    
    # ---- SPAM SCORE ----
    spam_score = math.log(p_spam)
    # print("\nProbabilities of word being spam:")
    for word in words:
        spam_score += math.log(
            calculate_word_probability(
                word,
                spam_word_counts,
                total_spam_words,
                vocab_size
            )
        )
    
    # ---- HAM SCORE ----
    ham_score = math.log(p_ham)
    # print("\nProbabilities of word being ham:")
    for word in words:
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

for message, label in test_data:
    
    prediction = predict(message)
    
    # print("--------------------------------")
    # print("Message:", message)
    # print("Prediction:", prediction)
    # print("Original Label:", label)
    # print("--------------------------------")
    
    if prediction == label:
        correct += 1
    else:
        print(f"Prediction Incorrect!!! -> Message: {message},\nPredicted: {prediction}, Actual: {label}\n")

accuracy = correct / len(test_data)

print("Accuracy:", accuracy)
