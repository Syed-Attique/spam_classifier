import math

dataset = [
    ("Win money now", "spam"),
    ("Free prize win", "spam"),
    ("Hello friend", "ham"),
    ("Lets meet tomorrow", "ham")
]
total_messages = len(dataset)
spam_word_counts = {}
ham_word_counts = {}
spam_messages = 0
ham_messages = 0

vocabulary = set()

for message, label in dataset:
    if label == "spam":
        spam_messages+=1
    else:
        ham_messages+=1
    
    words = message.lower().split()
    
    for word in words:
        
        vocabulary.add(word)

        if label == "spam":
            spam_word_counts[word] = spam_word_counts.get(word, 0) + 1
        
        else:
            ham_word_counts[word] = ham_word_counts.get(word, 0) + 1    

vocab_size = len(vocabulary)
p_spam = spam_messages / total_messages
p_ham = ham_messages / total_messages

print("Spam Counts:")
print(spam_word_counts)

print("\nHam Counts:")
print(ham_word_counts)


print("P(Spam):", p_spam)
print("P(Ham):", p_ham)

total_spam_words = sum(spam_word_counts.values())
total_ham_words = sum(ham_word_counts.values())

print("Total Spam Words:", total_spam_words)
print("Total Ham Words:", total_ham_words)

print("\nLikelihoods for Spam:")

for word, count in spam_word_counts.items():
    
    probability = count / total_spam_words
    
    print(word, ":", probability)

print("\nLikelihoods for Ham:")

for word, count in ham_word_counts.items():
    
    probability = count / total_ham_words
    
    print(word, ":", probability)


def calculate_word_probability(word, word_counts, total_words, vocab_size):
    
    count = word_counts.get(word, 0)
    
    probability = (count + 1) / (total_words + vocab_size)
    print(f"p({word}) = {probability}")
    
    return probability

def predict(message):
    
    words = message.lower().split()
    
    # ---- SPAM SCORE ----
    spam_score = math.log(p_spam)
    print("\nProbabilities of word being spam:")
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
    print("\nProbabilities of word being ham:")
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
        return "SPAM"
    else:
        return "HAM"


print(predict("hello lets play"))
