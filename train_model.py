import pickle
import random

from naive_bayes_model import preprocess


DATASET_PATH = "SMSSpamCollection"
MODEL_PATH = "model.pkl"
MIN_FREQ = 2
SEED = 42
TRAIN_RATIO = 0.75


def load_dataset(path):
    dataset = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            label, message = line.strip().split("\t")
            dataset.append((message, label))

    return dataset


def train_naive_bayes(train_data):
    total_messages = len(train_data)

    spam_word_counts = {}
    ham_word_counts = {}

    spam_messages = 0
    ham_messages = 0

    word_frequency = {}

    for message, label in train_data:
        words = preprocess(message)

        for word in words:
            word_frequency[word] = word_frequency.get(word, 0) + 1

    valid_words = {
        word
        for word, freq in word_frequency.items()
        if freq >= MIN_FREQ
    }

    vocabulary = set()

    for message, label in train_data:
        if label == "spam":
            spam_messages += 1
        else:
            ham_messages += 1

        words = preprocess(message)

        for word in words:
            if word not in valid_words:
                continue

            vocabulary.add(word)

            if label == "spam":
                spam_word_counts[word] = spam_word_counts.get(word, 0) + 1
            else:
                ham_word_counts[word] = ham_word_counts.get(word, 0) + 1

    model = {
        "spam_word_counts": spam_word_counts,
        "ham_word_counts": ham_word_counts,
        "spam_messages": spam_messages,
        "ham_messages": ham_messages,
        "total_spam_words": sum(spam_word_counts.values()),
        "total_ham_words": sum(ham_word_counts.values()),
        "p_spam": spam_messages / total_messages,
        "p_ham": ham_messages / total_messages,
        "vocabulary": vocabulary,
        "vocab_size": len(vocabulary),
        "min_freq": MIN_FREQ
    }

    return model


def evaluate(test_data, model):
    from naive_bayes_model import predict

    tp = tn = fp = fn = 0

    for message, label in test_data:
        prediction = predict(message, model)["prediction"]

        if prediction == "spam" and label == "spam":
            tp += 1
        elif prediction == "ham" and label == "ham":
            tn += 1
        elif prediction == "spam" and label == "ham":
            fp += 1
        elif prediction == "ham" and label == "spam":
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def main():
    dataset = load_dataset(DATASET_PATH)

    random.seed(SEED)
    random.shuffle(dataset)

    split_index = int(TRAIN_RATIO * len(dataset))

    train_data = dataset[:split_index]
    test_data = dataset[split_index:]

    model = train_naive_bayes(train_data)
    metrics = evaluate(test_data, model)

    print("Training Size:", len(train_data))
    print("Testing Size:", len(test_data))
    print("Vocabulary Size:", model["vocab_size"])

    print("\n--- Confusion Matrix ---")
    print("TP:", metrics["tp"])
    print("TN:", metrics["tn"])
    print("FP:", metrics["fp"])
    print("FN:", metrics["fn"])

    print("\n--- Metrics ---")
    print("Accuracy :", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall   :", metrics["recall"])
    print("F1 Score :", metrics["f1_score"])

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("\nModel saved to", MODEL_PATH)


if __name__ == "__main__":
    main()