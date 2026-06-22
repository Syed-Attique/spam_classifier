import csv
import pickle

from naive_bayes_model import predict
from logistic_regression_model import predict_text


def load_manual_tests(path):
    dataset = []

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            message = row["message"].strip()
            expected_label = row["expected_label"].strip().lower()
            case_type = row["type"].strip()

            dataset.append((message, expected_label, case_type))

    return dataset


def evaluate_model(model_name, model_path, predict_function, dataset):
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    total = 0
    correct = 0
    wrong = 0
    uncertain = 0
    wrong_cases = []
    uncertain_cases = []
    print(f"\n--- Testing {model_name} ---")

    for message, expected_label, case_type in dataset:
        result = predict_function(message, model)
        prediction = result["prediction"]

        total += 1

        if prediction == expected_label:
            correct += 1
            status = "CORRECT"
        elif prediction == "uncertain":
            uncertain += 1
            status = "UNCERTAIN"
            uncertain_cases.append((message, expected_label, prediction, case_type))
        elif prediction != expected_label and prediction != "uncertain":
            wrong += 1
            status = "WRONG"
            wrong_cases.append((message, expected_label, prediction, case_type))

        print(f"{status} | Expected: {expected_label} | Predicted: {prediction} | Type: {case_type}")

    accuracy = correct / total if total > 0 else 0
    strict_accuracy = correct / total if total > 0 else 0
    confident = total - uncertain
    coverage = confident / total if total > 0 else 0
    confident_accuracy = correct / confident if confident > 0 else 0

    print(f"\n{model_name} Strict Manual Accuracy: {strict_accuracy}")
    print(f"Correct: {correct}/{total}")
    print(f"Wrong: {wrong}/{total}")
    print(f"Uncertain: {uncertain}/{total}")
    print(f"Coverage: {coverage}")
    print(f"Confident Accuracy: {confident_accuracy}")
    combined_cases = wrong_cases + uncertain_cases

    if combined_cases:
        print(f"\nWrong and Uncertain cases for {model_name}:")

        for message, expected_label, prediction, case_type in combined_cases:
            print("\nMessage:", message)
            print("Expected:", expected_label)
            print("Predicted:", prediction)
            print("Type:", case_type)


dataset = load_manual_tests("manual_test_cases.csv")

evaluate_model(
    "Naive Bayes",
    "models/naive_bayes_v3.pkl",
    predict,
    dataset
)

evaluate_model(
    "Logistic Regression",
    "models/logistic_regression_v3.pkl",
    predict_text,
    dataset
)