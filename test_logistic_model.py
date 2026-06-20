import pickle
from logistic_regression_model import predict_text

with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

result = predict_text(
    "Congratulations! Claim your free prize now.",
    model
)

print(result)