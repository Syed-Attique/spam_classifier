import pickle
from flask import Flask, render_template, request

from naive_bayes_model import predict as naive_bayes_predict
from logistic_regression_model import predict_text as logistic_predict


app = Flask(__name__)


with open("models/naive_bayes_v3.pkl", "rb") as file:
    naive_bayes_model = pickle.load(file)


with open("models/logistic_regression_v3.pkl", "rb") as file:
    logistic_model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def index():
    naive_bayes_result = None
    logistic_result = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            naive_bayes_result = naive_bayes_predict(
                message,
                naive_bayes_model
            )

            logistic_result = logistic_predict(
                message,
                logistic_model
            )

            print("\nMessage:", message)

            print(
                "Naive Bayes:",
                naive_bayes_result["prediction"],
                "| Spam score:",
                naive_bayes_result["spam_score"],
                "| Ham score:",
                naive_bayes_result["ham_score"]
            )

            print(
                "Logistic Regression:",
                logistic_result["prediction"],
                "| Spam probability:",
                logistic_result["spam_probability"],
                "| Ham probability:",
                logistic_result["ham_probability"]
            )

    return render_template(
        "index.html",
        message=message,
        naive_bayes_result=naive_bayes_result,
        logistic_result=logistic_result
    )


if __name__ == "__main__":
    app.run(debug=True)