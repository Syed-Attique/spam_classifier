import pickle
from flask import Flask, render_template, request

from naive_bayes_model import predict

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        message = request.form.get("message", "")

        if message.strip():
            result = predict(message, model)
            result["message"] = message

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)