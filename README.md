# SMS Spam Classifier Using Naive Bayes

A spam-detection web application built using a **Multinomial Naive Bayes classifier implemented from scratch in Python**.

The project trains the model on the **SMS Spam Collection dataset** and provides a Flask web interface where a user can enter a message and classify it as:

* **Spam**
* **Ham** — a normal, non-spam message

The main goal of this project is to understand how Naive Bayes works internally instead of relying on machine-learning libraries such as scikit-learn.

---

## Features

* Naive Bayes implemented manually
* Text preprocessing
* Stopword removal
* Porter stemming
* Frequency-based vocabulary filtering
* Laplace smoothing
* Log-probability calculations
* Train/test split
* Confusion matrix
* Accuracy, precision, recall and F1-score
* Saved trained model using Pickle
* Flask web application
* Black and magenta user interface
* Displays processed, recognized and ignored words

---

## Project Structure

```text
spam_classifier/
│
├── app.py
├── train_model.py
├── naive_bayes_model.py
├── model.pkl
├── SMSSpamCollection
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── archive/
    └── previous versions and experiments
```

---

# How the Project Works

The project has two main phases:

```text
Training phase
      ↓
Saved model.pkl
      ↓
Prediction phase through Flask
```

The model is trained separately and saved to a file. The web application later loads that saved model and uses it to classify user messages.

The model is **not retrained every time the user submits a message**.

---

# File Explanation

## 1. `SMSSpamCollection`

This is the dataset used to train and test the classifier.

Each line contains:

```text
label    message
```

Example:

```text
ham     Hey, are we still meeting tomorrow?
spam    Congratulations! Claim your free prize now!
```

The two labels are:

* `spam`
* `ham`

---

## 2. `naive_bayes_model.py`

This file contains the reusable logic required to process and classify messages.

It does not load the dataset or train the model.

### `preprocess(message)`

This function cleans the input message before it is used by the classifier.

It performs the following steps:

### Lowercasing

```python
message = message.lower()
```

This makes words such as:

```text
FREE
Free
free
```

all become:

```text
free
```

### Punctuation removal

Characters such as:

```text
! , . ? :
```

are removed.

For example:

```text
WIN!!!
```

becomes:

```text
win
```

### Stopword removal

Common words such as:

```text
the, is, and, to
```

are removed because they usually provide little help in distinguishing spam from ham.

### Stemming

The Porter Stemmer converts related words into a common stem.

Examples:

```text
winning → win
claimed → claim
claims  → claim
```

This allows the model to treat related word forms as the same feature.

---

### `calculate_word_probability(...)`

This function calculates the probability of a word appearing in a class.

Examples:

```text
P(free | spam)
P(free | ham)
```

It uses Laplace smoothing:

```text
(count + 1) / (total words in class + vocabulary size)
```

Adding `1` prevents unseen words from receiving a probability of zero.

---

### `predict(message, model)`

This function classifies a new message.

First, the message is preprocessed.

The model then starts with the prior probabilities:

```text
P(spam)
P(ham)
```

For every recognized word, it adds the word's log-probability to both class scores.

Conceptually, it calculates:

```text
Spam score =
P(spam)
× P(word1 | spam)
× P(word2 | spam)
× ...
```

and:

```text
Ham score =
P(ham)
× P(word1 | ham)
× P(word2 | ham)
× ...
```

The implementation uses logarithms, so multiplication becomes addition:

```text
log(a × b) = log(a) + log(b)
```

This prevents numerical underflow when many small probabilities are multiplied.

The class with the larger score becomes the prediction.

The function returns:

```python
{
    "prediction": prediction,
    "spam_score": spam_score,
    "ham_score": ham_score,
    "processed_words": words,
    "used_words": used_words,
    "ignored_words": ignored_words
}
```

### Processed words

All words remaining after preprocessing.

### Used words

Words that exist in the model's trained vocabulary and contribute to the prediction.

### Ignored words

Words that were not present in the trained vocabulary.

---

## 3. `train_model.py`

This file trains and evaluates the Naive Bayes model.

It is run before starting the web application.

### Loading the dataset

The dataset is read and converted into tuples:

```python
(message, label)
```

### Train/test split

The dataset is shuffled and divided into:

```text
75% training data
25% testing data
```

A fixed random seed is used:

```python
random.seed(42)
```

This ensures the same train/test split is produced every time, making experiments reproducible.

### Frequency filtering

The program counts how frequently every word occurs in the training data.

Words occurring fewer than:

```python
MIN_FREQ = 2
```

are removed from the final vocabulary.

This reduces extremely rare features.

### Learning

The model learns:

* Number of spam messages
* Number of ham messages
* Frequency of every word in spam
* Frequency of every word in ham
* Total number of spam words
* Total number of ham words
* Spam prior probability
* Ham prior probability
* Final vocabulary

This is where the actual learning happens.

Unlike neural networks, this model does not learn weights through gradient descent. It learns by counting words and estimating probabilities.

### Evaluation

The model is evaluated on test messages that were not used during training.

The program calculates:

* True Positives
* True Negatives
* False Positives
* False Negatives
* Accuracy
* Precision
* Recall
* F1-score

### Saving the trained model

The learned values are stored inside a dictionary and saved using Pickle:

```python
pickle.dump(model, file)
```

This produces:

```text
model.pkl
```

---

## 4. `model.pkl`

This file contains the trained model.

It stores information such as:

```text
spam word counts
ham word counts
class priors
vocabulary
total word counts
```

The Flask application loads this file instead of training the model again.

Do not edit this file manually.

To create or update it, run:

```bash
python train_model.py
```

---

## 5. `app.py`

This is the Flask web application.

When the application starts, it loads the trained model:

```python
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
```

The `/` route accepts both:

```text
GET
POST
```

### GET request

Displays the empty message input form.

### POST request

Occurs when the user submits a message.

The application:

1. Reads the entered message
2. Sends it to `predict()`
3. Receives the prediction results
4. Sends those results to the HTML template

The original message is also added to the result dictionary:

```python
result["message"] = message
```

This allows the page to display the exact text entered by the user.

---

## 6. `templates/index.html`

This file contains the HTML interface.

It provides:

* Message input box
* Classify button
* Spam or ham prediction
* Original message
* Processed words
* Words used by the model
* Ignored words
* Advanced spam and ham scores

Flask uses Jinja syntax to display Python values:

```html
{{ result.prediction }}
```

Conditional blocks are used to display results only after a message has been submitted:

```html
{% if result %}
    ...
{% endif %}
```

---

## 7. `static/style.css`

This file controls the appearance of the web application.

The interface uses:

* Black background
* Magenta gradients and highlights
* Dark cards
* White readable text
* Red spam indicator
* Green ham indicator
* Responsive message input area
* Word chips for processed and ignored tokens

This file only controls presentation. It does not affect model predictions.

---

## 8. `archive/`

This folder stores:

* Earlier classifier versions
* Initial experiments
* TF-IDF practice code
* Previous README
* Original downloaded ZIP file

These files are kept for learning history but are not used by the current application.

---

## 9. `.gitignore`

This prevents unnecessary generated Python files from being uploaded to GitHub.

Example:

```gitignore
__pycache__/
*.pyc
```

---

# Features Used by the Model

The model uses words as features.

For example:

```text
Congratulations! Claim your free prize now!
```

may become:

```python
["congratul", "claim", "free", "prize", "now"]
```

Each word contributes evidence toward either spam or ham.

The model uses a **Bag-of-Words representation**, meaning:

* Word frequency matters
* Word order is ignored
* Sentence structure is ignored

For example:

```text
free prize claim
```

and:

```text
claim prize free
```

are treated as having the same features.

---

# Why It Is Called Naive Bayes

The model assumes that words are conditionally independent given the class.

Conceptually:

```text
P(free, prize, claim | spam)
=
P(free | spam)
× P(prize | spam)
× P(claim | spam)
```

This assumption is usually false because words in natural language are related.

However, the model can still classify text effectively because many individual words provide strong evidence.

Examples of spam-related words may include:

```text
claim
prize
award
winner
free
```
```text
"Why the hell is Naive Bayes working when its main assumption is obviously wrong?"

The Problem

Naive Bayes assumes:

P(X
1
	​

,X
2
	​

,…,X
n
	​

∣C)=
i
∏
	​

P(X
i
	​

∣C)

Meaning:

Given the class, all features are independent.

Example:

Message:

free prize winner

Naive Bayes assumes:

P(free,prize,winner∣spam)=P(free∣spam)×P(prize∣spam)×P(winner∣spam)

But are those words independent?

Obviously not.

Think:

free

often appears with:

prize
winner
claim
offer

These words are highly correlated.

The assumption is false.

Then Why Doesn't It Fail?

This puzzled ML researchers for years.

The key insight:

Naive Bayes doesn't need accurate probabilities.

It only needs the correct class ranking.

Example

Suppose true probabilities are:

P(spam|message) = 0.95
P(ham|message)  = 0.05

Naive Bayes might estimate:

P(spam|message) = 0.999999
P(ham|message)  = 0.000001

These probabilities are completely wrong.

But:

spam > ham

So classification is still correct.

Think About Your Project

Consider:

Claim your free prize now

Your model sees:

claim
free
prize

All three strongly indicate spam.

Even if the probabilities are exaggerated because the words are correlated:

spam score = huge
ham score = tiny

Spam still wins.

The Real Goal

Classification only cares about:

Which score is larger?

not:

Is the probability exactly correct?

This is a massive insight.

Why Text Classification Is Special

Suppose a message contains:

claim
free
winner
prize

Even if:

claim

and

prize

are dependent,

both are pointing toward the same class:

spam

The dependency doesn't usually change the final decision.

When Naive Bayes Works Best

When features are:

Individually informative

Example:

free
winner
claim

Each word alone already suggests spam.

High-dimensional

This is important.

Your dataset has thousands of possible words.

Each message uses only a tiny subset.

Example:

Vocabulary Size = 6000
Message Words = 10

Most features are zero.

Naive Bayes handles this very well.

When Naive Bayes Fails

Suppose meaning depends heavily on word combinations.

Example:

not good

Naive Bayes sees:

not
good

independently.

But the meaning comes from the combination.

This is where more advanced models outperform it.

Another Deep Insight

Look at your top spam words:

claim
prize
award
won
guarante

Even one of these words is often enough to push the score toward spam.

The classifier doesn't need a sophisticated understanding of language.

It only needs:

"These words appear far more often in spam than ham."

That's enough.

The Big Lesson

Naive Bayes succeeds because:

It only needs the correct class ranking, not accurate probabilities.
Many text features are individually informative.
High-dimensional sparse text data suits Naive Bayes well.
Correlated words often point to the same class, so independence violations don't hurt much.

Now here's the question I want you to think about:

Suppose the message is:

not a winner

Why might Naive Bayes struggle with this despite having the word:

winner

which is a very strong spam indicator? That question gets to one of the biggest weaknesses of the model.

The phrase not a winner is not individually informative. The meaning changes. winner and not a winner do not mean the same thing.

Naive Bayes sees:

not a winner

as individual words:

not
winner

and computes something like:

P(not∣class)×P(winner∣class)

It has no notion that "not" modifies "winner".

Humans understand:

winner

means:

positive / promotional

while:

not a winner

means:

negative / opposite meaning

But Naive Bayes doesn't understand phrases.

It only understands word frequencies.

So if:

winner

is a strong spam word, then:

not a winner

may still get a high spam score because the contribution of:

winner

can dominate.

This is called:

Loss of context

The model knows words.

It does not know relationships between words.

This happens because Naive Bayes only looks at individual words.

It does not look at word order or word relationships.

This is why more advanced models like Logistic Regression and SVM usually outperform Naive Bayes on real-world text classification tasks.

```text

---

# Installation

Install Python dependencies:

```bash
pip install flask nltk
```

---

# Training the Model

Run:

```bash
python train_model.py
```

This will:

1. Load the dataset
2. Split it into training and testing sets
3. Train the classifier
4. Print evaluation metrics
5. Create or replace `model.pkl`

---

# Running the Web Application

After training, run:

```bash
python app.py
```

Flask should display:

```text
Running on http://127.0.0.1:5000
```

Open that address in a browser.

---

# Example Inputs

### Spam example

```text
Congratulations! You have won a free prize. Claim it now.
```

Expected output:

```text
SPAM
```

### Ham example

```text
Hey, are we still meeting tomorrow afternoon?
```

Expected output:

```text
HAM
```

---

# Evaluation Metrics

The model is evaluated using:

### Accuracy

The percentage of all predictions that are correct.

### Precision

Out of all messages predicted as spam, how many were actually spam.

### Recall

Out of all actual spam messages, how many were detected.

### F1-score

A combined measure that balances precision and recall.

---

# Current Limitations

The model:

* Ignores word order
* Cannot fully understand context
* May struggle with negation such as `not a winner`
* May not recognize modern scam vocabulary absent from the dataset
* May perform poorly on languages other than those represented in training
* Treats unseen words as ignored features
* Uses an older SMS dataset, so real-world modern data may differ

This difference between training data and current real-world input is called **distribution shift**.

---

# Future Work

Planned improvements include:

* Logistic Regression classifier
* TF-IDF feature vectors
* Comparison between Naive Bayes and Logistic Regression
* Side-by-side predictions in the web interface
* Better explanation of model decisions
* Model deployment
* Improved preprocessing
* Modern spam datasets

---

# Learning Outcome

This project demonstrates the complete machine-learning workflow:

```text
Raw dataset
    ↓
Preprocessing
    ↓
Feature extraction
    ↓
Model training
    ↓
Evaluation
    ↓
Model serialization
    ↓
Web application
    ↓
Real user predictions
```

The classifier was built manually to provide a deeper understanding of how probability-based text classification works.
