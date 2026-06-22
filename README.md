# SMS Spam Classifier — Naive Bayes and Logistic Regression From Scratch

A Flask-based Machine Learning Spam detection project that classifies text messages as:

* **Ham** — normal/non-spam message
* **Spam** — spam, phishing, or smishing-style message
* **Uncertain** — used by Logistic Regression when the model is not confident enough to force a ham/spam decision

This project was built as a beginner machine-learning project, but the implementation goes beyond a "basic tutorial project". The main focus is to understand the complete ML pipeline manually:

```text
raw dataset
→ preprocessing
→ feature extraction
→ model training
→ evaluation
→ saved models
→ Flask web app
→ manual challenge testing
```

---

## Important Note

The machine-learning models in this project are implemented **from scratch in Python**.

No machine-learning library such as:

* scikit-learn
* TensorFlow
* PyTorch
* Keras

was used to train the models, create the classifiers, compute TF-IDF, or make predictions.

The project uses normal Python libraries for support tasks, such as:

* `pandas` for reading CSV files
* `pickle` for saving/loading trained models
* `Flask` for the web application
* `nltk` PorterStemmer for stemming

But the actual ML logic is manually implemented.

---

## Features

* Multinomial Naive Bayes classifier implemented from scratch
* Logistic Regression classifier implemented from scratch
* TF-IDF vectorization implemented from scratch
* Binary cross-entropy loss for Logistic Regression
* Gradient descent training for Logistic Regression
* Laplace smoothing for Naive Bayes
* Regex-based preprocessing for scam-related patterns
* Special tokens for URLs, emails, phone numbers, currency, and numbers
* Stopword removal
* Porter stemming
* Frequency-based vocabulary filtering
* Unigram and bigram feature extraction
* Confusion matrix evaluation
* Accuracy, precision, recall, and F1-score
* Manual challenge test set evaluation
* Logistic Regression uncertain mode
* Flask web interface
* Side-by-side model comparison in the app
* Displays processed words, used words, and ignored words
* Versioned model artifacts

---

## Project Structure

```text
spam_classifier/
│
├── app.py
├── README.md
├── .gitignore
│
├── Dataset_5971.csv
├── dataset_loader.py
├── manual_test_cases.csv
├── evaluate_manual_tests.py
│
├── naive_bayes_model.py
├── logistic_regression_model.py
├── tfidf_vectorizer.py
│
├── train_naive_bayes.py
├── train_logistic_regression.py
├── train_naive_bayes_v2.py
├── train_logistic_v2.py
│
├── models/
│   ├── naive_bayes_v1.pkl
│   ├── logistic_regression_v1.pkl
│   ├── naive_bayes_v2.pkl
│   ├── logistic_regression_v2.pkl
│   ├── naive_bayes_v3.pkl
│   └── logistic_regression_v3.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── archive/
    └── older experiments and previous versions
```

Some file names represent earlier checkpoints or experiment versions. The current best trained models are stored in the `models/` folder as v3 model artifacts.

---

# Dataset

The project originally started with the classic SMS Spam Collection dataset from 2012. Later, it was upgraded to a newer SMS phishing/smishing dataset from 2022:

```text
Dataset_5971.csv (current dataset)
```

This dataset contains:

* ham messages
* spam messages
* smishing messages
* URL indicator column
* email indicator column
* phone indicator column

For binary classification, labels are normalized as:

```text
ham      → ham
spam     → spam
smishing → spam
```

So the final task remains:

```text
ham vs spam
```

---

# How the Project Works

The system has two main phases:

```text
Training phase
      ↓
Saved model files
      ↓
Prediction phase through Flask app
```

The models are trained separately using training scripts. After training, the learned parameters are saved in `.pkl` files.

The Flask app does **not** retrain models when a user submits a message. It only loads the saved models and uses them for prediction.

---

# Core Files Explained

## 1. `dataset_loader.py`

This file loads the newer CSV dataset and converts it into a clean format used by the training scripts.

Main responsibilities:

* Read `Dataset_5971.csv`
* Normalize column names
* Normalize labels
* Remove duplicate messages
* Convert labels into binary format

Example mapping:

```text
ham      → ham
spam     → spam
Smishing → spam
smishing → spam
```

The final output is a list of tuples:

```python
(message, label)
```

Example:

```python
("Your account has been blocked. Verify now.", "spam")
```

---

## 2. `naive_bayes_model.py`

This file contains reusable Naive Bayes prediction logic and the shared preprocessing function.

### `preprocess(message)`

The preprocessing function prepares raw text before it enters the model.

It performs:

1. Lowercasing
2. Regex-based pattern matching
3. Special token replacement
4. Punctuation removal
5. Stopword removal
6. Stemming
7. Bigram creation

### Regex-Based Pattern Matching

Before punctuation is removed, the system detects important scam-related patterns.

Examples:

```text
spam.com              → urltoken
test@gmail.com        → emailtoken
+923001234567         → phonetoken
1000 dollars          → numbertoken currencytoken
1000$                 → numbertoken currencytoken
```

This matters because a naive punctuation-removal approach would turn:

```text
spam.com
```

into:

```text
spamcom
```

which loses the important URL structure.

### Special Tokens

The following special tokens are used:

```text
urltoken
emailtoken
phonetoken
currencytoken
numbertoken
```

These help the model learn general scam indicators instead of memorizing exact URLs or phone numbers.

### Bigrams

The model also creates bigrams.

Example:

```text
I won the debate competition today
```

After preprocessing and stemming:

```text
won debat competit today
```

Bigrams added:

```text
won_debat
debat_competit
competit_today
```

This helps the model learn limited phrase-level context.

For example:

```text
won_prize  → likely spam
won_debat  → likely ham
```

The models still use simple classical ML, but bigrams give them more context than single words alone.

---

## 3. `train_naive_bayes.py`

This is the earlier/baseline Naive Bayes training script.

It represents the original training flow before later dataset and preprocessing improvements.

---

## 4. `train_logistic_regression.py`

This is the earlier/baseline Logistic Regression training script.

It represents the original Logistic Regression implementation before later dataset and preprocessing improvements.

---

## 5. `train_naive_bayes_v2.py`

This script trains the newer Naive Bayes models using the newer dataset and improved preprocessing pipeline.

Main responsibilities:

* Load the newer dataset
* Shuffle data using a fixed random seed
* Split into train/test sets
* Preprocess messages
* Build vocabulary
* Apply frequency filtering using `MIN_FREQ`
* Count word frequencies for spam and ham
* Apply Laplace smoothing
* Evaluate using test data
* Save the trained model into the `models/` folder

The current v3 Naive Bayes model was produced after improvements such as:

* regex pattern tokens
* fixed currency regex
* bigram features

---

## 6. `train_logistic_v2.py`

This script trains the newer Logistic Regression models using the newer dataset and improved preprocessing pipeline.

Main responsibilities:

* Load the newer dataset
* Preprocess messages
* Build vocabulary
* Compute IDF values
* Convert messages into TF-IDF vectors
* Train Logistic Regression using gradient descent
* Evaluate using test data
* Save the trained model into the `models/` folder

### Logistic Regression Training

The model calculates:

```text
z = w · x + b
```

Then applies sigmoid:

```text
p = 1 / (1 + e^-z)
```

The model learns weights using gradient descent.

For each feature:

```text
weight = weight - learning_rate × gradient
```

The loss function is binary cross-entropy:

```text
loss = -[y log(p) + (1-y) log(1-p)]
```

All of this is implemented manually.

---

## 7. `tfidf_vectorizer.py`

This file contains the custom TF-IDF vectorization logic.

It does not use scikit-learn.

It includes:

* filtering words by vocabulary
* converting one document into a TF-IDF vector
* converting multiple documents into TF-IDF vectors

### TF-IDF Meaning

TF-IDF means:

```text
Term Frequency × Inverse Document Frequency
```

It gives higher weight to words that are important in a message but not common across all messages.

Example words/tokens such as:

```text
claim
prize
urltoken
```

may receive stronger values in spam-like messages.

---

## 8. `logistic_regression_model.py`

This file contains reusable Logistic Regression prediction logic.

It includes:

* sigmoid calculation
* dot product
* probability prediction
* binary prediction
* full text prediction using TF-IDF
* uncertain-mode prediction for the web app/manual tests

### Uncertain Mode

Normal binary classification forces the model to say:

```text
ham or spam
```

But some messages are borderline.

So Logistic Regression supports uncertain mode:

```text
spam_probability >= 0.75 → spam
spam_probability <= 0.35 → ham
otherwise                → uncertain
```

This does not make the model magically smarter. It simply prevents the app from being confidently wrong on borderline messages.

Example:

```text
Please visit github.com and check my project repository.
```

This may be suspicious because it contains a URL, but it can also be completely normal. Uncertain mode is useful for cases like this.

---

## 9. `evaluate_manual_tests.py`

This script evaluates the trained models on a manually created challenge set.

The manual test set is intentionally harder than the normal test split.

It includes:

* easy ham messages
* easy spam messages
* hard ham messages containing spam-like words
* hard spam messages written in casual wording

The script prints:

* expected label
* predicted label
* case type
* correct cases
* wrong cases
* uncertain cases
* strict manual accuracy
* coverage
* confident accuracy

This is useful because normal dataset metrics can look very strong while the model still struggles with real-world edge cases.

---

## 10. `manual_test_cases.csv`

This file contains manually written test examples.

It is used only for evaluation, not training.

Example format:

```csv
message,expected_label,type
"I won the debate competition today.","ham","hard_ham"
"Congratulations you won a free prize.","spam","easy_spam"
```

The purpose is to test whether the model handles confusing cases such as:

```text
won debate          → ham
won prize           → spam
claim certificate   → ham
claim reward        → spam
lecture link        → ham
click link          → spam
```

---

## 11. `app.py`

This is the Flask web application.

It loads the saved Naive Bayes and Logistic Regression models and shows predictions in the browser.

The app displays:

* original message
* Naive Bayes prediction
* Logistic Regression prediction
* processed words
* words used by each model
* ignored words
* Naive Bayes spam/ham scores
* Logistic Regression spam/ham probabilities
* disagreement notice when the two models disagree

The app allows side-by-side comparison between both classifiers.

---

## 12. `templates/index.html`

This file contains the HTML structure of the web interface.

It uses Flask/Jinja syntax to display prediction results.

The page contains:

* message input box
* classify button
* Naive Bayes result card
* Logistic Regression result card
* processed/used/ignored word sections
* probability and score display
* original message display

---

## 13. `static/style.css`

This file controls the visual design of the app.

The interface uses:

* dark background
* magenta highlights
* model comparison cards
* spam/ham prediction badges
* probability bars
* responsive layout

This file only affects the UI. It does not affect model behavior.

---

## 14. `models/`

This folder stores saved model artifacts.

Current model versions include:

```text
naive_bayes_v1.pkl
logistic_regression_v1.pkl
naive_bayes_v2.pkl
logistic_regression_v2.pkl
naive_bayes_v3.pkl
logistic_regression_v3.pkl
```

The `.pkl` files store learned parameters such as:

* vocabulary
* word counts
* class priors
* TF-IDF IDF values
* Logistic Regression weights
* Logistic Regression bias
* model thresholds

Do not edit these files manually.

---

# Model Version Timeline

## Version 1 — Baseline SMS Spam Classifier

The first version used the older SMS Spam Collection dataset.

### Implemented

* Multinomial Naive Bayes from scratch
* Text preprocessing
* stopword removal
* stemming
* punctuation removal
* vocabulary filtering
* Laplace smoothing
* confusion matrix
* accuracy, precision, recall, F1-score
* Flask app

### Later Added

* Logistic Regression from scratch
* TF-IDF from scratch
* model comparison in the app

### Main Limitation

The dataset was older and did not represent modern scam patterns well.

Examples of weak cases:

```text
crypto scams
modern phishing links
smishing messages
casual scam wording
```

---

## Version 2 — Newer Dataset and New Model Artifacts

The second version upgraded the project to the newer `Dataset_5971.csv`.

### Improvements

* Added `dataset_loader.py`
* Normalized labels
* Mapped smishing into spam
* Removed duplicate messages
* Trained newer Naive Bayes model
* Trained newer Logistic Regression model
* Saved v2 model artifacts

### Result

The models improved on modern spam/smishing examples.

However, preprocessing still had a weakness:

```text
spam.com → spamcom
```

URLs, emails, phone numbers, and money amounts were not handled properly yet.

---

## Version 3 — Regex Tokens, Bigrams, and Improved Evaluation

Version 3 introduced the biggest preprocessing improvements.

### Improvements

* Added regex-based pattern matching
* Replaced URLs with `urltoken`
* Replaced emails with `emailtoken`
* Replaced phone numbers with `phonetoken`
* Replaced currency expressions with `currencytoken`
* Replaced numbers with `numbertoken`
* Fixed currency regex bug where `rs` matched inside words like `university`
* Added bigram features
* Retrained Naive Bayes v3
* Retrained Logistic Regression v3
* Added manual challenge testing
* Added Logistic Regression uncertain mode

### Why This Helped

Instead of memorizing exact URLs like:

```text
secure-login.com
gift-claim.xyz
```

the model can learn the general signal:

```text
urltoken
```

Instead of treating `1000$` as a broken token, the model sees:

```text
numbertoken currencytoken
```

Bigrams also help capture phrase-level context:

```text
free_prize
click_link
claim_reward
```

### Result

Version 3 produced the strongest official dataset performance.

---

# Evaluation Results

## Official Test Split

The official train/test split uses the dataset itself and evaluates on held-out messages.

### Naive Bayes v3

```text
Accuracy : 0.9890
Precision: 0.9818
Recall   : 0.9609
F1 Score : 0.9712
```

### Logistic Regression v3

```text
Accuracy : 0.9835
Precision: 0.9707
Recall   : 0.9431
F1 Score : 0.9567
```

### Interpretation

Naive Bayes v3 achieved the strongest official test-set performance.

Logistic Regression v3 is still useful because it provides probability scores and supports uncertain prediction.

---

## Manual Challenge Test Set

A separate manually written test set was created to evaluate real-world edge cases.

This test set is not used for training.

It contains:

```text
15 hard ham
15 hard spam
5 easy ham
5 easy spam
```

### Why This Test Exists

Official test metrics can be high while the model still fails on tricky messages.

Examples of hard ham:

```text
I won the debate competition today.
Send me the Zoom link for class.
I will claim my certificate from the office tomorrow.
My phone number changed save this one.
```

These are normal messages, but they contain words that often appear in spam.

### Naive Bayes v3 Manual Test

```text
Correct: 29/40
Accuracy: 72.5%
```

Naive Bayes performed well on easy messages and most hard spam messages, but struggled with hard ham messages.

### Logistic Regression v3 Manual Test with Uncertain Mode

```text
Correct: 25/40
Wrong: 7/40
Uncertain: 8/40

Strict accuracy: 62.5%
Coverage: 80.0%
Confident accuracy: 78.1%
```

### Interpretation

Uncertain mode reduced the number of confident wrong predictions, but it also abstained on some cases.

This is useful in the web app because some messages are genuinely borderline for a simple model.

---

# Known Limitations

This project is intentionally built using beginner-friendly classical ML models.

The models do not truly understand language meaning.

They may struggle with normal messages containing spam-like words:

```text
won
claim
link
prize
phone
payment
rupees
dollars
```

Examples:

```text
I won the debate competition today.
Please send me the lecture link.
I will claim my certificate tomorrow.
```

The model may still classify these as spam because these words are statistically associated with spam in the training data.

Other limitations:

* Naive Bayes assumes feature independence
* Logistic Regression is still a linear model
* Word order is only partially captured through bigrams
* Rare phrases may be ignored due to vocabulary filtering
* The system may not generalize perfectly to unseen scam styles
* Manual challenge results are lower than official dataset metrics
* The app is a learning/demo project, not a production spam filter

---

# Why Naive Bayes Can Still Work

Naive Bayes assumes words are conditionally independent given the class.

This assumption is not fully true.

For example:

```text
free prize winner
```

The words `free`, `prize`, and `winner` are related. They are not independent.

However, Naive Bayes can still classify text well because it does not need perfect probability estimates. It mainly needs the correct class score to be larger.

If words like:

```text
claim
free
prize
winner
```

all strongly point toward spam, the spam score usually wins even if the exact probability is not perfectly calibrated.

This is why Naive Bayes often works surprisingly well for high-dimensional text classification.

---

# Running the Project

## 1. Install Dependencies

```bash
pip install flask pandas nltk
```

## 2. Train Naive Bayes

```bash
python train_naive_bayes_v2.py
```

This creates or updates:

```text
models/naive_bayes_v3.pkl
```

## 3. Train Logistic Regression

```bash
python train_logistic_v2.py
```

This creates or updates:

```text
models/logistic_regression_v3.pkl
```

## 4. Run Manual Evaluation

```bash
python evaluate_manual_tests.py
```

This evaluates models on `manual_test_cases.csv`.

## 5. Run the Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# Example Inputs

## Obvious Spam

```text
Congratulations you have won 1000 dollars claim now.
```

Expected:

```text
spam
```

## Normal Ham

```text
Are you coming to class today?
```

Expected:

```text
ham
```

## Hard Ham

```text
I won the debate competition today.
```

Expected:

```text
ham
```

This type of message may still confuse the model because the word `won` is strongly associated with spam.

## Borderline Message

```text
Please visit github.com and check my project repository.
```

Expected:

```text
ham or uncertain
```

The message contains a URL, which is suspicious in spam detection, but it can also be completely normal.

---

# Learning Outcomes

This project demonstrates:

* how Naive Bayes works internally
* how Logistic Regression works internally
* how TF-IDF vectors are built
* how preprocessing affects model behavior
* how regex pattern extraction can improve text classification
* how bigrams add limited phrase context
* why high dataset scores do not guarantee real-world reliability
* why manual challenge testing matters
* how precision, recall, and F1-score reveal more than accuracy
* how uncertain predictions can reduce confident wrong decisions
* how Git versioning helps track ML experiments

---

# Final Project Status

This is a strong beginner ML project.

It includes:

```text
classical ML from scratch
modern dataset
preprocessing experiments
feature engineering
model comparison
manual evaluation
Flask app
versioned improvements
clear known limitations
```

The project is not a production-grade spam filter. It is a learning project designed to understand the full ML workflow from data to deployment-style interface.