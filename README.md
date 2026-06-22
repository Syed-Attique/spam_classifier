<div align="center">

# 🛡️ SMS Spam Classifier

### Naive Bayes & Logistic Regression — Built From Scratch

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![ML From Scratch](https://img.shields.io/badge/ML-From%20Scratch-orange?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-5971%20Messages-brightgreen?style=flat-square)]()
[![No sklearn](https://img.shields.io/badge/No-scikit--learn-red?style=flat-square)]()

A spam detection system that classifies SMS messages using **two classical ML models built entirely from scratch** — no scikit-learn, TensorFlow, or any ML library.

[Overview](#-overview) • [Features](#-features) • [How It Works](#-how-it-works) • [Evaluation](#-evaluation-results) • [Setup](#-setup--usage) • [Examples](#-example-inputs)

</div>

---

## 📌 Overview

This project implements the **complete ML pipeline manually** — from raw data preprocessing to a Flask web app — covering every step of a real ML workflow:

```
Raw Dataset → Preprocessing → Feature Extraction → Model Training
     → Evaluation → Saved Artifacts → Flask Web App → Manual Testing
```

Two classifiers are trained and compared side by side:

| Model | Approach |
|---|---|
| **Multinomial Naive Bayes** | Word frequency-based probabilistic classification with Laplace smoothing |
| **Logistic Regression** | TF-IDF vectors trained via gradient descent with binary cross-entropy loss |

> **No ML libraries were used for core logic.** Libraries are used only for utilities:
> `pandas` (CSV loading) · `pickle` (model saving) · `flask` (web app) · `nltk` PorterStemmer (stemming)

---

## 🏷️ Classification Labels

| Label | Meaning |
|---|---|
| 🟢 **Ham** | Normal, non-spam message |
| 🔴 **Spam** | Spam, phishing, or smishing message |
| 🟡 **Uncertain** | Logistic Regression is not confident enough for a binary decision |

---

## ✨ Features

<details>
<summary><b>Models & Training</b></summary>

- Multinomial Naive Bayes from scratch
- Logistic Regression from scratch
- TF-IDF vectorization from scratch
- Binary cross-entropy loss
- Gradient descent weight updates
- Laplace smoothing
- Uncertain prediction mode for borderline messages

</details>

<details>
<summary><b>Preprocessing Pipeline</b></summary>

- Regex-based scam pattern detection (before punctuation removal)
- Special token replacement for URLs, emails, phones, currency, and numbers
- Stopword removal
- Porter stemming
- Unigram + bigram feature extraction
- Frequency-based vocabulary filtering (`MIN_FREQ`)

</details>

<details>
<summary><b>Evaluation</b></summary>

- Confusion matrix
- Accuracy, Precision, Recall, F1-Score
- Manually curated 40-case challenge test set
- Strict accuracy, coverage, and confident accuracy metrics

</details>

<details>
<summary><b>Web App</b></summary>

- Side-by-side model comparison (Naive Bayes vs Logistic Regression)
- Processed words / used words / ignored words display
- Probability bars and spam/ham scores
- Disagreement notice when models differ
- Dark UI with magenta highlights

</details>

---

## 🗂️ Project Structure

```
spam_classifier/
│
├── app.py                          # Flask web application
├── README.md
├── .gitignore
│
├── Dataset_5971.csv                # Primary dataset (2022 smishing dataset)
├── dataset_loader.py               # Dataset loading and label normalization
├── manual_test_cases.csv           # Hand-written 40-case challenge set
├── evaluate_manual_tests.py        # Manual evaluation script
│
├── naive_bayes_model.py            # NB prediction logic + shared preprocessing
├── logistic_regression_model.py    # LR prediction + uncertain mode
├── tfidf_vectorizer.py             # Custom TF-IDF vectorizer
│
├── train_naive_bayes.py            # Baseline NB training script (v1)
├── train_logistic_regression.py    # Baseline LR training script (v1)
├── train_naive_bayes_v2.py         # Improved NB training (produces v2/v3)
├── train_logistic_v2.py            # Improved LR training (produces v2/v3)
│
├── models/
│   ├── naive_bayes_v1.pkl
│   ├── logistic_regression_v1.pkl
│   ├── naive_bayes_v2.pkl
│   ├── logistic_regression_v2.pkl
│   ├── naive_bayes_v3.pkl          # ← Current best
│   └── logistic_regression_v3.pkl  # ← Current best
│
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── archive/                        # Older experiments and previous versions
```

---

## ⚙️ How It Works

### Preprocessing — `naive_bayes_model.py`

The `preprocess(message)` function prepares raw text before any model sees it:

| Step | Action |
|---|---|
| 1 | Lowercase the message |
| 2 | Detect scam patterns with regex (URLs, emails, phones, currency, numbers) |
| 3 | Replace matches with special tokens |
| 4 | Remove punctuation |
| 5 | Remove stopwords |
| 6 | Apply Porter stemming |
| 7 | Generate bigrams |

**Why regex tokens?** A naive punctuation-removal approach destroys URL structure:

```
# Without regex tokens
spam.com  →  spamcom    ✗  (URL signal lost)

# With regex tokens
spam.com  →  urltoken   ✓  (general scam signal preserved)
1000$     →  numbertoken currencytoken   ✓
```

**Special tokens:**

| Token | Matches |
|---|---|
| `urltoken` | `spam.com`, `gift-claim.xyz`, etc. |
| `emailtoken` | Any email address |
| `phonetoken` | `+923001234567`, etc. |
| `currencytoken` | `$`, `dollars`, `rupees`, etc. |
| `numbertoken` | Standalone numeric values |

**Bigrams add phrase-level context:**

```
Input:   "I won the debate competition today"
Stems:   won  debat  competit  today
Bigrams: won_debat · debat_competit · competit_today

won_prize  →  likely spam   ✗
won_debat  →  likely ham    ✓
```

---

### Logistic Regression — `logistic_regression_model.py`

Training is fully manual via gradient descent:

```
z     =  w · x + b
p     =  1 / (1 + e^−z)
loss  =  −[y·log(p) + (1−y)·log(1−p)]
w     =  w − lr × gradient
```

**Uncertain Mode** prevents confident wrong predictions on borderline messages:

| Spam Probability | Prediction |
|---|---|
| `p ≥ 0.75` | 🔴 Spam |
| `p ≤ 0.35` | 🟢 Ham |
| `0.35 < p < 0.75` | 🟡 Uncertain |

```
"Please visit github.com and check my project repository."
→ Contains urltoken → borderline → Uncertain   ✓ (better than a wrong confident call)
```

---

### TF-IDF — `tfidf_vectorizer.py`

Custom TF-IDF vectorizer, no scikit-learn:

```
TF-IDF  =  Term Frequency  ×  Inverse Document Frequency
```

Words frequent in a message but rare across the full corpus receive higher weight — e.g. `claim`, `prize`, `urltoken` score higher in spam-like messages.

---

## 🕰️ Model Version History

| Version | Dataset | Key Changes |
|---|---|---|
| **v1** | SMS Spam Collection (2012) | Baseline NB + LR, basic preprocessing, Flask app |
| **v2** | Dataset_5971.csv (2022) | Newer dataset, smishing → spam mapping, duplicate removal |
| **v3** ✅ | Dataset_5971.csv (2022) | Regex tokens, bigrams, currency regex fix, manual challenge set, uncertain mode |

---

## 📊 Evaluation Results

### Official Test Split

| Metric | Naive Bayes v3 | Logistic Regression v3 |
|---|:---:|:---:|
| **Accuracy** | **0.9890** | 0.9835 |
| **Precision** | **0.9818** | 0.9707 |
| **Recall** | **0.9609** | 0.9431 |
| **F1 Score** | **0.9712** | 0.9567 |

Naive Bayes v3 achieves the stronger official test-set performance. Logistic Regression v3 adds probability scores and uncertain-mode predictions.

---

### Manual Challenge Test Set

40 hand-written cases designed to expose real-world failure modes:

| Type | Count | Example |
|---|:---:|---|
| Easy Ham | 5 | `"Are you coming to class today?"` |
| Easy Spam | 5 | `"Congratulations you won a free prize."` |
| Hard Ham | 15 | `"I won the debate competition today."` |
| Hard Spam | 15 | Casual-wording scam messages |

**Hard ham** tests whether the model confuses normal sentences with spam-associated keywords:

```
"I won the debate competition today."    →  expected: ham
"I will claim my certificate tomorrow." →  expected: ham
"Send me the Zoom link for class."      →  expected: ham
```

| Model | Correct | Wrong | Uncertain | Accuracy |
|---|:---:|:---:|:---:|:---:|
| **Naive Bayes v3** | 29 / 40 | 11 / 40 | — | 72.5% |
| **Logistic Regression v3** | 25 / 40 | 7 / 40 | 8 / 40 | 62.5% strict · **78.1% confident** |

> ⚠️ High official accuracy ≠ strong real-world performance. The models still struggle with ham messages containing spam-associated words — an expected limitation of simple classical models.

---

## ⚠️ Known Limitations

| Limitation | Detail |
|---|---|
| Feature independence | Naive Bayes assumes words are conditionally independent |
| Linear boundary | Logistic Regression cannot capture non-linear patterns |
| Partial word order | Only bigrams — no full sequence modeling |
| Ambiguous keywords | `won`, `claim`, `link`, `prize`, `rupees` hurt ham precision |
| Vocabulary filtering | Rare phrases may be dropped entirely |
| Unseen patterns | Model may not generalize to new scam styles |

This is a **learning project** — not a production-grade spam filter.

---

## 🛠️ Setup & Usage

### 1. Install Dependencies

```bash
pip install flask pandas nltk
```

### 2. Train Models

```bash
# Naive Bayes — produces models/naive_bayes_v3.pkl
python train_naive_bayes_v2.py

# Logistic Regression — produces models/logistic_regression_v3.pkl
python train_logistic_v2.py
```

### 3. Run Manual Evaluation

```bash
python evaluate_manual_tests.py
```

### 4. Launch the Web App

```bash
python app.py
```

Open in browser: **`http://127.0.0.1:5000`**

---

## 🧪 Example Inputs

| Message | Expected | Notes |
|---|:---:|---|
| `Congratulations you have won 1000 dollars claim now.` | 🔴 Spam | Obvious spam |
| `Are you coming to class today?` | 🟢 Ham | Clear ham |
| `I won the debate competition today.` | 🟢 Ham | Hard ham — `won` may confuse the model |
| `Please visit github.com and check my project repository.` | 🟡 Uncertain | Contains `urltoken` — borderline |

---

## 🎓 Learning Outcomes

This project demonstrates:

- How Naive Bayes works internally — and why it still works despite its independence assumption
- How Logistic Regression is trained manually via gradient descent
- How TF-IDF vectors are computed from scratch
- How regex preprocessing significantly affects model behavior
- How bigrams add limited but useful phrase-level context
- Why high dataset accuracy does **not** guarantee real-world reliability
- Why manual challenge testing is a necessary complement to official metrics
- How uncertain predictions reduce confident wrong decisions
- How Git versioning supports iterative ML experimentation

---

<div align="center">

Built as a learning project to understand the full ML pipeline — from raw data to a working web interface.

</div>