# SMS Spam Classifier — Naive Bayes and Logistic Regression From Scratch

A Flask-based Machine Learning Spam detection project that classifies text messages as:

- **Ham** — normal/non-spam message
- **Spam** — spam, phishing, or smishing-style message  
- **Uncertain** — used by Logistic Regression when the model is not confident enough to force a ham/spam decision

This project was built as a beginner machine-learning project, but the implementation goes beyond a "basic tutorial project". The main focus is to understand the complete ML pipeline manually:

```
raw dataset
  ↓
preprocessing
  ↓
feature extraction
  ↓
model training
  ↓
evaluation
  ↓
saved models
  ↓
Flask web app
  ↓
manual challenge testing
```

---

## ⚠️ Important Note

The machine-learning models in this project are implemented **from scratch in Python**.

No machine-learning library such as:
- `scikit-learn`
- `TensorFlow`
- `PyTorch`
- `Keras`

was used to train the models, create the classifiers, compute TF-IDF, or make predictions.

The project uses normal Python libraries for support tasks, such as:
- `pandas` — for reading CSV files
- `pickle` — for saving/loading trained models
- `Flask` — for the web application
- `nltk` PorterStemmer — for stemming

But the actual ML logic is manually implemented.

---

## ✨ Features

- ✅ Multinomial Naive Bayes classifier implemented from scratch
- ✅ Logistic Regression classifier implemented from scratch
- ✅ TF-IDF vectorization implemented from scratch
- ✅ Binary cross-entropy loss for Logistic Regression
- ✅ Gradient descent training for Logistic Regression
- ✅ Laplace smoothing for Naive Bayes
- ✅ Regex-based preprocessing for scam-related patterns
- ✅ Special tokens for URLs, emails, phone numbers, currency, and numbers
- ✅ Stopword removal
- ✅ Porter stemming
- ✅ Frequency-based vocabulary filtering
- ✅ Unigram and bigram feature extraction
- ✅ Confusion matrix evaluation
- ✅ Accuracy, precision, recall, and F1-score
- ✅ Manual challenge test set evaluation
- ✅ Logistic Regression uncertain mode
- ✅ Flask web interface
- ✅ Side-by-side model comparison in the app
- ✅ Displays processed words, used words, and ignored words
- ✅ Versioned model artifacts

---

## 📁 Project Structure

```
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

## 📊 Dataset

The project originally started with the classic SMS Spam Collection dataset from 2012. Later, it was upgraded to a newer SMS phishing/smishing dataset from 2022:

```
Dataset_5971.csv (current dataset)
```

### Dataset Contents

This dataset contains:
- Ham messages
- Spam messages
- Smishing messages
- URL indicator column
- Email indicator column
- Phone indicator column

### Label Normalization

For binary classification, labels are normalized as:

```
ham      → ham
spam     → spam
smishing → spam
```

So the final task remains:

```
ham vs spam
```

---

## 🔄 How the Project Works

The system has two main phases:

```
Training phase
    ↓
Saved model files (.pkl)
    ↓
Prediction phase through Flask app
```

The models are trained separately using training scripts. After training, the learned parameters are saved in `.pkl` files.

**The Flask app does NOT retrain models when a user submits a message.** It only loads the saved models and uses them for prediction.

---

## 📖 Core Files Explained

### 1. `dataset_loader.py`

This file loads the newer CSV dataset and converts it into a clean format used by the training scripts.

**Main responsibilities:**
- Read `Dataset_5971.csv`
- Normalize column names
- Normalize labels
- Remove duplicate messages
- Convert labels into binary format

**Example mapping:**
```
ham      → ham
spam     → spam
Smishing → spam
smishing → spam
```

The final output is a list of tuples:
```python
(message, label)
```

**Example:**
```python
("Your account has been blocked. Verify now.", "spam")
```

---

### 2. `naive_bayes_model.py`

This file contains reusable Naive Bayes prediction logic and the shared preprocessing function.

#### `preprocess(message)`

The preprocessing function prepares raw text before it enters the model.

**It performs:**
1. Lowercasing
2. Regex-based pattern matching
3. Special token replacement
4. Punctuation removal
5. Stopword removal
6. Stemming
7. Bigram creation

#### Regex-Based Pattern Matching

Before punctuation is removed, the system detects important scam-related patterns.

**Examples:**
```
spam.com              → urltoken
test@gmail.com        → emailtoken
+923001234567         → phonetoken
1000 dollars          → numbertoken currencytoken
1000$                 → numbertoken currencytoken
```

This matters because a naive punctuation-removal approach would turn:

```
spam.com
```

into:

```
spamcom
```

which loses the important URL structure.

#### Special Tokens

The following special tokens are used:
- `urltoken`
- `emailtoken`
- `phonetoken`
- `currencytoken`
- `numbertoken`

These help the model learn general scam indicators instead of memorizing exact URLs or phone numbers.

#### Bigrams

The model also creates bigrams.

**Example:**
```
I won the debate competition today
```

After preprocessing and stemming:
```
won debat competit today
```

Bigrams added:
```
won_debat
debat_competit
competit_today
```

This helps the model learn limited phrase-level context.

For example:
```
won_prize  → likely spam
won_debat  → likely ham
```

The models still use simple classical ML, but bigrams give them more context than single words alone.

---

### 3. `train_naive_bayes.py`

This is the earlier/baseline Naive Bayes training script.

It represents the original training flow before later dataset and preprocessing improvements.

---

### 4. `train_logistic_regression.py`

This is the earlier/baseline Logistic Regression training script.

It represents the original Logistic Regression implementation before later dataset and preprocessing improvements.

---

### 5. `train_naive_bayes_v2.py`

This script trains the newer Naive Bayes models using the newer dataset and improved preprocessing pipeline.

**Main responsibilities:**
- Load the newer dataset
- Shuffle data using a fixed random seed
- Split into train/test sets
- Preprocess messages
- Build vocabulary
- Apply frequency filtering using `MIN_FREQ`
- Count word frequencies for spam and ham
- Apply Laplace smoothing
- Evaluate using test data
- Save the trained model into the `models/` folder

The current v3 Naive Bayes model was produced after improvements such as:
- Regex pattern tokens
- Fixed currency regex
- Bigram features

---

### 6. `train_logistic_v2.py`

This script trains the newer Logistic Regression models using the newer dataset and improved preprocessing pipeline.

**Main responsibilities:**
- Load the newer dataset
- Preprocess messages
- Build vocabulary
- Compute IDF values
- Convert messages into TF-IDF vectors
- Train Logistic Regression using gradient descent
- Evaluate using test data
- Save the trained model into the `models/` folder

#### Logistic Regression Training

The model calculates:

```
z = w · x + b
```

Then applies sigmoid:

```
p = 1 / (1 + e^-z)
```

The model learns weights using gradient descent.

For each feature:

```
weight = weight - learning_rate × gradient
```

The loss function is binary cross-entropy:

```
loss = -[y log(p) + (1-y) log(1-p)]
```

The model learns θ parameters that maximize the probability of the training data.

Gradient descent iteratively updates weights to reduce loss.

---

### 7. `tfidf_vectorizer.py`

This file implements TF-IDF (Term Frequency - Inverse Document Frequency) from scratch.

**TF-IDF Formula:**

For a word `w` in a document:

```
IDF = log(total_documents / documents_containing_w)
TF = word_count / total_words_in_document
TF-IDF = TF × IDF
```

This gives higher weight to words that are:
- Frequent in the document (TF)
- Rare across all documents (IDF)

**Example:**

In spam classification:
- `free` appears in many documents → lower IDF
- `phishing` appears in few documents → higher IDF

The vectorizer:
- Builds a vocabulary from training data
- Counts word frequencies
- Computes IDF values
- Converts new messages into TF-IDF vectors for prediction

---

### 8. `app.py`

The Flask web application.

**Main responsibilities:**
- Load both trained models
- Render the web interface
- Handle user text input
- Preprocess the input message
- Make predictions using both models
- Display results side-by-side
- Show processed and ignored words

The app does not retrain models. It only performs inference using pre-trained `.pkl` files.

---

### 9. `evaluate_manual_tests.py`

This script evaluates both models on a manually written test set.

The manual test set contains carefully chosen edge cases, not the standard train/test split.

Results are displayed per model and overall.

---

## 📈 Version History

### Version 1 — Baseline Implementation

The original implementation using the classic 2012 SMS Spam Collection dataset.

This version introduced:
- Naive Bayes from scratch
- Logistic Regression from scratch
- TF-IDF vectorization from scratch
- Flask web app
- Basic preprocessing

### Version 2 — Upgraded Dataset and Preprocessing

Later, the dataset was upgraded to a modern SMS phishing/smishing dataset from 2022.

### Improvements
- Mapped smishing into spam
- Removed duplicate messages
- Trained newer Naive Bayes model
- Trained newer Logistic Regression model
- Saved v2 model artifacts

### Result

The models improved on modern spam/smishing examples.

However, preprocessing still had a weakness:

```
spam.com → spamcom
```

URLs, emails, phone numbers, and money amounts were not handled properly yet.

---

### Version 3 — Regex Tokens, Bigrams, and Improved Evaluation

Version 3 introduced the biggest preprocessing improvements.

#### Improvements
- Added regex-based pattern matching
- Replaced URLs with `urltoken`
- Replaced emails with `emailtoken`
- Replaced phone numbers with `phonetoken`
- Replaced currency expressions with `currencytoken`
- Replaced numbers with `numbertoken`
- Fixed currency regex bug where `rs` matched inside words like `university`
- Added bigram features
- Retrained Naive Bayes v3
- Retrained Logistic Regression v3
- Added manual challenge testing
- Added Logistic Regression uncertain mode

#### Why This Helped

Instead of memorizing exact URLs like:

```
secure-login.com
gift-claim.xyz
```

the model can learn the general signal:

```
urltoken
```

Instead of treating `1000$` as a broken token, the model sees:

```
numbertoken currencytoken
```

Bigrams also help capture phrase-level context:

```
free_prize
click_link
claim_reward
```

#### Result

Version 3 produced the strongest official dataset performance.

---

## 📊 Evaluation Results

### Official Test Split

The official train/test split uses the dataset itself and evaluates on held-out messages.

#### Naive Bayes v3

```
Accuracy : 0.9890
Precision: 0.9818
Recall   : 0.9609
F1 Score : 0.9712
```

#### Logistic Regression v3

```
Accuracy : 0.9835
Precision: 0.9707
Recall   : 0.9431
F1 Score : 0.9567
```

#### Interpretation

Naive Bayes v3 achieved the strongest official test-set performance.

Logistic Regression v3 is still useful because it provides probability scores and supports uncertain prediction.

---

### Manual Challenge Test Set

A separate manually written test set was created to evaluate real-world edge cases.

This test set is **not** used for training.

It contains:
```
15 hard ham
15 hard spam
5 easy ham
5 easy spam
```

#### Why This Test Exists

Official test metrics can be high while the model still fails on tricky messages.

Examples of hard ham:

```
I won the debate competition today.
Send me the Zoom link for class.
I will claim my certificate from the office tomorrow.
My phone number changed save this one.
```

These are normal messages, but they contain words that often appear in spam.

#### Naive Bayes v3 Manual Test

```
Correct: 29/40
Accuracy: 72.5%
```

Naive Bayes performed well on easy messages and most hard spam messages, but struggled with hard ham messages.

#### Logistic Regression v3 Manual Test with Uncertain Mode

```
Correct: 25/40
Wrong: 7/40
Uncertain: 8/40

Strict accuracy: 62.5%
Coverage: 80.0%
Confident accuracy: 78.1%
```

#### Interpretation

Uncertain mode reduced the number of confident wrong predictions, but it also abstained on some cases.

This is useful in the web app because some messages are genuinely borderline for a simple model.

---

## ⚠️ Known Limitations

This project is intentionally built using beginner-friendly classical ML models.

The models do not truly understand language meaning.

They may struggle with normal messages containing spam-like words:

```
won
claim
link
prize
phone
payment
rupees
dollars
```

**Examples:**

```
I won the debate competition today.
Please send me the lecture link.
I will claim my certificate tomorrow.
```

The model may still classify these as spam because these words are statistically associated with spam in the training data.

### Other Limitations

- Naive Bayes assumes feature independence
- Logistic Regression is still a linear model
- Word order is only partially captured through bigrams
- Rare phrases may be ignored due to vocabulary filtering
- The system may not generalize perfectly to unseen scam styles
- Manual challenge results are lower than official dataset metrics
- The app is a learning/demo project, not a production spam filter

---

## 💡 Why Naive Bayes Can Still Work

Naive Bayes assumes words are conditionally independent given the class.

This assumption is not fully true.

For example:

```
free prize winner
```

The words `free`, `prize`, and `winner` are related. They are not independent.

However, Naive Bayes can still classify text well because it does not need perfect probability estimates. It mainly needs the correct class score to be larger.

If words like:

```
claim
free
prize
winner
```

all strongly point toward spam, the spam score usually wins even if the exact probability is not perfectly calibrated.

This is why Naive Bayes often works surprisingly well for high-dimensional text classification.

---

## 🚀 Running the Project

### 1. Install Dependencies

```bash
pip install flask pandas nltk
```

### 2. Train Naive Bayes

```bash
python train_naive_bayes_v2.py
```

This creates or updates:

```
models/naive_bayes_v3.pkl
```

### 3. Train Logistic Regression

```bash
python train_logistic_v2.py
```

This creates or updates:

```
models/logistic_regression_v3.pkl
```

### 4. Run Manual Evaluation

```bash
python evaluate_manual_tests.py
```

This evaluates models on `manual_test_cases.csv`.

### 5. Run the Flask App

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 💬 Example Inputs

### Obvious Spam

```
Congratulations you have won 1000 dollars claim now.
```

**Expected:** `spam`

---

### Normal Ham

```
Are you coming to class today?
```

**Expected:** `ham`

---

### Hard Ham

```
I won the debate competition today.
```

**Expected:** `ham`

This type of message may still confuse the model because the word `won` is strongly associated with spam.

---

### Borderline Message

```
Please visit github.com and check my project repository.
```

**Expected:** `ham or uncertain`

The message contains a URL, which is suspicious in spam detection, but it can also be completely normal.

---

## 🎓 Learning Outcomes

This project demonstrates:

- How Naive Bayes works internally
- How Logistic Regression works internally
- How TF-IDF vectors are built
- How preprocessing affects model behavior
- How regex pattern extraction can improve text classification
- How bigrams add limited phrase context
- Why high dataset scores do not guarantee real-world reliability
- Why manual challenge testing matters
- How precision, recall, and F1-score reveal more than accuracy
- How uncertain predictions can reduce confident wrong decisions
- How Git versioning helps track ML experiments

---

## ✅ Final Project Status

This is a strong beginner ML project.

It includes:

```
✓ classical ML from scratch
✓ modern dataset
✓ preprocessing experiments
✓ feature engineering
✓ model comparison
✓ manual evaluation
✓ Flask app
✓ versioned improvements
✓ clear known limitations
```

**The project is not a production-grade spam filter.** It is a learning project designed to understand the full ML workflow from data to deployment-style interface.

