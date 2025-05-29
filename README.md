# Twitter Sentiment Analysis

This project applies Natural Language Processing (NLP) techniques and machine learning to classify the sentiment of tweets as **positive** or **negative**. It uses the **Sentiment140** dataset containing 1.6 million labeled tweets and implements a **Logistic Regression** model for binary sentiment classification.

---

## Project Overview

- **Objective**: Develop a machine learning pipeline to classify the sentiment of tweets using text preprocessing and logistic regression.
- **Dataset**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (sourced from Kaggle)
- **Model**: Logistic Regression

---

## Workflow

1. **Data Collection**
   - Dataset downloaded using the Kaggle API.

2. **Data Preprocessing**
   - Removed Twitter handles, URLs, punctuation, and special characters using regular expressions.
   - Converted all text to lowercase.
   - Tokenized the text using NLTK.
   - Removed common stopwords.
   - Applied stemming using the Porter Stemmer.

3. **Feature Extraction**
   - Converted preprocessed text into numerical feature vectors using `TfidfVectorizer`.

4. **Model Training**
   - Split the dataset into training and testing sets using `train_test_split`.
   - Trained a `LogisticRegression` classifier on the TF-IDF vectors.

5. **Model Evaluation**
   - Evaluated model performance using accuracy score.
   - Additional metrics (precision, recall, F1-score) can be integrated for more in-depth analysis.

---

## Tools and Libraries Used

| Library         | Description                                         |
|-----------------|-----------------------------------------------------|
| Python          | Programming language used for implementation       |
| Pandas          | Data loading and manipulation                      |
| NumPy           | Numerical computations                             |
| NLTK            | Text preprocessing (tokenization, stopword removal, stemming) |
| re              | Regular expressions for text cleaning              |
| Scikit-learn    | Model training, vectorization, evaluation          |
| Kaggle API      | Dataset download and access                        |

---

## Results

- **Model Used**: Logistic Regression
- **Feature Representation**: TF-IDF vectors
- **Evaluation Metric**: Accuracy Score
- **Performance**: The model achieved reasonable classification accuracy (typically around 80%, depending on preprocessing and data split)

