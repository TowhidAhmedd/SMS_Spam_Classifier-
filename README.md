# SMS Spam Classifier (End-to-End ML Project)
 Overview

This project is a Machine Learning based SMS Spam Classifier that detects whether a given message is Spam or Ham (Not Spam) using Natural Language Processing (NLP) techniques.

The model is trained on labeled SMS data and uses text preprocessing + feature extraction to make predictions.

 Problem Statement  

SMS spam messages are a major issue in communication systems. The goal is to automatically classify messages into:

🟢 Ham (Normal message)  
🔴 Spam (Unwanted message)

Approach

1. Data Collection
Used SMS dataset containing labeled messages (ham/spam)

3. Data Preprocessing (NLP)
Lowercasing text
Removing punctuation
Removing stopwords
Tokenization
Stemming

5. Feature Engineering
Converted text into numerical vectors using:
TF-IDF Vectorizer / CountVectorizer

7. Model Training
Machine Learning models used:
Naive Bayes (Best performing)
Logistic Regression (optional comparison)

9. Evaluation
Accuracy Score
Precision, Recall, F1-score
Confusion Matrix

# Results
## Accuracy: 97.06%  
## Confusion_Matrix: False Positives = 0 And False Negatives = 30  
## Precision: 100%  
## Best Model: Naive Bayes  

## Tech Stack
Python 🐍
Pandas, NumPy
Scikit-learn
NLTK
Streamlit


SMS_Spam_Classifier  
│
├── dataset                 # SMS dataset  
├── model.pkl               # Trained ML model  
├── vectorizer.pkl          # TF-IDF/CountVectorizer  
├── app.py                  # Streamlit app    
├── train_model.ipynb       # Model training google colab   
├── requirements.txt        # Dependencies  
└── README.md               # Project documentation  


 Installation
git clone https://github.com/TowhidAhmedd/SMS_Spam_Classifier-.git  
cd SMS_Spam_Classifier  
pip install -r requirements.txt  

How to Run (Google Colab)
 Option 1: Run Notebook in Google Colab (Recommended)  
Open Google Colab: https://colab.research.google.com  
Clone the repository: !git clone https://github.com/TowhidAhmedd/SMS_Spam_Classifier-.git  
Move into project folder: %cd SMS_Spam_Classifier-  
Install dependencies:  
!pip install -r requirements.txt  
Open and run your notebook:  
from google.colab import drive  
drive.mount('/content/drive')  

OR directly open .ipynb file from the repo and run all cells.  
  
Example Prediction


Message	Prediction
"Congratulations! You won $1000"	Spam 🔴
"Are you coming to class today?"	Ham 🟢
