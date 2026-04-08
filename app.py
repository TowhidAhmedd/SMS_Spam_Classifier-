import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ✅ Setup NLTK properly (IMPORTANT)
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)

# Download required resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()
import nltk
import shutil
import os

try:
    stop_words = set(stopwords.words('english'))
except AttributeError:
    # যদি এরর দেয়, তবে ম্যানুয়ালি পাথ থেকে লোড করার চেষ্টা করুন
    nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    
    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords & punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)

#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])   # ❗ .toarray() remove

#     # ✅ DEBUG (এখানে add করো)
#     st.write("Vectorizer features:", len(tfidf.get_feature_names_out()))
#     st.write("Model expects:", model.n_features_in_)
#     st.write("Input shape:", vector_input.shape)

#     # 3. predict
#     result = model.predict(vector_input)[0]

#     # 4. display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")


if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray() 

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")