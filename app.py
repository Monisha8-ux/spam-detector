import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

threshold = 0.6

def predict_spam(text):
    text = preprocess(text)
    vector = tfidf.transform([text])
    prob = model.predict_proba(vector)[0][1]
    
    if prob > threshold:
        return f"🚨 Spam ({prob:.2f})"
    else:
        return f"✅ Not Spam ({prob:.2f})"

# UI
st.title("📧 Email Spam Detector")
st.write("Enter a message to check if it's Spam or Not")

user_input = st.text_area("Enter message:")

if st.button("Check"):
    if user_input.strip() != "":
        result = predict_spam(user_input)
        st.success(result)
    else:
        st.warning("Please enter a message")