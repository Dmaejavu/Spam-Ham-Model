# app.py
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/knn_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# App UI
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check if it's **spam** or **ham**.")

# User input
user_input = st.text_area("Your message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        text_vector = vectorizer.transform([user_input.lower()])
        prediction = model.predict(text_vector)[0]
        if prediction == "spam":
            st.error("🚨 This message is SPAM.")
        else:
            st.success("✅ This message is HAM (not spam).")
