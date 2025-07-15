import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Text preprocessing function
def preprocess_text(text):
    # Remove links, mentions, hashtags, and non-alphanumeric characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    return text.lower().strip()

# Streamlit UI
st.set_page_config(page_title="Spam Mail Classifier", page_icon="📩")
st.title("📩 Spam Mail Prediction using Machine Learning")
st.markdown("This app classifies emails as **Spam** or **Not Spam**.")

# Text input
user_input = st.text_area("✉️ Enter the email content below:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        # cleaned_text = preprocess_text(user_input)
        # vectorized_input = vectorizer.transform([cleaned_text])
        vectorized_input = vectorizer.transform([user_input])

        
        # Predict class and probability
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        confidence = max(probabilities) * 100

        if prediction == 1:
            st.error(f"🚫 This message is **SPAM**.")
        else:
            st.success(f"✅ This message is **NOT SPAM**.")

        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
