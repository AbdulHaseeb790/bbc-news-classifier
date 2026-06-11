import streamlit as st
import pickle

model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

st.title("🗞️ News Analyzer")
st.write("Enter any news headline or text to analyze it!")

text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if text:
        st.subheader("Results")
        text_tfidf = tfidf.transform([text])
        category = model.predict(text_tfidf)[0]
        st.write("📰 **Category:**", category)
    else:
        st.warning("Please enter some text first!")