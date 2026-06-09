import sys
import os

# ✅ MUST be first (before any local imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from predict import predict_news


st.set_page_config(page_title="News Classifier", layout="centered")

st.title("📰 BERT News Topic Classifier")
st.write("Enter a news headline and get its category prediction.")

text = st.text_area("Enter News Headline")

if st.button("Predict"):

    if text.strip():

        with st.spinner("Predicting..."):
            result = predict_news(text)

        st.success(f"Predicted Category: {result}")

    else:
        st.warning("Please enter a valid headline")

st.markdown("---")
st.caption("Fine-tuned BERT Model • News Classification System")
st.caption("Built by Haris Hussain")