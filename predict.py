import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.settings import MODEL_PATH, LABEL_MAP


# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ✅ Load model ONLY ONCE using Streamlit cache (fixes slow/blank screen issue)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    return tokenizer, model


# Load tokenizer + model globally (cached)
tokenizer, model = load_model()


# ✅ Prediction function (used by app.py)
def predict_news(text):

    # Convert text into tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Move tensors to device (CPU/GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # No training, only inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted class index
    pred = torch.argmax(outputs.logits, dim=1).item()

    # Return readable label
    return LABEL_MAP[pred]