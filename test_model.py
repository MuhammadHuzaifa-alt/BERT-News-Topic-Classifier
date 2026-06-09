print("STEP 1: Script started")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/fine_tuned_bert_model_on_news"

print("STEP 2: Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

print("STEP 3: Tokenizer loaded")

print("STEP 4: Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("STEP 5: Model loaded successfully")

print("ALL DONE ✔")
