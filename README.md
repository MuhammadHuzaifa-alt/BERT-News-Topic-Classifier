# BERT News Topic Classifier

## Overview
This project fine-tunes a BERT model to classify news headlines into four categories using the AG News dataset.
The model is trained using Hugging Face Transformers and PyTorch, then deployed through a Streamlit web application for real-time predictions.
---

## Features

✅ Fine-tuned BERT (bert-base-uncased)

✅ AG News Dataset

✅ Multi-Class Text Classification

✅ Accuracy and F1-Score Evaluation

✅ PyTorch Backend

✅ Hugging Face Transformers

✅ Streamlit Deployment

✅ GPU Support (CUDA)

---

## News Categories

| Label | Category |
| ----- | -------- |
| 0     | World    |
| 1     | Sports   |
| 2     | Business |
| 3     | Sci/Tech |

Example:

```text
Input:
Apple launches new AI powered iPhone

Prediction:
Sci/Tech
```

---

## Dataset

Dataset Used:

AG News Dataset

Contains approximately:

* 120,000 Training Samples
* 7,600 Test Samples

Each news article belongs to one of four categories:

* World
* Sports
* Business
* Sci/Tech

Dataset loaded using:

```python
from datasets import load_dataset

dataset = load_dataset("ag_news")
```

---

## Project Workflow

### 1. Load Dataset

```python
dataset = load_dataset("ag_news")
```

---

### 2. Tokenization

Tokenizer:

```python
BertTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

Text preprocessing:

```python
tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128
)
```

---

### 3. Model Selection

Pretrained Model:

```python
bert-base-uncased
```

Loaded as:

```python
BertForSequenceClassification(
    num_labels=4
)
```

Architecture:

* 12 Transformer Layers
* Hidden Size: 768
* 12 Attention Heads
* Maximum Sequence Length: 512

---

### 4. Fine-Tuning

Training configuration:

```python
TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)
```

---

### 5. Model Evaluation

Metrics Used:

* Accuracy
* F1 Score (Weighted)

Implementation:

```python
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)

    f1 = f1_score(
        labels,
        preds,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }
```

---

## Model Training

Trainer API used:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)
```

Training:

```python
trainer.train()
```

---

## Model Saving

After training:

```python
trainer.save_model(save_path)

tokenizer.save_pretrained(save_path)
```

Saved Directory:

```text
fine_tuned_bert_model_on_news/
```

Contains:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

---

## Prediction Pipeline

Example:

```python
predict_news(
    "Apple launches new AI powered iPhone"
)
```

Output:

```text
Sci/Tech
```

More Examples:

```text
Manchester United wins Champions League final
→ Sports

Scientists discover new quantum computing breakthrough
→ Sci/Tech

Stock market reaches record high
→ Business

Global leaders meet for climate summit
→ World
```

---

## Streamlit Web Application

The project includes a Streamlit frontend for live predictions.

Features:

* User-friendly interface
* Real-time predictions
* Cached model loading
* Fast inference

Run application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
BERT-News-Classifier/
│
├── app/
│   └── app.py
│
├── model/
│   └── fine_tuned_bert_model_on_news/
│
├── config/
│   └── settings.py
│
├── predict.py
├── train.py
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## Installation

Clone Repository

```bash
git clone https://github.com/yourusername/bert-news-classifier.git

cd bert-news-classifier
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Main Libraries:

```text
torch
transformers
datasets
scikit-learn
numpy
pandas
streamlit
```

---

## Run Training

```bash
python train.py
```

---

## Run Prediction

```bash
python predict.py
```

---

## Run Streamlit Application

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

## Results

Evaluation Metrics:

* Accuracy: High classification performance on AG News dataset
* F1 Score: Balanced performance across all classes

The fine-tuned BERT model successfully classifies unseen news headlines into the correct category.

---

## Future Improvements

* DistilBERT for faster inference
* RoBERTa comparison
* Hyperparameter optimization
* Model deployment on Hugging Face Spaces
* Docker containerization
* REST API using FastAPI
* News article classification instead of headlines only

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* AG News Dataset
* Streamlit
* Scikit-Learn
* NumPy
* Pandas

---

## Author

Muhammad Huzaifa

Machine Learning | Deep Learning | Natural Language Processing (NLP)

---

## License

This project is licensed under the MIT License.
