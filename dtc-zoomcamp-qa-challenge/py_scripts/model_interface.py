import numpy as np
from tensorflow.nn import softmax
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained('model.h5')

def predict(question: str, answer: str) -> bool:
    tokens = tokenizer(question, answer, padding=True, truncation=True, return_tensors="tf")
    res = model(**tokens)
    return np.argmax(res[0]).astype(bool)

def predict_probabilities(question: str, answer: str) -> list:
    tokens = tokenizer(question, answer, padding=True, truncation=True, return_tensors="tf")
    res = model(**tokens)
    return np.array(softmax(res.logits)).tolist()[0]

def ask_question(question: str, number_of_answers: int) -> list|str:
    pass
