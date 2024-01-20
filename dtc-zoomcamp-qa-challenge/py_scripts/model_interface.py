import numpy as np
import pandas as pd
from tensorflow.nn import softmax
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained('model.h5')
answers = None

def predict(question: str, answer: str) -> bool:
    tokens = tokenizer(question, answer, padding=True, truncation=True, return_tensors="tf")
    res = model(**tokens)
    return np.argmax(res[0]).astype(bool)

def predict_probabilities(question: str, answer: str) -> list:
    tokens = tokenizer(question, answer, padding=True, truncation=True, return_tensors="tf")
    res = model(**tokens)
    return np.array(softmax(res.logits)).tolist()[0]

def ask_question(question: str, course_name: str = None, max_num_answers: int = 5) -> list|str:
    if answers is None:
        load_answers()
    
    proper_answers = answers.copy()
    if course_name is not None:
        proper_answers = proper_answers[proper_answers['course'] == course_name]

    if len(course_name) == 0:
        return []
    
    max_num_answers = min(max_num_answers, len(course_name))

    def get_pred(q, ans) -> int :
        return predict_probabilities(q, ans)[1]

    proper_answers['prob'] = proper_answers.apply(lambda row: get_pred(question, row['answer']), axis=1)

    return proper_answers.sort_values(by='prob', ascending=False)['answer'].tolist()[:max_num_answers]


def load_answers():
    train_answers = pd.read_csv('data/train_answers.csv')
    test_answers = pd.read_csv('data/test_answers.csv')
    global answers
    answers = pd.concat([train_answers, test_answers])
