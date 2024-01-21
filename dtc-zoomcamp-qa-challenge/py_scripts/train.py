import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers.legacy import Adam # as tf.keras.optimizers.Adam runs slowly on M1/M2 Macs, I am using legacy Keras optimizer instead

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset, DatasetDict

from py_scripts.data_prepration import get_training_set_data


def train_model(checkpoint = "bert-base-uncased", seed=11):
    df = get_training_set_data(seed=seed)
    df.index = df.index.rename('idx')
    df.drop(columns=['question_row_id', 'answer_row_id', 'question_id', 'answer_id'], inplace=True)

    train, validation = train_test_split(df, test_size=0.2, random_state=seed)
    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train),
        'validation': Dataset.from_pandas(validation)
    })

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(record):
        return tokenizer(record["question"], record["answer"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
