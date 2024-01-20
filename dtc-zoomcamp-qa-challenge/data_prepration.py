import pandas as pd
import numpy as np


def get_training_set_data(seed = 22):
    questions = pd.read_csv('data/train_questions.csv')
    answers = pd.read_csv('data/train_answers.csv')

    number_of_qa = len(questions)
    range_ids = np.arange(number_of_qa)
    col1, col2 = np.meshgrid(range_ids, range_ids)
    df = pd.DataFrame({'question_row_id': col1.flatten(), 'answer_row_id': col2.flatten()})

    df['question_id'] = df['question_row_id'].map(questions['question_id'])
    df['question'] = df['question_row_id'].map(questions['question'])
    df['answer_id'] = df['answer_row_id'].map(answers['answer_id'])
    df['answer'] = df['answer_row_id'].map(answers['answer'])

    def get_label(question_id: int, answer_id: int) -> int:
        if answer_id == questions[questions['question_id'] == question_id]['answer_id'].values[0]:
            return 1
        if answer_id in list(map(int,questions[questions['question_id'] == question_id]['candidate_answers'].values[0].split(","))):
            return 1
        return 0
    
    df['label'] = df.apply(lambda row: get_label(row['question_id'], row['answer_id']), axis=1)

    positive_samples = df[df['label'] == 1]
    num_negative_samples = int(0.015 * len(df[df['label'] == 0]))
    negative_samples = df[df['label'] == 0].sample(n=num_negative_samples, random_state=seed)
    training_set = pd.concat([positive_samples, negative_samples])
    training_set = training_set.sample(frac=1, random_state=seed)
    return training_set
