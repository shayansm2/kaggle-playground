import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack


def get_clean_text(mess: str):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    nopunc = nopunc.lower().strip()

    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


def add_new_features_from_text(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()
    df['words_count'] = df.text.apply(len)

    df['has_location'] = df['location'].notnull().astype(int)
    del df['location']
    df['has_question_mark'] = df['text'].str.contains('\?').astype(int)
    df['has_exclamation_mark'] = df['text'].str.contains('\!').astype(int)
    df['has_hashtag'] = df['text'].str.contains('\#').astype(int)
    df['has_capital_words'] = df['text'].apply(lambda x: str(x).isupper()).astype(int)
    df['has_link'] = df['text'].str.contains(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+').astype(int)

    return df


def add_clean_text_features(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()
    df['clean_text'] = df['text'].apply(get_clean_text)
    df['clean_words_count'] = df['clean_text'].apply(len)
    return df


class InputProviderInterface(object):
    def get_train_inputs(self, df: pd.DataFrame) -> tuple:
        pass

    def get_test_inputs(self, df: pd.DataFrame) -> tuple:
        pass


class InputProvider1(InputProviderInterface):
    @staticmethod
    def _get_input_base(df: pd.DataFrame) -> tuple:
        df = add_new_features_from_text(df)
        y = df.target
        df.drop(columns=['id', 'text', 'keyword', 'target'], inplace=True)
        x = df.values
        return x, y

    def get_train_inputs(self, df: pd.DataFrame) -> tuple:
        return self._get_input_base(df)

    def get_test_inputs(self, df: pd.DataFrame) -> tuple:
        return self._get_input_base(df)


class InputProvider2(InputProviderInterface):
    def __init__(self):
        self.vect = DictVectorizer()

    def get_train_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_new_features_from_text(df)
        y = df.target
        df.drop(columns=['id', 'text', 'target'], inplace=True)
        df['keyword'].fillna('', inplace=True)
        self.vect.fit(df.to_dict(orient='records'))
        x = self.vect.transform(df.to_dict(orient='records'))
        return x, y

    def get_test_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_new_features_from_text(df)
        y = df.target
        df.drop(columns=['id', 'text', 'target'], inplace=True)
        df.fillna(0)
        x = self.vect.transform(df.to_dict(orient='records'))
        return x, y


class InputProvider3(InputProviderInterface):
    def __init__(self):
        self.vect = CountVectorizer()

    def get_train_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_clean_text_features(df)
        y = df.target
        self.vect.fit(df['clean_text'])
        x = self.vect.transform(df['clean_text'])
        return x, y

    def get_test_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_clean_text_features(df)
        y = df.target
        x = self.vect.transform(df['clean_text'])
        return x, y


class InputProvider4(InputProviderInterface):
    def __init__(self):
        self.vect = CountVectorizer()

    def get_train_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_new_features_from_text(add_clean_text_features(df))
        y = df.target
        self.vect.fit(df['clean_text'])
        tokens = self.vect.transform(df['clean_text'])
        sparce_features = csr_matrix(df.drop(columns=['text', 'clean_text', 'keyword', 'id', 'target']).values)
        x = hstack([tokens, sparce_features])
        return x, y

    def get_test_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_new_features_from_text(add_clean_text_features(df))
        y = df.target
        tokens = self.vect.transform(df['clean_text'])
        sparce_features = csr_matrix(df.drop(columns=['text', 'clean_text', 'keyword', 'id', 'target']).values)
        x = hstack([tokens, sparce_features])
        return x, y


class InputProvider5(InputProviderInterface):
    def __init__(self):
        self.count_vect = CountVectorizer()
        self.dict_vect = DictVectorizer()

    def get_train_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_new_features_from_text(add_clean_text_features(df))
        y = df.target
        self.count_vect.fit(df['clean_text'])
        tokens = self.count_vect.transform(df['clean_text'])
        df.drop(columns=['text', 'clean_text', 'id', 'target'], inplace=True)
        df['keyword'].fillna('', inplace=True)
        self.dict_vect.fit(df.to_dict(orient='records'))
        features = self.dict_vect.transform(df.to_dict(orient='records'))
        sparce_features = csr_matrix(features)
        x = hstack([tokens, sparce_features])
        return x, y

    def get_test_inputs(self, df: pd.DataFrame) -> tuple:
        df = add_new_features_from_text(add_clean_text_features(df))
        y = df.target
        tokens = self.count_vect.transform(df['clean_text'])
        df.drop(columns=['text', 'clean_text', 'id', 'target'], inplace=True)
        features = self.dict_vect.transform(df.to_dict(orient='records'))
        sparce_features = csr_matrix(features)
        x = hstack([tokens, sparce_features])
        return x, y
