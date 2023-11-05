from flask import Flask, request, jsonify
import pickle
from models import ModelInterface
from input_providers import InputProvider5
from pandas import DataFrame

app = Flask(__name__)


class PreTrainedModel(object):
    def __init__(self):
        self.model: ModelInterface = self.load_model()
        self.input_provider: InputProvider5 = self.load_input_provider()

    def load_model(self) -> ModelInterface:
        with open('model.bin', 'rb') as f_in:
            model = pickle.load(f_in)
        return model

    def load_input_provider(self) -> InputProvider5:
        with open('input_provider.bin', 'rb') as f_in:
            input_provider = pickle.load(f_in)
        return input_provider

    def get_prediction_probability(self, data: dict) -> float:
        df = DataFrame(data, index=[0])
        x = self.input_provider.get_input(df)
        return self.model.predict_proba(x)[:, 1][0]


@app.route("/")
def hello_world():
    return jsonify({
        'project_name': 'Disaster Tweets Detection',
        'project_description': 'The aim of this project is to detect whether a tweet is announcing a disaster of not',
        'more_info': 'https://github.com/shayansm2/kaggle_playground/tree/main/nlp-getting-started'
    })


@app.post("/api/disaster-predict/")
def get_disaster_prediction():
    try:
        prob = PreTrainedModel().get_prediction_probability(prepare_data())
    except AssertionError as error:
        return jsonify({
            'error': error.args[0]
        })
    return jsonify({
        'probability': round(prob, 3),
        'status': 'disaster' if prob >= 0.5 else 'non-disaster'
    })


def prepare_data():
    data = request.get_json()
    assert 'text' in data, 'text is not provided'
    if 'keyword' not in data:
        data['keyword'] = None
    if 'location' not in data:
        data['location'] = None
    return data


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1234)
