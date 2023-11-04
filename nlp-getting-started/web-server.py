from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)


class PreTrainedModel:
    def __init__(self):
        self.load_model()
        self.load_vectorizer()

    def load_model(self):
        with open('model.bin', 'rb') as f_in:
            model: pickle.load(f_in)
        self.model = model

    def load_vectorizer(self):
        with open('vectorizer.bin', 'rb') as f_in:
            vectorizer: pickle.load(f_in)
        self.vectorizer = vectorizer

    def get_prediction_probability(self, data: dict) -> float:
        pass


@app.route("/")
def hello_world():
    return jsonify({
        'project_name': 'Disaster Tweets Detection',
        'project_description': 'The Aim of this project is to detect whether a tweet is announcing a disaster of not',
        'more_info': 'https://github.com/shayansm2/kaggle_playground/tree/main/nlp-getting-started'
    })


@app.post("/api/disaster-predict/")
def get_disaster_prediction():
    data = request.get_json()
    prob = PreTrainedModel().get_prediction_probability(data)
    result = {
        'probability': prob,
        'status': 'disaster' if prob >= 0.5 else 'non-disaster'
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
