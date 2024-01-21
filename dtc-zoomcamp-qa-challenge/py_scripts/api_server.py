from flask import Flask, request, jsonify
from py_scripts.model_interface import *

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello_world():
    return jsonify({
        'project_name': 'DTC Zoomcamp Q&A Challenge',
        'project_description': 'The aim of this project is to answer a zommcamp question based on zommcamps in ML, MLOPS and DE engineering',
        'more_info': 'https://github.com/shayansm2/kaggle-playground/tree/main/dtc-zoomcamp-qa-challenge'
    })


@app.route("/predict/", methods=['GET'])
def predict():
    params = request.args
    if params.get('question') is None:
        return jsonify({'error': 'question is not provided in the request'})
    if params.get('answer') is None:
        return jsonify({'error': 'answer is not provided in the request'})
    
    probs = predict_probabilities(params.get('question'), params.get('answer'))
    return jsonify({
        'probability': round(probs[1], 3),
        'flag': 'related' if probs[1] > 0.5 else 'unrelated'
    })

@app.route("/ask/", methods=['GET'])
def ask():
    params = request.args
    if params.get('question') is None:
        return jsonify({'error': 'question is not provided in the request'})

    return jsonify({
        'answers': ask_question(params.get('question'), params.get('course_name'), params.get('max_num_answers'))
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5678)