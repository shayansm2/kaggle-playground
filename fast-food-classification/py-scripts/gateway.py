import os
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

from protobuf import np_to_protobuf

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor('xception', target_size=(299, 299))


def prepare_request(x):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(x))
    return pb_request


classes = ['Baked Potato', 'Burger', 'Crispy Chicken', 'Donut', 'Fries',
           'Hot Dog', 'Pizza', 'Sandwich', 'Taco', 'Taquito']


def prepare_response(pb_response):
    return dict(zip(classes, pb_response.outputs['dense_7'].float_val))


def predict(url):
    return prepare_response(
        stub.Predict(
            prepare_request(preprocessor.from_url(url)),
            timeout=20.0
        )
    )


app = Flask('gateway')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
