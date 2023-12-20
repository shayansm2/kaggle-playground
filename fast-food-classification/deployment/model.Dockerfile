FROM tensorflow/serving:2.7.0
LABEL authors="shayan"
COPY clothing-model /models/clothing-model/1
ENV MODEL_NAME="clothing-model"