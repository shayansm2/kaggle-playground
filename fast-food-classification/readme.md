## Title of the project

### Problem Description

This project is dedicated to developing a deep learning model for the classification of fast food images using
Keras. you can find and download the data which was being used in order to train the model can be found htere:
> data source: https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset


the overall structure of the project is as follows:
![ML capstone project 1 schema.png](ML%20capstone%20project%201%20schema.png)

### EDA

- [eda](jupyter-files/eda.ipynb)

### Model Training

- [base models analysis](jupyter-files/base-models-analysis.ipynb)
- [transfer learning](jupyter-files/transfer-learning.ipynb)
- [hyperparameter tunning](jupyter-files/hyperparameter-tunning.ipynb)

### Script files

- [gateway](py-scripts%2Fgateway.py)

### Installation and Deployment

follow the following steps in order to install and deploy the model

```commandline
git clone https://github.com/shayansm2/kaggle-playground.git
cd fast-food-classification
pip install pipenv
pipenv install
pipenv shell
```

```commandline
docker-compose -f ./deployment/docker-compose.yml up
docker-compose start
```

### Containerization

- [deployment codes](deployment)
- [kuber configz](kube-config)

evaluation source:
> https://docs.google.com/spreadsheets/d/e/2PACX-1vQCwqAtkjl07MTW-SxWUK9GUvMQ3Pv_fF8UadcuIYLgHa0PlNu9BRWtfLgivI8xSCncQs82HDwGXSm3/pubhtml
