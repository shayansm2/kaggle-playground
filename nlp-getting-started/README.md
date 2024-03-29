# Disaster Tweets Detection

## Problem Description:

The Aim of this project is to detect whether a tweet is announcing a disaster of not. The data is based
on [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/)
competition in [Kaggle](https://www.kaggle.com/). Twitter has become an important communication channel in times of
emergency.But, it’s not always clear whether a person’s words are actually announcing a disaster.
In this project a machine learning model is build in order to detect this issue. This problem is very similar to smap
detection problem. Some NLP definitions and techniques will also be introduced in this project.

### Input Data

| column name | type    | metadata                                                  |
|-------------|---------|-----------------------------------------------------------|
| id          | integer |                                                           |
| keyword     | string  | 222 unique values, <br> has missing values (less than 1%) |
| location    | string  | has missing values (49%)                                  |
| text        | string  | no missing value                                          |
| target      | boolean | no missing value                                          |

### Exposed API Schema

```http request
POST /api/disaster-predict/ HTTP/1.1
Host: localhost:1234
Content-Type: application/json

{
    "text": "Christian Attacked by Muslims at the Temple Mount after Waving Israeli Flag via Pamela Geller - ... http://t.co/OGoyzOlJk5",
    "keyword": "attacked",
    "location": "Revolutionary Road, USA"
}
```

response:

```json
{
  "probability": 0.8794235284958574,
  "status": "disaster"
}
```

in order to use the model

1. build the image from Dockerfile

```commandline
docker build -t twitter_disaster_detection ./
```

2. build a container from the image and run it

```commandline
docker run -it -p 1234:1234 twitter_disaster_detection
```

## All steps from discovering data to building an api for disaster detection

1. [EDA](./eda.ipynb)
2. [Feature extraction](./feature-eng.ipynb)
3. [Model training and validation](./models.ipynb)
4. [Hyper-parameter tuning](./hyper-parameter.ipynb)
5. [Web service](./web_server.py)
6. [Containerization](./Dockerfile)

## Sources

- [competition link](https://www.kaggle.com/competitions/nlp-getting-started/)
- [a notebook learning NLP](https://www.kaggle.com/code/faressayah/natural-language-processing-nlp-for-beginners#%F0%9F%94%81-Representing-text-as-numerical-data)
- [another notebook learning NLP](https://www.kaggle.com/code/philculliton/nlp-getting-started-tutorial/notebook)

[//]: # (add this to kaggle note book too and mention it here)