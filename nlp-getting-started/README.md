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
Host: localhost
Content-Type: application/json

{
    "keyword": "ablaze",
    "location": "USA",
    "text": "INEC Office in Abia Set Ablaze - http://t.co/3ImaomknnA"
}
```

## steps, links, contents,

- [EDA](./eda.ipynb)
- [feature extraction](./feature-eng.ipynb)
- [Model training and validation](./models.ipynb)
- hyper-parameter tuning
- Web service

## Sources

- [competition link](https://www.kaggle.com/competitions/nlp-getting-started/)
- [a notebook learning NLP](https://www.kaggle.com/code/faressayah/natural-language-processing-nlp-for-beginners#%F0%9F%94%81-Representing-text-as-numerical-data)
- [another notebook learning NLP](https://www.kaggle.com/code/philculliton/nlp-getting-started-tutorial/notebook)
- [Getting started with natural language processing (microsoft)](https://microsoft.github.io/ML-For-Beginners/#/6-NLP/README?id=lessons)

[//]: # (add this to kaggle note book too and mention it here)