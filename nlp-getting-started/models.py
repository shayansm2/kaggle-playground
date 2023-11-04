import abc
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class ModelInterface:
    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def predict_proba(self, x):
        pass


class LogisticRegressionModel(LogisticRegression, ModelInterface):
    pass


class DecisionTreeModel(DecisionTreeClassifier, ModelInterface):
    pass


class RandomForestModel(RandomForestClassifier, ModelInterface):
    pass


class GradientBoostingModel(GradientBoostingClassifier, ModelInterface):
    pass


class XGBoostModel(ModelInterface):
    def __init__(self):
        self.xgb_model = None
        self.xgb_params = {
            'eta': 0.3,
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'nthread': 8,
            'silent': 1,
            'eval_metric': 'auc'
        }

    def fit(self, x, y):
        d_train = xgb.DMatrix(x, y)
        self.xgb_model = xgb.train(self.xgb_params, d_train)

    def predict(self, x):
        d_val = xgb.DMatrix(x)
        return self.xgb_model.predict(d_val) >= 0.5

    def predict_proba(self, x):
        d_val = xgb.DMatrix(x)
        prob = self.xgb_model.predict(d_val)
        return np.column_stack((1 - prob, prob))

    def set_hyper_parameter(self, name: str, value):
        self.xgb_params[name] = value
        return self
