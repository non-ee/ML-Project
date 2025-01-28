from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, **kwargs):
    model = XGBClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, kernel='linear', **kwargs):
    model = SVC(kernel=kernel, **kwargs)
    model.fit(X_train, y_train)
    return model