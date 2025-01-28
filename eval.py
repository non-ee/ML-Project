import pickle
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import os

with open('data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# Load models
logistic_model = joblib.load('models/logistic_reg.pkl')
random_forest_model = joblib.load('models/random_forest.pkl')
xgboost_model = joblib.load('models/xgboost.pkl')
svm_linear_model = joblib.load('models/svm_linear.pkl')
svm_rbf_model = joblib.load('models/svm_rbf.pkl')

# Make predictions
logistic_pred = logistic_model.predict(X_test)
random_forest_pred = random_forest_model.predict(X_test)
xgboost_pred = xgboost_model.predict(X_test)
svm_linear_pred = svm_linear_model.predict(X_test)
svm_rbf_pred = svm_rbf_model.predict(X_test)

# Evaluate accuracy
logistic_acc = accuracy_score(y_test, logistic_pred)
random_forest_acc = accuracy_score(y_test, random_forest_pred)
xgboost_acc = accuracy_score(y_test, xgboost_pred)
svm_linear_acc = accuracy_score(y_test, svm_linear_pred)
svm_rbf_acc = accuracy_score(y_test, svm_rbf_pred)

print("====================================================")
print("Evaluate Accuracy")
print("Logistic Regression Accuracy:", logistic_acc)
print("Random Forest Accuracy:", random_forest_acc)
print("XGBoost Accuracy:", xgboost_acc)
print("SVM Linear Accuracy:", svm_linear_acc)
print("SVM RBF Accuracy:", svm_rbf_acc)
print("====================================================")

# Evaluate F1-score
AVERAGE = 'macro'
logistic_f1 = f1_score(y_test, logistic_pred, average=AVERAGE)
random_forest_f1 = f1_score(y_test, random_forest_pred, average=AVERAGE)
xgboost_f1 = f1_score(y_test, xgboost_pred, average=AVERAGE)
svm_linear_f1 = f1_score(y_test, svm_linear_pred, average=AVERAGE)
svm_rbf_f1 = f1_score(y_test, svm_rbf_pred, average=AVERAGE)

print("====================================================")
print("Evaluate F1-score")
print("Logistic Regression F1-score:", logistic_f1)
print("Random Forest F1-score:", random_forest_f1)
print("XGBoost F1-score:", xgboost_f1)
print("SVM Linear F1-score:", svm_linear_f1)
print("SVM RBF F1-score:", svm_rbf_f1)
print("====================================================")

file_name = "results_2.csv"
Sensor = "light"

if not os.path.isfile(file_name):
    with open(file_name, 'w') as f:
        f.write('sensors,metrics,logistic,rf,xgboost,svm_linear,svm_rbf\n')

with open(file_name, 'a') as f:
    # f.write('{},accuracy,{},{},{},{},{}\n'.format(Sensor, logistic_acc, random_forest_acc, xgboost_acc, svm_linear_acc, svm_rbf_acc))
    f.write('{},f1,{},{},{},{},{}\n'.format(Sensor, logistic_f1, random_forest_f1, xgboost_f1, svm_linear_f1, svm_rbf_f1))
