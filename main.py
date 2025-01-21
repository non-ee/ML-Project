from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import apply_pca
from src.model_training import train_logistic_regression, train_random_forest, train_xgboost

import joblib
import pickle

# Load and preprocess data
data = load_data("data/Occupancy_Estimation.csv")
features, target = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(features, target)

# Apply PCA
X_train_pca = apply_pca(X_train, n_components=10)
X_test_pca = apply_pca(X_test, n_components=10)


# Save preprocessed data (PCA Transformed Data)
with open('data/X_train_pca.pkl', 'wb') as f:
    pickle.dump(X_train_pca, f)

with open('data/X_test_pca.pkl', 'wb') as f:
    pickle.dump(X_test_pca, f)

with open('data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

"""
Starting training
"""
# Train models
logistic_model = train_logistic_regression(X_train_pca, y_train)
rf_model = train_random_forest(X_train_pca, y_train, n_estimators=100)
xgb_model = train_xgboost(X_train_pca, y_train)

# Save models
joblib.dump(logistic_model, "models/logistic_reg.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(xgb_model, "models/xgboost.pkl")
