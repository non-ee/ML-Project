from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import apply_pca
from src.model_training import train_logistic_regression, train_random_forest, train_xgboost

import joblib

# Load and preprocess data
data = load_data("data/Occupancy_Estimation.csv")
features, target = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(features, target)

# Apply PCA
X_train_pca = apply_pca(X_train, n_components=10)
X_test_pca = apply_pca(X_test, n_components=10)

# Train models
logistic_model = train_logistic_regression(X_train_pca, y_train)
rf_model = train_random_forest(X_train_pca, y_train, n_estimators=100)
xgb_model = train_xgboost(X_train_pca, y_train)

print(logistic_model.coef_, logistic_model.intercept_)

# Save models
# joblib.dump(logistic_model, "models/logistic_reg.pkl")
# joblib.dump(rf_model, "models/random_forest.pkl")
# joblib.dump(xgb_model, "models/xgboost.pkl")
