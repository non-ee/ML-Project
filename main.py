from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import apply_pca
from src.model_training import train_logistic_regression, train_random_forest, train_xgboost, train_svm

import joblib
import pickle

# Drop columns
drop_columns = ['Date', 'Time', 'Room_Occupancy_Count']
drop_temps = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']
drop_lights = ['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']
drop_sounds = ['S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']
drop_co2 = ['S5_CO2', 'S5_CO2_Slope']
drop_pir = ['S6_PIR', 'S7_PIR']


# sel_columns = drop_lights + drop_temps
# sel_columns = drop_sounds
# sel_columns = drop_temps
# sel_columns = drop_lights + drop_co2
sel_columns = drop_lights

# drop_columns += drop_sounds
# drop_columns += drop_co2
# drop_columns += drop_co2
# drop_columns += drop_sounds + drop_co2 + drop_temps
# drop_columns += drop_sounds + drop_temps + drop_lights + drop_co2

# Load and preprocess data
data = load_data("data/Occupancy_Estimation.csv")
features, target = preprocess_data(data, sel_columns)
X_train, X_test, y_train, y_test = split_data(features, target)


with open('data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

"""
Starting training
"""
# Train models
logistic_model = train_logistic_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train, n_estimators=100)
xgb_model = train_xgboost(X_train, y_train)
svm_linear_model = train_svm(X_train, y_train, kernel='linear', C=1.0)
svm_rbf_model = train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale')

# Save models
joblib.dump(logistic_model, "models/logistic_reg.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(xgb_model, "models/xgboost.pkl")
joblib.dump(svm_linear_model, "models/svm_linear.pkl")
joblib.dump(svm_rbf_model, "models/svm_rbf.pkl")
