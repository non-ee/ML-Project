import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(df, colums):
    """
    `Scaler` standardizes features by removing the mean and scaling them to unit variance
    """
    scaler = StandardScaler()
    features = scaler.fit_transform(df[colums])
    target = df['Room_Occupancy_Count']
    return features, target

def split_data(features: pd.DataFrame, target: pd.Series, test_size: float=0.2, random_state: int=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test