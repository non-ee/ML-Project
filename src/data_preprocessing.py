import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(df):
    """
    `Scaler` standardizes features by removing the mean and scaling them to unit variance
    """
    scaler = StandardScaler()
    features = scaler.fit_transform(df.drop(['Date', 'Time', 'Room_Occupancy_Count'], axis=1))
    target = df['Room_Occupancy_Count']
    return features, target

def split_data(features, target, test_size=0.2, random_state=42):
    return train_test_split(features, target, test_size=test_size, random_state=random_state)