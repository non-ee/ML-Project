from src.data_preprocessing import split_data
# Features and target
features = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
target = [0, 1, 0, 1, 0]

# Split data
X_train, X_test, y_train, y_test = split_data(features, target, test_size=0.4, random_state=0)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)
