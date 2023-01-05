import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV

# load data
X_path = os.path.join("data", "X.npy")
theta_star_path = os.path.join("data", "theta_star.npy")
y_path = os.path.join("data", "y.npy")
X = np.load(X_path)
y = np.load(y_path)
theta_star = np.load(theta_star_path)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# load estimator object
# estimator = Ridge(alpha = 10)
estimator = Ridge()

# Fit inputs to outputs on train set
estimator.fit(X_train, y_train)


def prediction_squared_error(estimator, X, y):
    predictions = estimator.predict(X)
    n_samples = X.shape[0]
    error = predictions-y
    return np.linalg.norm(error)**2/n_samples

print(f"train r2 score: {estimator.score(X_train, y_train)}")
print(f"test r2 score: {estimator.score(X_test, y_test)}")
print(f"\ntrain prediction mean squared error: {prediction_squared_error(estimator, X_train, y_train):.2f}")
print(f"test prediction mean squared error: {prediction_squared_error(estimator, X_test, y_test):.2f}")
