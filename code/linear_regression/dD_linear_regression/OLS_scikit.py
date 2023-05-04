"""
Perform an ordinary least squares regression
on the toy data
"""


import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    # load data
    X_path = os.path.join("data", "X.npy")
    y_path = os.path.join("data", "y.npy")
    X = np.load(X_path)
    y = np.load(y_path)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # load estimator object
    estimator = LinearRegression()

    # Fit inputs to outputs on train set
    estimator.fit(X_train, y_train)

    def prediction_squared_error(estimator, X, y):
        predictions = estimator.predict(X)
        n_samples = X.shape[0]
        error = predictions - y
        return np.linalg.norm(error) ** 2 / n_samples

    print(f"train r2 score: {estimator.score(X_train, y_train)}")
    print(f"test r2 score: {estimator.score(X_test, y_test)}")
    print(
        f"\ntrain prediction mean squared error: {prediction_squared_error(estimator, X_train, y_train):.2f}"
    )
    print(
        f"test prediction mean squared error: {prediction_squared_error(estimator, X_test, y_test):.2f}"
    )


if __name__ == "__main__":
    main()
