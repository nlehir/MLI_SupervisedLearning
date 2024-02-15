"""
OLS with scikit
"""

import os

import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    # load data
    data_path = os.path.join("data", "samples.npy")
    data = np.load(data_path)
    X = data[:, 0]
    y = data[:, 1]
    X = X.reshape((-1, 1))

    # laod an instance of the LinearRegression class
    regressor = LinearRegression()

    # optimize its parameters according to the data
    regressor.fit(X, y)

    # print the r2 score
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score
    print(f"r2 score: {regressor.score(X, y)}")

    # print theta
    print(f"{regressor.coef_=}")

    # print b
    print(f"{regressor.intercept_=}")

    # predict on some new inputs
    print(regressor.predict(np.array([1, 2, -5, 25]).reshape(-1, 1)))


if __name__ == "__main__":
    main()
