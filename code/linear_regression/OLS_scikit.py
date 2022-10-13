import numpy as np
from sklearn.linear_model import LinearRegression

data = np.load("samples.npy")
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# load estimator object
estimator = LinearRegression()

# Fit inputs to outputs
estimator.fit(X, y)

print(f"estimator score: {estimator.score(X, y)}")
print(f"etimator coefficient: {estimator.coef_}")
print(f"estimator b: {estimator.intercept_}")
