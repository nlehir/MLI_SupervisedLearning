import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

data = np.load("samples.npy")
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# set up parameters to test
parameters = {"alpha": [1, 10], "solver": ("svd", "lsqr")}

# load estimator object
ridge = Ridge()
estimator = GridSearchCV(ridge, parameters)

# Fit inputs to outputs
estimator.fit(X, y)

print(f"best estimator : {estimator.best_estimator_}")
__import__('ipdb').set_trace()
