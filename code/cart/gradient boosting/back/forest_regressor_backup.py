# Import the necessary modules and libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt

xmin = 0
xmax = 2
sigma = 0.5
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
nb_samples = 80
X = np.linspace(xmin, xmax, nb_samples).reshape(nb_samples, 1)
noise = rng.normal(0, sigma, (X.shape[0], 1))
modulation = (1-X/xmax).ravel()
y = modulation*(np.sin(8*X)+noise).ravel()

# Fit regression model
max_depth = 3
n_estimators = 10
regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
regr.fit(X, y)

# title and params
params = f"\nnumber of estimators: {n_estimators}\n"
params += f"max depth: {max_depth}"
title = "Random forest regression"
title += params
params_fig = f"_nest_{n_estimators}"
params_fig += f"_d_{max_depth}"
figtitle = "Random forest regression"
figtitle += params_fig

# Predict
X_test = np.arange(xmin, xmax, 0.01)[:, np.newaxis]
y_pred = regr.predict(X_test)
# y_bayes = modulation*(np.sin(8*X_test))

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_pred, color="cornflowerblue",
         label="prediction", linewidth=2)
# plt.plot(X_test, y_bayes, color="aquamarine", label="Bayes predictor", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title(title)
plt.legend()
plt.tight_layout()
plt.savefig(figtitle+".pdf")
