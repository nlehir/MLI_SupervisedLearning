"""
Perform logistic regression on the tranformed data
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X = np.load("X.npy")
y = np.load("y.npy")


"""
Edit from here
"""
phi_X = (X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)

# split the X into training and test
phi_X_train, phi_X_test, y_train, y_test = train_test_split(
    phi_X, y, test_size=0.33)

clf = LogisticRegression().fit(phi_X_train, y_train.ravel())
print("train")
print(clf.score(phi_X_train, y_train))
print("test")
print(clf.score(phi_X_test, y_test))
print("theta")
print(clf.coef_[0])
print("intercept")
print(clf.intercept_)

print(clf.predict(phi_X))
