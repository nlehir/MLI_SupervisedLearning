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
phi_X = X

# split the X into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33)

clf = LogisticRegression().fit(X_train, y_train.ravel())
print("train")
print(clf.score(X_train, y_train))
print("test")
print(clf.score(X_test, y_test))
print("theta")
print(clf.coef_[0])
print("intercept")
print(clf.intercept_)

print(clf.predict(X))
