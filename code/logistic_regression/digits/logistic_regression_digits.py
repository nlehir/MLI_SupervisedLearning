from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

digits = load_digits()

X = digits.data
y = digits.target

# split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# load and fit clasificer
clf = LogisticRegression().fit(X_train, y_train)

print(f"train accuracy: {clf.score(X_train, y_train)}")
print(f"test accuracy: {clf.score(X_test, y_test)}")
