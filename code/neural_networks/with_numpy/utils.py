import numpy as np


def prediction_error(
    x: np.ndarray, y: np.ndarray, w1: np.ndarray, w2: np.ndarray
) -> float:
    hh = x.dot(w1)
    h_relu = np.maximum(hh, 0)
    y_pred = h_relu.dot(w2)
    loss = np.square(y_pred - y).sum()
    return loss
