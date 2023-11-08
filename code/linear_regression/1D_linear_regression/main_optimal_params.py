"""
Find the optimal parameters for a 1D linear regression
and plot the prediction made by this estimator.
"""


import os

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_optimal_params, empirical_risk


def main():
    data_path = os.path.join("data", "samples.npy")
    data = np.load(data_path)

    theta_star, b_star = compute_optimal_params(data)
    empirical_risk_star = empirical_risk(theta_star, b_star, data)
    print(
        "theta star:"
        f" {theta_star:.2f}"
        "\nb star:"
        f" {b_star:.2f}"
        "\nemipirical risk star:"
        f" {empirical_risk_star:.2f}"
    )

    # plot dataset
    X = data[:, 0]
    y = data[:, 1]
    plt.plot(X, y, "o", alpha=0.7, label="dataset")

    # plot linear regression
    x_linspace = np.linspace(min(X), max(X), num=100)
    """
    Add some lines here to plot the computed regressor
    """

    # save figure
    plt.xlabel("temperature (°C)")
    plt.ylabel("power_consumption (MW)")
    plt.legend(loc="best")
    plt.title("Linear regression")
    figpath = os.path.join("images", "optimal_regression.pdf")
    plt.savefig(figpath)


if __name__ == "__main__":
    main()
