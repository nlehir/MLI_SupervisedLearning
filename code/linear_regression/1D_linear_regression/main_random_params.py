"""
Evaluate the quality of randomly drawn parameters for a
1 dimensional linear regression.
"""

import os

import numpy as np

from utils import empirical_risk


def main() -> None:
    n_tests = 100

    data_path = os.path.join("data", "samples.npy")
    data = np.load(data_path)

    rng = np.random.default_rng()

    best_empirical_risk = 10e12
    best_theta = 0
    best_b = 0

    for test_id in range(n_tests):
        theta = rng.uniform(-100, 100)
        b = rng.uniform(-100, 100)
        empirical_risk_ = empirical_risk(
                theta=theta,
                b=b,
                data=data,
                )
        if empirical_risk_ < best_empirical_risk:
            best_empirical_risk = empirical_risk_
            best_theta = theta
            best_b = b
        print(
            f"test:             {test_id}"
            f"\ntheta:            {theta:.2f}"
            f"\nb:                {b:.2f}"
            f"\nempirical risk:   {empirical_risk_:.2E}\n"
        )
    print(
        "\n--------"
        f"\nbest theta:          {best_theta:.2f}"
        f"\nbest b:              {best_b:2f}"
        f"\nbest empirical risk: {best_empirical_risk:.2E}\n"
    )


if __name__ == "__main__":
    main()
