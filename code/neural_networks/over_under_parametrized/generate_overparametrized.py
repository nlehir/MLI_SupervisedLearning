"""
    generate an overparametrized problem
    (there exists an estimator that interpolates the data)
"""

import torch


def main() -> None:
    n_samples = 20
    dim_in = 50
    dim_h = 80
    dim_out = 1

    X = torch.randn(n_samples, dim_in)
    Neural_network = torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_h), torch.nn.ReLU(), torch.nn.Linear(dim_h, dim_out)
    )
    y = Neural_network(X)

    torch.save(X, "data/X_overparametrized")
    torch.save(y, "data/y_overparametrized")


if __name__ == "__main__":
    main()
