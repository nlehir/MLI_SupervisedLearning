import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import optuna
import seaborn as sns

digits = load_digits()

X = digits.data
y = digits.target


def objective(trial):
    """
    Objective function

    This function return the test mean accuracy
    after fitting a ridge estimator with a given set of hyperparameters.

    Fix this function by using the optuna API.
    https://optuna.org/
    """
    # split the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # suggest a alpha
    C = trial.suggest_float("C", 1e-10, 5e2)

    # suggest a penalty
    penaltys = ["l2", None]
    penalty = trial.suggest_categorical("penalty", penaltys)

    solvers = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    solvers = ["sag", "saga"]
    solver = trial.suggest_categorical("solver", solvers)

    # instantiate Ridge regressor object
    estimator = LogisticRegression(C=C,
                                   solver=solver,
                                   penalty = penalty,
                                   max_iter = 300)

    estimator.fit(X_train, y_train)
    return estimator.score(X_test, y_test)


def prediction_squared_error(estimator, X, y):
    predictions = estimator.predict(X)
    n_samples = X.shape[0]
    error = predictions - y
    return np.linalg.norm(error) ** 2 / n_samples


def main():
    storage_name = "lr.db"
    # clean study if exists
    # you may not want this behavior in practical applications
    if os.path.exists(storage_name):
        os.remove(storage_name)
    # create a study
    study = optuna.create_study(
        storage=f"sqlite:///{storage_name}",
        study_name="logistic_regression",
        load_if_exists=False,
        direction="maximize",  # we want to maximize the R2 score
    )
    study.optimize(objective, n_trials=200)

    # print best trial
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    for key, value in study.best_trial.params.items():
        if type(value) == float:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # extract dataframe
    df = study.trials_dataframe()

    # boxplot with solver
    sns.boxplot(data=df, x="params_solver", y="value")
    plt.title("influence of the solver on the final mean accuracy")
    plt.savefig("boxplot.pdf")

    # boxplot with penalty
    sns.boxplot(data=df, x="params_penalty", y="value")
    plt.title("influence of the penalty on the final mean accuracy")
    plt.savefig("boxplot.pdf")


if __name__ == "__main__":
    main()
