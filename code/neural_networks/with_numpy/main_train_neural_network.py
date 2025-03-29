"""
In this script a neural network tries to fit some manually generated data
"""
# import plot_net
import os

import matplotlib.pyplot as plt
import numpy as np
from utils import mean_squared_error, nn_r2
from sklearn.metrics import r2_score

# Hyperparameters
HIDDEN_DIM = 6
LEARNING_RATE = 1e-7
# plotting constants
NB_ITERATIONS = 50000
NB_PLOTTED_ITERATIONS = 100
# select the data
NOISE_STD_DEV = 1


def train_neural_net(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Perform empirical risk minimization by gradient descent
    and backpropagation on a neural network with one hidden layer.
    """
    # get problem dimensions
    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]


    # Randomly initialize weights
    rng = np.random.default_rng()
    w1 = rng.normal(loc=0, scale=1, size=(input_dim, HIDDEN_DIM))
    w2 = rng.normal(loc=0, scale=1, size=(HIDDEN_DIM, output_dim))

    # we will store the loss as a function of the
    # iteration
    MSE_losses = list()
    R2_train_list = list()
    iterations = list()
    for iteration in range(NB_ITERATIONS):
        # forward pass
        # Prediction of the network for a given input x
        hh = x_train @ w1
        h_relu = np.maximum(hh, 0)
        y_pred = h_relu @ w2

        # loss function
        # Here we use the mean squared error (MSE) loss
        MSE_train = np.square(y_pred - y_train).sum() / len(y_train)
        R2_train = r2_score(y_true=y_train, y_pred=y_pred)

        # ----------------------
        # Plot the network and the loss
        if iteration % (NB_ITERATIONS / NB_PLOTTED_ITERATIONS) == 0:
            # print("plot net")
            print(f"iteration: {iteration}, train MSE: {MSE_train:.2E}")

            # keep storing the MSE loss and the iterations
            MSE_losses.append(MSE_train)
            R2_train_list.append(R2_train)
            iterations.append(iteration)

            # Plot the evolution of the MSE loss
            # We will not plot all the points
            scale = 5
            # printed_iterations = [
            #     iterations[x[0]] for x in enumerate(MSE_losses) if x[1] < MSE * scale
            # ]
            # printed_losses = [x for x in MSE_losses if x < MSE * scale]
            # printed_R2 = [x for x in MSE_losses if x < MSE * scale]
            plt.plot(
                    # printed_iterations,
                    # printed_losses,
                    R2_train_list,
                    "o",
                    alpha=0.5,
                    color="blue",
                    )
            plt.yscale("log")

            # set the limits of the plots
            # plt.xlim([min(printed_iterations) * 0.5, max(printed_iterations) * 1.2 + 1])
            # plt.ylim([0.2 * min(printed_losses), 1.5 * max(printed_losses)])
            title = (
                # "Loss function (squared loss)\n"
                "Train R2\n"
                f"{HIDDEN_DIM} hidden neurons\n"
                f"noise std: {NOISE_STD_DEV:.3f}\n"
                f"train R2: {R2_train_list[-1]:.3f}\n"
                f"learning rate: {LEARNING_RATE}"
                    )
            plt.title(title)
            plt.xlabel("iteration")
            # plt.ylabel("squared error")
            plt.ylabel("Train R2")
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

        # backpropagation
        # computation of the gradients of the loss function
        # with respect to w1 and w2
        grad_y_pred = 2.0 * (y_pred - y_train)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[hh < 0] = 0
        grad_w1 = x_train.T.dot(grad_h)

        # update the weigths
        w1 -= LEARNING_RATE * grad_w1
        w2 -= LEARNING_RATE * grad_w2

    # save the plot of the loss function
    final_train_R2 = nn_r2(
            x=x_train,
            y=y_train,
            w1=w1,
            w2=w2,
            )
    final_test_R2 = nn_r2(
            x=x_test,
            y=y_test,
            w1=w1,
            w2=w2,
            )
    plt.close()
    # plt.plot(iterations, MSE_losses)
    plt.plot(iterations, R2_train_list)
    title = (
        # "Loss function (squared error)\n"
        f"Train R2\n"
        f"{HIDDEN_DIM} hidden neurons\n"
        f"noise std: {NOISE_STD_DEV:.3f}\n"
        f"Final train R2 {final_train_R2:.3f}\n"
        f"Final test R2 {final_test_R2:.3f}\n"
        f"learning rate: {LEARNING_RATE}"
            )
    plt.title(title)
    plt.xlabel("iteration")
    # plt.ylabel("squared error")
    plt.ylabel("Train R2")
    plt.yscale("log")
    plt.tight_layout()
    figname = f"loss_fun_d_in_{input_dim}_d_out_{output_dim}_hidden_{HIDDEN_DIM}_std_{NOISE_STD_DEV}_rate_{LEARNING_RATE}.pdf"
    fig_path = os.path.join("images", figname)
    plt.savefig(fig_path)
    plt.show()
    plt.close("all")
    print("----")
    print(f"train MSE : {mean_squared_error(x_test, y_test, w1, w2)}")
    print(f"test MSE : {mean_squared_error(x_test, y_test, w1, w2)}")
    print(f"train R2: {final_train_R2:.3f}")
    print(f"test R2: {final_test_R2:.3f}")


def main() -> None:
    # select the data
    folder = "data"
    x_train = np.load(os.path.join(folder, f"training_inputs_std_{NOISE_STD_DEV}.npy"))
    y_train = np.load(os.path.join(folder, f"training_outputs_std_{NOISE_STD_DEV}.npy"))
    x_test = np.load(os.path.join(folder, f"test_inputs_std_{NOISE_STD_DEV}.npy"))
    y_test = np.load(os.path.join(folder, f"test_outputs_std_{NOISE_STD_DEV}.npy"))

    # train
    train_neural_net(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    main()
