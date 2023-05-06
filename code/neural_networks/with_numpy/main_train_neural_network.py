"""
In this script a neural network tries to fit randomly generated data
"""
# import plot_net
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import prediction_error


# Hyperparameters
HIDDEN_DIM = 3
LEARNING_RATE = 1e-5
# plotting constants
NB_STEPS = 5000
NB_PLOTTED_STEPS = 50
# select the data
NOISE_STD_DEV = 0.00


def main() -> None:
    # select the data
    folder = "data"
    x_train = np.load(os.path.join(folder, f"training_inputs_std_{NOISE_STD_DEV}.npy"))
    y_train = np.load(os.path.join(folder, f"training_outputs_std_{NOISE_STD_DEV}.npy"))
    x_test = np.load(os.path.join(folder, f"test_inputs_std_{NOISE_STD_DEV}.npy"))
    y_test = np.load(os.path.join(folder, f"test_outputs_std_{NOISE_STD_DEV}.npy"))

    # train
    train_neural_net(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


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

    # directory where we will store results
    figname = f"d_in_{input_dim}_d_out_{output_dim}_hidden_{HIDDEN_DIM}_std_{NOISE_STD_DEV}_rate_{LEARNING_RATE}"
    dir_name = "visualization/structured_data/" + figname + "/"

    # Randomly initialize weights
    w1 = np.random.randn(input_dim, HIDDEN_DIM)
    w2 = np.random.randn(HIDDEN_DIM, output_dim)

    # we will store the loss as a function of the
    # optimization step
    losses = list()
    steps = list()
    for step in range(NB_STEPS):
        # forward pass
        # Prediction of the network for a given input x
        hh = x_train @ w1
        h_relu = np.maximum(hh, 0)
        y_pred = h_relu @ w2

        # loss function
        # Here we use the squared error loss
        loss = np.square(y_pred - y_train).sum()

        # ----------------------
        # Plot the network and the loss
        if step % (NB_STEPS / NB_PLOTTED_STEPS) == 0:
            # print("plot net")
            print(f"step: {step}, train loss: {loss:.2E}")
            graph_name = f"net_{step}"

            # keep storing the loss and the steps
            losses.append(loss)
            steps.append(step)

            # Plot the evolution of the loss
            # We will not plot all the points
            scale = 5
            printed_steps = [
                steps[x[0]] for x in enumerate(losses) if x[1] < loss * scale
            ]
            printed_losses = [x for x in losses if x < loss * scale]
            plt.plot(printed_steps, printed_losses, "o")

            # set the limits of the plots
            plt.xlim([min(printed_steps) * 0.5, max(printed_steps) * 1.2 + 1])
            plt.ylim([0.2 * min(printed_losses), 1.5 * max(printed_losses)])
            plt.title("Loss function (square error)")
            plt.draw()
            plt.pause(0.01)

            # ----------------------
            # Print the network with gaphviz
            # plot_net.show_net(
            #     step,
            #     w1,
            #     w2,
            #     input_dim,
            #     HIDDEN_DIM,
            #     output_dim,
            #     figname,
            #     dir_name,
            #     graph_name,
            #     loss,
            #     LEARNING_RATE,
            # )

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

        # test_error = prediction_error(x_test, y_test, w1, w2)
        # train_error = prediction_error(x_train, y_train, w1, w2)
        # if test_error > 2 * train_error:
        #     __import__("ipdb").set_trace()

    # save the plot of the loss function
    plt.close()
    plt.plot(steps, losses)
    plt.title("Loss function (square error)")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(os.path.join(dir_name, "Loss_function.pdf"))
    plt.show()
    plt.close("all")
    print("----")
    print(f"error on test set : {prediction_error(x_test, y_test, w1, w2)}")


if __name__ == "__main__":
    main()
