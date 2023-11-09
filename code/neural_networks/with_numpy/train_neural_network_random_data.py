# -*- coding: utf-8 -*-
"""
In this script a neural network tries to fit randomly generated data
"""
# import plot_net
import os

import matplotlib.pyplot as plt
import numpy as np
from utils import prediction_error

# HYPERPARAMETERS
batch_size, input_dim, hidden_dim, output_dim = 64, 20, 10, 1
Learning_rate = 1e-4
Nb_steps = 50000
Nb_plotted_steps = 20


def fit_random_data(batch_size, input_dim, hidden_dim, output_dim):
    """
    function creating random data and
    a neural network learning
    to predict the outputs
    as a function of the inputs.
    """

    figname = (
        f"d_in_{input_dim}_d_out_{output_dim}_hidden_{hidden_dim}_rate_{Learning_rate}"
    )
    # directory where we will store results
    dir_name = "visualization/random_data/" + figname + "/"

    # Create random input and output data
    x = np.random.randn(batch_size, input_dim)
    y = np.random.randn(batch_size, output_dim)

    # ----------------------
    # INITIALIZATION OF THE WEIGHTS
    # Randomly initialize weights
    w1 = np.random.randn(input_dim, hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim)

    # we will store the loss as a function of the
    # optimization step
    losses = list()
    steps = list()
    for step in range(Nb_steps):
        # ----------------------
        # FORWARD PASS
        # Prediction of the network for a given input x
        hh = x.dot(w1)
        h_relu = np.maximum(hh, 0)
        y_pred = h_relu.dot(w2)

        # ----------------------
        # LOSS FUNCTION
        # Here we use the Square Error loss
        loss = np.square(y_pred - y).sum()

        # ----------------------
        # Plot the network and the loss
        if step % (Nb_steps / Nb_plotted_steps) == 0:
            print(step, loss)
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
            # set the limits of the plots to make it nice
            plt.xlim([min(printed_steps) * 0.5, max(printed_steps) * 1.2 + 1])
            plt.ylim([0.2 * min(printed_losses), 1.5 * max(printed_losses)])
            plt.title("Loss function (square error)")
            plt.draw()
            plt.pause(0.05)

            # ----------------------
            # Print the network with graphviz
            # plot_net.show_net(
            #     step,
            #     w1,
            #     w2,
            #     input_dim,
            #     hidden_dim,
            #     output_dim,
            #     figname,
            #     dir_name,
            #     graph_name,
            #     loss,
            #     Learning_rate,
            # )

        # -----------------------
        # BACKPROPAGATION
        # computation of the gradients of w1 and w2 with respect
        # to the loss function
        # -----------------------
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[hh < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # -----------------------
        # UPDATE OF THE WEIGTHS
        # with the results of backpropagation
        w1 -= Learning_rate * grad_w1
        w2 -= Learning_rate * grad_w2

    # -----------------------
    # save the plot of the loss function
    plt.close()
    plt.plot(steps, losses)
    plt.title("Loss function (Mean square error)")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(dir_name + "Loss_function.pdf")
    plt.show()
    plt.close("all")

    x_test = np.random.randn(batch_size, input_dim)
    y_test = np.random.randn(batch_size, output_dim)
    print("----")
    print(f"error on test set : {prediction_error(x_test, y_test, w1, w2)}")


if __name__ == "__main__":
    fit_random_data(batch_size, input_dim, hidden_dim, output_dim)
