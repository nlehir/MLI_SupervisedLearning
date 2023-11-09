import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.utils import plot_model

# adapted from this tutorial
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

def plot_datapoint(data_index: int, x_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    plot a mnist sample from the dataset
    """
    print(f"plot data point {data_index}")
    nb_train_data = x_train.shape[0]
    label = y_train[data_index]
    if data_index > nb_train_data:
        raise ValueError(
            "data index too large : only {nb_train_data} training samples available"
        )
    plt.imshow(x_train[data_index], cmap="Greys")
    plt.title(f"training point: {data_index}\nlabel: {label}")
    plt.savefig(f"images/data_{data_index}.pdf")
    plt.close()


def main() -> None:
    # Load the data
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    y_test = np.load("data/y_test.npy")
    x_test = np.load("data/x_test.npy")

    samples_to_plot = [27, 299, 720, 1936, 52917]
    for sample in samples_to_plot:
        plot_datapoint(data_index=sample, x_train=x_train, y_train=y_train)

    # Preprocess the dataset
    print("pre process the dataset")
    # reshape the arrays for the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # cast the data to floats in order to work with decimal division
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Normalize the vectors codes
    # Initially, their maximum value is 255
    x_train = x_train / 255
    x_test = x_test / 255

    # Building the convolutional network using Keras API
    print("\nbuild the model")
    model = Sequential()
    # https://keras.io/getting-started/sequential-model-guide/

    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    # https://keras.io/layers/core/
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flattening the 2D arrays for fully connected layers
    model.add(Flatten())
    # adding one densely connected layer
    model.add(Dense(128, activation=tf.nn.relu))
    # set the dropout rate
    model.add(Dropout(0.2))
    # final layer for output prediction
    model.add(Dense(10, activation=tf.nn.softmax))

    # visualize the neural network
    plot_model(model, to_file="images/network.pdf")
    plot_model(model, to_file="images/network_with_shapes.pdf", show_shapes=True)

    # compile and fit the model
    print("\ncompile and fit the model")
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x=x_train, y=y_train, epochs=10)

    # Evaluate the model
    print("\nevaluate the model")
    model.evaluate(x_test, y_test)


    # save the model to a json file
    print("\nsave the model")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")

if __name__ == "__main__":
    main()
