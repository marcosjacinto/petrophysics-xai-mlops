import argparse
import sys

import mlflow
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential


parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment", default="sonic_prediction", type=str, help="experiment name"
)
parser.add_argument(
    "--epochs", default=2000, type=int, help="number of epochs to train"
)


def neural_network(x_train: np.ndarray) -> Sequential:
    """This function creates neural network with three hidden layers and single
    layer for the output value which can be used for predict DTC or DTS.
    The neural network also contains Dropout layers used to avoid overfitting.
    The number of features is obtained from the number of columns in the x_train
    array. This value is used to inform the input shape in the first hidden
    layer.

    Args:
        x_train (np.ndarray): numpy array which contains the features used to
        perform the prediction.

    Returns:
        Sequential: neural network model which can be trained to predict DTC or
        DTS.
    """

    model = Sequential()

    # First hidden layer
    model.add(Dense(64, activation="relu", input_shape=(x_train.shape[-1],)))
    model.add(Dropout(rate=0.1))

    # Second hidden layer
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(rate=0.1))

    # Third hidden layer
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(rate=0.1))

    # Output layer
    model.add(Dense(1, activation="relu"))

    model.compile(
        loss="mean_squared_error",
        optimizer="Adam",
        metrics=[RootMeanSquaredError(), "mae"],
    )

    return model


def main(arguments: list) -> None:
    """Main function responsible for loading the data, creating the models,
    training the DTC and DTS model, as well as registering each experiment
    with MLFlow.

    Args:
        arguments (list): command line arguments, such as the number of epochs
        and the base name of the experiment.
    """

    early_stop = EarlyStopping(
        patience=50,
        monitor="val_mae",
        mode="min",
        restore_best_weights=True,
    )

    args = parser.parse_args(arguments[1:])

    # A new experiment starts here: training a DTC model
    mlflow.set_experiment(args.experiment + "_DTC")
    mlflow.start_run()
    mlflow.tensorflow.autolog()

    DATA_PATH = "data/processed/"

    x_train = np.load(DATA_PATH + "x_train.npy")
    x_test = np.load(DATA_PATH + "x_test.npy")
    y_train_dtc = np.load(DATA_PATH + "y_train_dtc.npy")
    y_test_dtc = np.load(DATA_PATH + "y_test_dtc.npy")
    y_train_dts = np.load(DATA_PATH + "y_train_dts.npy")
    y_test_dts = np.load(DATA_PATH + "y_test_dts.npy")

    dtc_model = neural_network(x_train)  # Instantiates a new neural network

    # Starts training
    dtc_model.fit(
        x_train,
        y_train_dtc,
        epochs=args.epochs,
        batch_size=16,
        verbose=2,
        callbacks=[early_stop],
        validation_data=(x_test, y_test_dtc),
    )

    dtc_model.save("model/dtc_model.h5")
    mlflow.log_artifact("model/dtc_model.h5")

    mlflow.end_run()  # The experiment ends here: training a DTC model

    # A new experiment starts here: training a DTS model
    mlflow.set_experiment(args.experiment + "_DTS")
    mlflow.start_run()
    mlflow.tensorflow.autolog()

    dts_model = neural_network(x_train)  # Instantiates a new neural network

    # Starts training
    dts_model.fit(
        x_train,
        y_train_dts,
        epochs=args.epochs,
        batch_size=16,
        verbose=2,
        callbacks=[early_stop],
        validation_data=(x_test, y_test_dts),
    )

    dts_model.save("model/dts_model.h5")
    mlflow.log_artifact("model/dts_model.h5")

    mlflow.end_run()  # The experiment ends here: training a DTS model


if __name__ == "__main__":

    # If this file is executed directly using a command in the terminal run the
    # main function:
    main(sys.argv)
