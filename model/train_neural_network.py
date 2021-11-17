import argparse
import sys

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment", default="lithology_prediction", type=str, help="experiment name"
)
parser.add_argument(
    "--epochs", default=2000, type=int, help="number of epochs to train"
)


def neural_network(x_train):

    model = Sequential()

    model.add(Dense(
        64,
        activation = 'relu',
        input_shape = (x_train.shape[-1],))
        )

    model.add(Dropout(rate = 0.2))

    model.add(Dense(32, activation = 'relu'))

    model.add(Dense(12, activation = 'softmax'))

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'Adam',
        metrics = 'categorical_accuracy'
    )

    return model

def main(arguments):

    args = parser.parse_args(arguments[1:])
    mlflow.set_experiment(args.experiment)

    DATA_PATH = "data/processed/"

    x_train = np.load(DATA_PATH + "x_train.npy")
    x_test = np.load(DATA_PATH + "x_test.npy")
    y_train = np.load(DATA_PATH + "y_train.npy")
    y_test = np.load(DATA_PATH + "y_test.npy")

    #One-Hot Enconding the Y:
    y_train_onehot = tf.one_hot(y_train, 12)
    y_test_onehot = tf.one_hot(y_test, 12)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    early_stop = EarlyStopping(
        patience=100,
        monitor="val_categorical_accuracy",
        mode="max",
        restore_best_weights=True,
    )

    mlflow.tensorflow.autolog()

    model = neural_network(x_train)

    train_history = model.fit(
        x_train,
        y_train_onehot,
        epochs=args.epochs,
        batch_size=1024,
        verbose=2,
        callbacks=[early_stop],
        validation_data=(x_test, y_test_onehot),
    )

if __name__ == "__main__":

    main(sys.argv)


# Próximos passos:
# Documentar funções
# Gerar figuras de resultados
# Guardar figuras como artefatos no Mlflow
