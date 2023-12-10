import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import load_model, Sequential
from keras.layers import Bidirectional, Dense, LSTM, SimpleRNN


def loss_plot(history, image_path):
    plt.plot(history.history["loss"], label="Training data")
    plt.plot(history.history["val_loss"], label="Validation data")
    plt.title("Loss")
    plt.ylabel("Loss value")
    plt.xlabel("No. epoch")
    plt.legend(loc="upper right")
    plt.savefig(image_path)
    plt.show()


def classification_metrics(ground_truth, predictions, test_loss, logs_directory):
    image_path = os.path.join(logs_directory, "confusion_matrix.png")
    report_path = os.path.join(logs_directory, "metrics.txt")

    target_names = ["NoPitStop", "PitStop"]
    accuracy = accuracy_score(ground_truth, predictions)
    matrix = confusion_matrix(ground_truth, predictions)
    disp = ConfusionMatrixDisplay(matrix, display_labels=target_names)
    disp.plot()
    plt.xticks(rotation=45)
    plt.savefig(image_path)
    report = classification_report(
        ground_truth, predictions, target_names=target_names, digits=3, zero_division=0
    )
    print("Accuracy Score: {:.3f}".format(accuracy))
    print("Confusion Matrix\n {}".format(matrix))
    print("Classification Report\n {}".format(report))
    print("\n")

    report_file = open(report_path, "w")
    report_file.write("Loss: {:.3f}\n".format(test_loss))
    report_file.write("Accuracy: {:.3f}\n".format(accuracy))
    report_file.write("Classification Report\n")
    report_file.write("{}\n".format(report))
    report_file.write("Confusion Matrix\n")
    report_file.write("{}\n".format(matrix))
    report_file.write("-----------------------------\n")
    report_file.close()


def generate_sequence(sequence_file_path):
    recurrent_sequence = np.load(sequence_file_path, allow_pickle=True)
    sequence = []
    for race_sequence in recurrent_sequence:
        for grand_prix_sequence in race_sequence:
            for driver_sequence in grand_prix_sequence:
                for lap_sequence in driver_sequence:
                    sequence.append(lap_sequence)
    sequence = np.array(sequence)
    return sequence


def generate_data(sequence_file_path, label_sequence_file_path, val_size, test_size):
    X = generate_sequence(sequence_file_path)
    y = generate_sequence(label_sequence_file_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=test_size, random_state=42
    )

    knn_imputer = KNNImputer(n_neighbors=3)
    X_train[:, :, 9] = knn_imputer.fit_transform(X_train[:, :, 9])
    X_val[:, :, 9] = knn_imputer.transform(X_val[:, :, 9])
    X_test[:, :, 9] = knn_imputer.transform(X_test[:, :, 9])

    knn_imputer = KNNImputer(n_neighbors=3)
    X_train[:, :, 10] = knn_imputer.fit_transform(X_train[:, :, 10])
    X_val[:, :, 10] = knn_imputer.transform(X_val[:, :, 10])
    X_test[:, :, 10] = knn_imputer.transform(X_test[:, :, 10])

    knn_imputer = KNNImputer(n_neighbors=3)
    X_train[:, :, 11] = knn_imputer.fit_transform(X_train[:, :, 11])
    X_val[:, :, 11] = knn_imputer.transform(X_val[:, :, 11])
    X_test[:, :, 11] = knn_imputer.transform(X_test[:, :, 11])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_model(
    model_flag,
    bidirectional_flag,
    recurrent_units,
    dropout_value,
    sequence_size,
    number_of_features,
):
    num_classes = 1
    input_shape = (sequence_size, number_of_features)

    model = Sequential()
    if model_flag == "rnn":
        if bidirectional_flag:
            model.add(
                Bidirectional(
                    SimpleRNN(
                        units=recurrent_units,
                        dropout=dropout_value,
                        input_shape=input_shape,
                    )
                )
            )
        else:
            model.add(
                SimpleRNN(
                    units=recurrent_units,
                    dropout=dropout_value,
                    input_shape=input_shape,
                )
            )
    else:
        if bidirectional_flag:
            model.add(
                Bidirectional(
                    LSTM(
                        units=recurrent_units,
                        dropout=dropout_value,
                        input_shape=input_shape,
                    )
                )
            )
        else:
            model.add(
                LSTM(
                    units=recurrent_units,
                    dropout=dropout_value,
                    input_shape=input_shape,
                )
            )
    model.add(Dense(128, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="sigmoid"))
    print(model.summary())
    return model


def train_model(
    model, X_train, y_train, X_val, y_val, epochs, batch_size, logs_directory
):
    loss_function = "binary_crossentropy"
    optimizer_function = "adam"
    image_path = os.path.join(logs_directory, "loss.png")
    model_path = os.path.join(logs_directory, "model.keras")

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    callbacks = [early_stopping]

    model.compile(
        loss=loss_function, optimizer=optimizer_function, metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )
    model.save(model_path)
    loss_plot(history, image_path)


def evaluate_model(X_test, y_test, logs_directory):
    model_path = os.path.join(logs_directory, "model.keras")
    model = load_model(model_path)
    test_loss, _ = model.evaluate(X_test, y_test, verbose=0)

    ground_truth = y_test
    predictions = model.predict(X_test, verbose=0)
    predictions = predictions.reshape(predictions.shape[0])

    upper, lower = 1, 0
    predictions = np.where(predictions > 0.5, upper, lower)

    classification_metrics(ground_truth, predictions, test_loss, logs_directory)
