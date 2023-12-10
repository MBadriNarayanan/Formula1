import json
import sys
from model_utils import *

from keras.models import load_model


def main():
    if len(sys.argv) != 2:
        print("Pass config file of model as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    sequence_file_path = config["Data"]["sequencePath"]
    label_sequence_file_path = config["Data"]["labelPath"]
    val_size = config["Data"]["valSize"]
    test_size = config["Data"]["testSize"]

    model_flag = config["Model"]["modelFlag"]
    bidirectional_flag = config["Model"]["bidirectionalFlag"]
    recurrent_units = config["Model"]["recurrentUnits"]
    dropout_value = config["Model"]["dropoutValue"]
    sequence_size = config["Model"]["sequenceSize"]
    number_of_features = config["Model"]["numberOfFeatures"]
    logs_directory = config["Model"]["logsDirectory"]
    epochs = config["Model"]["numberOfEpochs"]
    batch_size = config["Model"]["batchSize"]

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_data(
        sequence_file_path, label_sequence_file_path, val_size, test_size
    )

    model = create_model(
        model_flag,
        bidirectional_flag,
        recurrent_units,
        dropout_value,
        sequence_size,
        number_of_features,
    )
    train_model(
        model, X_train, y_train, X_val, y_val, epochs, batch_size, logs_directory
    )
    evaluate_model(X_test, y_test, logs_directory)


if __name__ == "__main__":
    main()
    print(
        "\n--------------------\nModel Training and Evaluation completed!\n--------------------\n"
    )
