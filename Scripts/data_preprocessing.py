import json
import sys
from data_preprocessing_utils import *


def main():
    if len(sys.argv) != 2:
        print("Pass config file of data preprocessing as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    data_directory = config["Data"]["dataDirectory"]
    valid_lap_columns_one = config["Data"]["validLapColumnsOne"].split(",")
    valid_lap_columns_two = config["Data"]["validLapColumnsTwo"].split(",")
    valid_lap_columns = (
        valid_lap_columns_one
        + valid_lap_columns_two
    )
    generate_recurrent_sequence(data_directory, valid_lap_columns)


if __name__ == "__main__":
    main()
    print("\n--------------------\nData preprocessing complete!\n--------------------\n")
