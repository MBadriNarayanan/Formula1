import json
import sys
from utils import *


def main():
    if len(sys.argv) != 2:
        print("Pass config file of data preprocessing as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    data_directory = config["Data"]["dataDirectory"]
    preprocess_dataframe(data_directory)
    filter_dataframe(data_directory)


if __name__ == "__main__":
    main()
    print("\n--------------------\nData preprocessing complete!\n--------------------\n")
