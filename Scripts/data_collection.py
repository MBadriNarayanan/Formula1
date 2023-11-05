import json
import sys
from utils import *


def main():
    if len(sys.argv) != 2:
        print("Pass config file of data collection as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    data_directory = config["Data"]["dataDirectory"]
    valid_years = [int(x) for x in config["Data"]["validYears"].split(",")]
    valid_result_columns = config["Data"]["validResultColumns"].split(",")
    valid_lap_columns = config["Data"]["validLapColumns"].split(",")

    create_data_frame(
        data_directory, valid_years, valid_result_columns, valid_lap_columns
    )


if __name__ == "__main__":
    main()
    print("\n--------------------\nData collection complete!\n--------------------\n")
