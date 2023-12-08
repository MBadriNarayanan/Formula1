import json
import sys
from data_collection_utils import *


def main():
    if len(sys.argv) != 2:
        print("Pass config file of data collection as argument!")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        config = json.load(fjson)

    data_directory = config["Data"]["dataDirectory"]
    valid_years = [int(x) for x in config["Data"]["validYears"].split(",")]

    valid_result_columns_one = config["Data"]["validResultColumnsOne"].split(",")
    valid_result_columns_two = config["Data"]["validResultColumnsTwo"].split(",")
    valid_result_columns = valid_result_columns_one + valid_result_columns_two

    valid_lap_columns_one = config["Data"]["validLapColumnsOne"].split(",")
    valid_lap_columns_two = config["Data"]["validLapColumnsTwo"].split(",")
    valid_lap_columns_three = config["Data"]["validLapColumnsThree"].split(",")
    valid_lap_columns_four = config["Data"]["validLapColumnsFour"].split(",")
    valid_lap_columns_five = config["Data"]["validLapColumnsFive"].split(",")
    valid_lap_columns_six = config["Data"]["validLapColumnsSix"].split(",")
    valid_lap_columns_seven = config["Data"]["validLapColumnsSeven"].split(",")
    valid_lap_columns_eight = config["Data"]["validLapColumnsEight"].split(",")
    valid_lap_columns = (
        valid_lap_columns_one
        + valid_lap_columns_two
        + valid_lap_columns_three
        + valid_lap_columns_four
    )
    valid_lap_columns += (
        valid_lap_columns_five
        + valid_lap_columns_six
        + valid_lap_columns_seven
        + valid_lap_columns_eight
    )

    create_data_frame(
        data_directory, valid_years, valid_result_columns, valid_lap_columns
    )


if __name__ == "__main__":
    main()
    print("\n--------------------\nData collection complete!\n--------------------\n")
