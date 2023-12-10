import json
import sys
from data_preprocessing_utils import *

# FIXME: need to talk about handling nulls and normalizing steps.
# runs preprocessing functions to return a fully filtered dataset
# Input: pandas dataframe, df_raw. this should be same as the PreprocessedLapData.csv
# Output:
#    1. df_new: dataframe of fully preprocessed data
#    2. stats: dictonary of stats on what each step does to the dataframe for documentation.
#           (the dictonary does not include null count, although it could if people want)
def get_fully_preprocessed(df_raw):
  stats = {}
  # load in the data
  df_raw = pd.read_csv('/content/PreprocessedLapData.csv')
  stats['1'] = {"df shape": df_raw.shape,
                "description of new task": "load in data collection data"}
  print(df_raw.columns)

  # removes the rows whose behavior we do not want to measure
  df_raw = filter_dataframe(df_raw)
  stats['2'] = {"df shape": df_raw.shape,
                "description of new task": "removes the rows whose behavior we do not want to measure"}

  # turn time strings into integers
  time_to_seconds(df_raw)
  stats['3'] = {"df shape": df_raw.shape,
                "description of new task": "turn time strings into integers"}

  # turn categorical data into one hot encoded data
  all_the_hotness(df_raw)
  stats['4'] = {"df shape": df_raw.shape,
                "description of new task": "turn categorical data into one hot encoded data"}

  # drop the columns I do not want
  drop_cols(df_raw)
  stats['5'] = {"df shape": df_raw.shape,
                "description of new task": "turn categorical data into one hot encoded data"}

  # handle nulls
  # FIXME: this is a placeholder
  df_new = remove_nulls(df_raw)
  stats['6'] = {"df shape": df_new.shape,
                "description of new task": "removes nulls in a horrible way"}

  # FIXME: normalize

  return df_new, stats


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
    
    # FIXME: df_raw
    df_new, stats = get_fully_preprocessed(df_raw)
    # 


if __name__ == "__main__":
    main()
    print("\n--------------------\nData preprocessing complete!\n--------------------\n")
