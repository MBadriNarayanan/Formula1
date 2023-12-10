import os

import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder


def get_pit_stop_duration(in_time, out_time):
    if (in_time != in_time) and (out_time != out_time):
        return -1
    else:
        if in_time != in_time:
            return -2
        elif out_time != out_time:
            return -3
        return abs(in_time - out_time)


def generate_labels(lap_df):
    pit_stop_dict = {}
    for driver_name in Counter(lap_df["Driver"]).keys():
        driver_df = lap_df.loc[lap_df["Driver"] == driver_name]
        index_list = list(driver_df.index)
        for index in range(len(index_list)):
            current_index = index_list[index]
            current_stint = lap_df.iloc[current_index]["Stint"]

            previous_index = index_list[index - 1]
            previous_stint = lap_df.iloc[previous_index]["Stint"]

            if current_stint != current_stint:
                pit_stop_dict[current_index] = -1
            else:
                if current_stint != previous_stint:
                    pit_stop_dict[current_index] = 1
                else:
                    pit_stop_dict[current_index] = 0
    lap_df["PitStop"] = -500
    lap_df.loc[list(pit_stop_dict.keys()), "PitStop"] = list(pit_stop_dict.values())
    return lap_df

# FIXME: commented out Remaining pit stops
def filter_dataframe(dataframe):
    df = dataframe.loc[dataframe["FinishingPosition"] <= 10]
    df = df.loc[df["PitStop"] != -1]
    df = df.loc[df["LabelCompound"] != 4]
    df = df.loc[df["RemainingPitStops"] <= 3]
    df = df.loc[df["LapTimeInSeconds"] <= 200]
    df = df.loc[(df["LapsCompleted"] >= 0.1) & (df["LapsCompleted"] <= 0.9)]
    df = df.reset_index(drop = True)
    return df


def generate_sequence(df, sequence_size=4):
    sequence = []
    for idx in range(df.shape[0]):
        data = df.iloc[idx : idx + sequence_size]
        data = np.array(data)
        if data.shape[0] < sequence_size:
            break
        sequence.append(data)
    return sequence


def generate_recurrent_sequence(data_directory, valid_lap_columns):
    recurrent_sequence = []
    recurrent_sequence_path = os.path.join(data_directory, "LapSequence.npy")
    for year_name in tqdm(os.listdir(data_directory)):
        year_path = os.path.join(data_directory, year_name)
        grand_prix_sequence = []
        for grand_prix in os.listdir(year_path):
            grand_prix_path = os.path.join(year_path, grand_prix)
            lap_csv_path = os.path.join(grand_prix_path, "LapData.csv")
            lap_df = pd.read_csv(lap_csv_path)
            lap_df["PitTimeDifference"] = lap_df.apply(
                lambda x: get_pit_stop_duration(
                    x["PitInTimeInSeconds"], x["PitOutTimeInSeconds"]
                ),
                axis=1,
            )
            lap_df["SwitchedCompounds"] = lap_df["SwitchedCompounds"].apply(
                lambda x: int(x)
            )
            lap_df["CloseLeader"] = lap_df["CloseLeader"].apply(lambda x: int(x))
            lap_df = generate_labels(lap_df)
            lap_df = filter_dataframe(lap_df)
            lap_df = lap_df[valid_lap_columns].reset_index(drop=True)
            lap_sequence = []
            for driver in Counter(lap_df["DriverNumber"]).keys():
                driver_df = lap_df.loc[lap_df["DriverNumber"] == driver].reset_index(
                    drop=True
                )
                driver_sequence = generate_sequence(driver_df, sequence_size=4)
                driver_sequence = np.array(driver_sequence, dtype=object)
                lap_sequence.append(driver_sequence)
            lap_sequence = np.array(lap_sequence, dtype=object)
            grand_prix_sequence.append(lap_sequence)
        recurrent_sequence.append(grand_prix_sequence)
    recurrent_sequence = np.array(recurrent_sequence, dtype=object)
    np.save(recurrent_sequence_path, recurrent_sequence)
    return recurrent_sequence

from datetime import datetime
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


# turn pandas string representation of time columns into floats of seconds.
# input: time_cols -> a list of columns which are in string date version
#         string_format -> representation of the format of the string
#         epoch_time -> represents the default time for when day and month isnt represented
#             NOTE: this can change based on the version running the thingy. (but should be normalized away)
# output: the dataframe, but with all columns previously represented as string dates, are now seconds from race/lap start time
def time_to_seconds(df, time_cols = None, string_format = None, epoch_time = None ):
  if time_cols == None:
    time_cols= [
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "Sector1SessionTime",
    "Sector2SessionTime" ,
    "Sector3SessionTime",  
    "InFrontTime",
    "PursuerTime",
    "PitInTime",
    "PitOutTime",
    "Time",
    "LapTime"]

  if string_format == None:
    string_format = "0 days %H:%M:%S.%f"
  if epoch_time == None:
    epoch_time = datetime(1900, 1, 1)


  for el in time_cols:
    df[el] = (pd.to_datetime(df[el], 
                            format = string_format,
                            errors = 'coerce'
                            )-epoch_time).dt.total_seconds()
              
  return None

# Takes the team name and one hot encodes it.
# Teams have changed names over the past 5 years, so the code combines teams
# final label is based off of 2023 team names
def team_hotness(df_raw):
  temp = df_raw["Team"].astype(str).to_numpy().reshape(-1, 1)
  uniques = np.unique(temp).tolist()

  enc = OneHotEncoder(sparse = False,).fit(temp)
  enc.columns = enc.get_feature_names_out()
  matrix = (enc.transform(temp))

  panda_df = pd.DataFrame(data = matrix,  
                        columns = enc.get_feature_names_out()) 
    
  panda_df["Alfa_Romeo"] = panda_df['x0_Alfa Romeo'] + panda_df['x0_Alfa Romeo Racing'] + panda_df['x0_Sauber']
  panda_df["AlphaTauri"] = panda_df['x0_Toro Rosso'] + panda_df['x0_AlphaTauri']
  panda_df["Alpine"] = panda_df['x0_Alpine'] + panda_df['x0_Renault']
  panda_df["Aston_Martin"] = panda_df["x0_Aston Martin"] + panda_df["x0_Racing Point"] + panda_df['x0_Force India']
  panda_df["Ferrari"] = panda_df["x0_Ferrari"]
  panda_df["Haas"] = panda_df["x0_Haas F1 Team"]
  panda_df["McLaren"] = panda_df["x0_McLaren"]
  panda_df["Mercedes"] = panda_df["x0_Mercedes"]
  panda_df["Red_Bull"] = panda_df["x0_Red Bull Racing"]

  proper_name = ["Alfa_Romeo", "AlphaTauri","Alpine", "Aston_Martin", "Ferrari", 
                 "Haas", "McLaren", "Mercedes","Red_Bull"]
  for name in proper_name:
    df_raw[name] = panda_df[name]
  return None


# gets the one hot encoding of trackstatus
# because they're put in kinda weird, it all got one hot encoded kinda weird
def track_status_hotness(df_raw):
  temp = df_raw["TrackStatus"].astype(str).to_numpy().reshape(-1, 1)
  uniques = np.unique(temp).tolist()

  # custom_fnames_enc = OneHotEncoder(feature_name_combiner=custom_combiner).fit(X)

  enc = OneHotEncoder(sparse = False,).fit(temp)
  enc.columns = enc.get_feature_names_out()
  matrix = (enc.transform(temp))
  
  panda_df = pd.DataFrame(data = matrix,  
                        columns = enc.get_feature_names_out()) 
  panda_df["TrackStatus_1"] = panda_df['x0_1.0'] + panda_df['x0_71.0']
  panda_df["TrackStatus_2"] = (panda_df['x0_2.0'] + panda_df['x0_24.0'] + panda_df['x0_26.0']
                               + panda_df['x0_264.0'] + panda_df['x0_267.0'] 
                               + panda_df['x0_672.0'] + panda_df['x0_6724.0'] + panda_df['x0_72.0']
                               + panda_df['x0_724.0']) # panda_df['x0_52.0']
  panda_df["TrackStatus_4"] = (panda_df["x0_24.0"] + panda_df["x0_264.0"] + panda_df['x0_4.0'] 
                               + panda_df["x0_45.0"] + panda_df["x0_64.0"] + panda_df["x0_6724.0"]
                              + panda_df["x0_724.0"])
  panda_df["TrackStatus_5"] = (  panda_df["x0_45.0"]
                                ) # panda_df["x0_52.0"]
  panda_df["TrackStatus_6"] = (panda_df["x0_26.0"] + panda_df['x0_264.0'] + panda_df['x0_267.0']
                               + panda_df['x0_6.0'] + panda_df['x0_64.0'] + panda_df['x0_67.0']
                               + panda_df['x0_672.0'] + panda_df['x0_6724.0'])
  panda_df["TrackStatus_7"] = (panda_df["x0_267.0"] + panda_df['x0_67.0'] + panda_df['x0_672.0']
                               + panda_df['x0_6724.0'] + panda_df['x0_7.0']
                               + panda_df['x0_71.0'] + panda_df['x0_72.0'] + panda_df['x0_724.0']
                               )
  # panda_df["TrackStatus_nan"] = panda_df['x0_nan']
 
  proper_name = ["TrackStatus_7","TrackStatus_6", "TrackStatus_5", "TrackStatus_4", 
                 "TrackStatus_2", "TrackStatus_1"]
  for name in proper_name:
    df_raw[name] = panda_df[name]
  return None

# takes the raw and a column name of a non-binary categorical data
# then adds those columns in a col_name_type format into the df_raw
def non_special_hotness(df_raw, col_name):
  temp = df_raw[col_name].astype(str).to_numpy().reshape(-1, 1)
  uniques = np.unique(temp).tolist()

  # custom_fnames_enc = OneHotEncoder(feature_name_combiner=custom_combiner).fit(X)

  enc = OneHotEncoder(sparse = False,).fit(temp)
  enc.columns = enc.get_feature_names_out()
  matrix = (enc.transform(temp))
  feat_names = enc.get_feature_names_out()
  proper_name = []
  for name in feat_names:
    proper_name.append(name.replace("x0", col_name))
  panda_df = pd.DataFrame(data = matrix,  
                        columns = proper_name) 
  for name in proper_name:
    df_raw[name] = panda_df[name]

  return None

# one hot encodes all categorical data
def all_the_hotness(df_raw, categorical_data = None):
  team_hotness(df_raw)
  track_status_hotness(df_raw)

  if categorical_data == None:
    categorical_data = ['Driver',
                        'DriverNumber',
                        'Stint',
                        'Compound', 
                        ]
  for col_name in categorical_data:
    non_special_hotness(df_raw, col_name)

def drop_cols(df_raw):
  to_drop = ['DeletedReason',
  'LabelCompound',
  'PitInTime',
  'PitOutTime',
  'FCYStatus',
  # below became one-hot-encoded
  'Driver',
  'DriverNumber',
  'Stint',
  'Compound', 
  'Team',
  'RaceTrackStatus',
  "TrackStatus",
  # Not in original, and many nulls.
  "PitInTimeInSeconds",
  "PitOutTimeInSeconds",
  "PitTimeDifference",
  # in original, many nuls.
  'SpeedI1',
  ]

  # drop them
  df_raw.drop(columns=to_drop, inplace = True)
  return None

# removes a row with a nan value there was no pitstop
# this is done because of a high data imbalance in the data
def remove_if_no_pit(df_raw):
  indices = []
  # print(df_raw.shape)
  for row in df_raw.iterrows():
    num_nulls_in_row = row[1].isna().sum()
    if ((row[1].get("PitStop")) == 0) and (num_nulls_in_row > 0):
      indices.append(row[0])

  df_raw = df_raw.drop(index = indices)
  # print(df_raw.shape)
  return df_raw

# takes an array and a list of columns, 
# returns a dictonary of null values and counts
def nan_counts(df_raw, column_name = None):
  if column_name == None:
    column_name = df_raw.columns
  null_count = {}
  for name in column_name:
    num_nulls = df_raw[name].isna().sum()
    if num_nulls > 0:
      null_count[name] = df_raw[name].isna().sum()
  return null_count

# FIXME: it takes the mean of all values, even though that does not make sense
# for all features
def remove_nulls(df_raw):
  df_new = remove_if_no_pit(df_raw)
  make_zero = [
      "IsPersonalBest",
  ]
  
  for name in make_zero:
    df_new[name] = df_new[name].fillna(0)

  # make_mean = [
  #     'Sector1Time',
  #     'Sector2Time',
  #     'Sector3Time',
  #     'Sector1SessionTime',
  #     'Sector2SessionTime',
  #     'Sector3SessionTime',
  #     'Time',
  #     'LapTime',
  #     'SpeedI1', dropped this guy because of many missing numbers
  #     'SpeedI2',
  #     'SpeedFL',
  #     'SpeedST',
  # ]

  nulls = nan_counts(df_new)
  for name in nulls:
    df_new[name] = df_new[name].fillna(df_new[name].mean)
  
  return df_new  

