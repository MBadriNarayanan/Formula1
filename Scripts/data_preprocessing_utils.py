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
    df = df.reset_index(drop=True)
    return df


def generate_sequence(df, sequence_size=4, flag=False):
    sequence = []
    for idx in range(df.shape[0]):
        data = df.iloc[idx : idx + sequence_size]
        data = np.array(data)
        if data.shape[0] < sequence_size:
            break
        if flag:
            data = data[-1]
        sequence.append(data)
    return sequence


def generate_recurrent_sequence(data_directory, valid_lap_columns):
    recurrent_sequence = []
    recurrent_label_sequence = []
    recurrent_sequence_path = os.path.join(data_directory, "LapSequence.npy")
    recurrent_label_sequence_path = os.path.join(data_directory, "LabelSequence.npy")
    for year_name in tqdm(os.listdir(data_directory)):
        year_path = os.path.join(data_directory, year_name)
        grand_prix_sequence = []
        grand_prix_label_sequence = []
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
            lap_columns = valid_lap_columns + ["PitStop"]
            lap_df = lap_df[lap_columns].reset_index(drop=True)
            lap_sequence = []
            label_sequence = []
            for driver in Counter(lap_df["DriverNumber"]).keys():
                driver_df = lap_df.loc[lap_df["DriverNumber"] == driver].reset_index(
                    drop=True
                )
                y = driver_df["PitStop"]
                driver_df = driver_df[valid_lap_columns]
                driver_sequence = generate_sequence(
                    driver_df, sequence_size=4, flag=False
                )
                label = generate_sequence(y, sequence_size=4, flag=True)
                driver_sequence = np.array(driver_sequence)
                lap_sequence.append(driver_sequence)
                label_sequence.append(label)
            lap_sequence = np.array(lap_sequence)
            label_sequence = np.array(label_sequence)
            grand_prix_sequence.append(lap_sequence)
            grand_prix_label_sequence.append(label_sequence)
        recurrent_sequence.append(grand_prix_sequence)
        recurrent_label_sequence.append(grand_prix_label_sequence)
    recurrent_sequence = np.array(recurrent_sequence)
    recurrent_label_sequence = np.array(recurrent_label_sequence)
    np.save(recurrent_sequence_path, recurrent_sequence)
    np.save(recurrent_label_sequence_path, recurrent_label_sequence)
    return recurrent_sequence, recurrent_label_sequence
