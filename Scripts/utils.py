import fastf1
import os
import shutil

import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm

race_track_status = {
    "Austin": 2,
    "Baku": 2,
    "Budapest": 2,
    "Catalunya": 1,
    "Barcelona": 1,
    "Spain": 1,
    "Hockenheim": 2,
    "KualaLumpur": 2,
    "LeCastellet": 2,
    "Melbourne": 2,
    "MexicoCity": 2,
    "MonteCarlo": 3,
    "Monaco": 3,
    "Montréal": 3,
    "Monza": 2,
    "Sakhir": 1,
    "Bahrain": 1,
    "SãoPaulo": 1,
    "Shanghai": 2,
    "Silverstone": 1,
    "Singapore": 3,
    "MarinaBay": 3,
    "Sochi": 2,
    "SpaFrancorchamps": 1,
    "Spielberg": 2,
    "Suzuka": 1,
    "YasMarina": 3,
    "YasIsland": 3,
}


def create_directory(directory_path):
    if os.path.exists(directory_path):
        print("{} Directory exists!".format(directory_path))
        print("Removing contents!")
        shutil.rmtree(directory_path)
    print("Creating {} directory!".format(directory_path))
    os.makedirs(directory_path)


def get_percentage(x, total_laps):
    return round((x / total_laps), 3)


def get_switched_compounds_data(lap_data, result_data):
    lap_df = pd.DataFrame()
    for driver_name in result_data["Abbreviation"]:
        driver_df = lap_data.loc[lap_data["Driver"] == driver_name].reset_index(
            drop=True
        )
        switched_compounds_list = []
        for index in range(len(driver_df)):
            if index == 0 or index == 1:
                switched_compounds_list.append(False)
            else:
                if (
                    driver_df.iloc[index]["Compound"]
                    == driver_df.iloc[index - 1]["Compound"]
                ):
                    switched_compounds_list.append(False)
                else:
                    switched_compounds_list.append(True)
        driver_df["SwitchedCompounds"] = switched_compounds_list
        lap_df = pd.concat([lap_df, driver_df], ignore_index=True)
    return lap_df


def convert_compound(x):
    if x == "SOFT":
        return 1
    elif x == "MEDIUM":
        return 2
    elif x == "HARD":
        return 3
    return 4


def get_close_pursuer_leader(lap_data, total_laps):
    lap_df = pd.DataFrame()
    for lap_number in range(1, total_laps + 1):
        individual_lap_data = lap_data.loc[(lap_data["LapNumber"] == float(lap_number))]
        individual_lap_data = individual_lap_data.sort_values(
            by=["Position"]
        ).reset_index(drop=True)
        pursuer_list = []
        leader_list = []
        for index in range(len(individual_lap_data)):
            lap_time = individual_lap_data.iloc[index]["LapTime"].total_seconds()
            try:
                pursuer_lap_time = individual_lap_data.iloc[index + 1][
                    "LapTime"
                ].total_seconds()
            except:
                pursuer_lap_time = np.inf
            if index == 0:
                leader_lap_time = np.inf
            else:
                leader_lap_time = individual_lap_data.iloc[index - 1][
                    "LapTime"
                ].total_seconds()

            if lap_time - pursuer_lap_time > 1.5:
                pursuer_list.append(False)
            else:
                pursuer_list.append(True)
            if lap_time - leader_lap_time > 1.5:
                leader_list.append(False)
            else:
                leader_list.append(True)
        individual_lap_data["CloseLeader"] = leader_list
        individual_lap_data["ClosePursuer"] = pursuer_list
        lap_df = pd.concat([lap_df, individual_lap_data], ignore_index=True)
    return lap_df


def get_fcy_status(track_status):
    if track_status != track_status:
        return np.nan
    track_status_dict = {str(idx): 0 for idx in range(10)}
    for char in track_status:
        track_status_dict[char] += 1
    return "".join([str(value) for value in track_status_dict.values()])


def prev_follower_pitted_func(lap_data, index):
    driver_num = lap_data.loc[index, "DriverNumber"]
    lap_num = lap_data.loc[index, "LapNumber"]

    if lap_num == 1 or lap_num == 2 or lap_num == 0:
        return np.nan, np.nan, np.nan

    orig_driver_prev_pos = (
        lap_data.pick_lap(lap_number=lap_num - 1)
        .loc[lap_data["DriverNumber"] == driver_num]["Position"]
        .iloc[0]
    )
    if orig_driver_prev_pos == 20:
        return np.nan, np.nan, np.nan

    prev_follower = lap_data.pick_lap(lap_number=lap_num - 1).loc[
        lap_data["Position"] == (orig_driver_prev_pos + 1)
    ]
    prev_follower_pitted = 0
    try:
        if (not pd.isnull(prev_follower["PitInTime"].iloc[0])) and lap_num != 1.0:
            prev_follower_pitted = 1
    except:
        return np.nan, np.nan, np.nan
    prev_follower_stint = int(prev_follower["Stint"].iloc[0])
    prev_follower_pace = prev_follower["LapTime"].iloc[0].total_seconds()
    return prev_follower_pitted, prev_follower_stint, prev_follower_pace


def drivers_fastest_laps(laps):
    drivers_fastest = {}
    for un_d in laps["DriverNumber"].unique():
        fastest_lap_time = laps.pick_drivers(un_d).pick_fastest()["LapTime"]
        if fastest_lap_time == fastest_lap_time:
            fastest_lap_time = fastest_lap_time.total_seconds()
        drivers_fastest[un_d] = fastest_lap_time
    return drivers_fastest


def generate_fastest_lap(driver_number, lap_time, fastest_laps):
    if lap_time != lap_time:
        return False
    lap_time = lap_time.total_seconds()
    if fastest_laps[driver_number] == lap_time:
        return True
    return False


def get_fastest_lap_data(lap_data):
    fastest_laps = drivers_fastest_laps(lap_data)
    fastest_lap_of_race = -100000
    for lap_time in fastest_laps.values():
        if lap_time < fastest_lap_of_race:
            fastest_lap_of_race = lap_time
    lap_data["FastestLapOfRace"] = lap_data.apply(
        lambda x: True
        if x["LapTime"].total_seconds() == fastest_lap_of_race
        else False,
        axis=1,
    )
    return lap_data


def get_in_front_info(lap_data, index):
    pos = lap_data.loc[index, "Position"]
    lap_num = lap_data.loc[index, "LapNumber"]
    if pos == 1 or pos == None:
        return np.nan, np.nan, np.nan, np.nan
    driver_in_front = lap_data.pick_lap(lap_number=lap_num).loc[
        lap_data["Position"] == (pos - 1)
    ]
    if driver_in_front.empty:
        return np.nan, np.nan, np.nan, np.nan

    in_front_pitted = 0
    if (not pd.isnull(driver_in_front["PitInTime"].iloc[0])) and lap_num != 1.0:
        in_front_pitted = 1
    if driver_in_front["Stint"].iloc[0] == driver_in_front["Stint"].iloc[0]:
        in_front_stint = int(driver_in_front["Stint"].iloc[0])
    else:
        in_front_stint = np.nan
    in_front_pace = driver_in_front["LapTime"].iloc[0].total_seconds()
    in_front_time = driver_in_front["Sector2SessionTime"].iloc[0]
    return in_front_pace, in_front_time, in_front_stint, in_front_pitted


def get_pursuer_info(lap_data, index):
    pos = lap_data.loc[index, "Position"]
    lap_num = lap_data.loc[index, "LapNumber"]
    if pos == 20:
        return np.nan, np.nan, np.nan

    try:
        driver_following = lap_data.pick_lap(lap_number=lap_num).loc[
            lap_data["Position"] == (pos + 1)
        ]
        pursuer_stint = int(driver_following["Stint"].iloc[0])

    except:
        return np.nan, np.nan, np.nan
    pursuer_pace = driver_following["LapTime"].iloc[0].total_seconds()
    pursuer_time = driver_following["Sector2SessionTime"].iloc[0]
    return pursuer_pace, pursuer_time, pursuer_stint


def construct_lap_data(lap_data):
    column_names = [
        "PrevFollowerPitted",
        "PrevFollowerStint",
        "PrevFollowerPace",
        "InFrontPace",
        "InFrontTime",
        "InFrontStint",
        "InFrontPitted",
        "PursuerPace",
        "PursuerTime",
        "PursuerStint",
    ]
    lap_data = get_fastest_lap_data(lap_data)
    lap_df = pd.DataFrame(lap_data)
    lap_df[column_names] = np.nan

    for index in lap_data.index:
        (
            prev_follower_pitted,
            prev_follower_stint,
            prev_follower_pace,
        ) = prev_follower_pitted_func(lap_data, index)
        (
            in_front_pace,
            in_front_time,
            in_front_stint,
            in_front_pitted,
        ) = get_in_front_info(lap_data, index)
        pursuer_pace, pursuer_time, pursuer_stint = get_pursuer_info(lap_data, index)

        var_names = [
            prev_follower_pitted,
            prev_follower_stint,
            prev_follower_pace,
            in_front_pace,
            in_front_time,
            in_front_stint,
            in_front_pitted,
            pursuer_pace,
            pursuer_time,
            pursuer_stint,
        ]

        for idx in range(len(var_names)):
            lap_df.loc[index, column_names[idx]] = var_names[idx]
    return lap_df


def create_data_frame(
    data_directory, valid_years, valid_result_columns, valid_lap_columns
):
    for year in valid_years:
        year_directory = os.path.join(data_directory, "Year_{}".format(year))
        create_directory(year_directory)

        schedule = fastf1.get_event_schedule(year)
        for index, row in tqdm(schedule.iterrows()):
            if row["RoundNumber"] != 0:
                event_name = row["EventName"]
                location = row["Location"]
                location = "".join(
                    character for character in location if character.isalnum()
                )
                session = fastf1.get_session(int(year), event_name, "Race")
                session.load(laps=True)

                result_data = session.results
                lap_data = session.laps
                total_laps = session.total_laps

                lap_data["LabelCompound"] = lap_data["Compound"].apply(
                    lambda x: convert_compound(x)
                )
                lap_data["LapsCompleted"] = lap_data["LapNumber"].apply(
                    lambda x: get_percentage(x, total_laps)
                )
                lap_data["TyreAge"] = lap_data["TyreLife"].apply(
                    lambda x: get_percentage(x, total_laps)
                )
                lap_data = get_switched_compounds_data(lap_data, result_data)
                lap_data = get_close_pursuer_leader(lap_data, total_laps)
                lap_data["RaceTrackStatus"] = lap_data.apply(
                    lambda x: race_track_status.get(location, 0), axis=1
                )
                lap_data["FCYStatus"] = lap_data["TrackStatus"].apply(
                    lambda track_status: get_fcy_status(track_status)
                )
                lap_df = construct_lap_data(lap_data)

                event_name = "".join(event_name.split())
                event_directory = os.path.join(year_directory, event_name)
                create_directory(event_directory)

                lap_filepath = os.path.join(event_directory, "LapData.csv")
                result_filepath = os.path.join(event_directory, "Result.csv")

                print("Event Name: {}".format(event_name))
                print("Event Directory: {}".format(event_directory))
                print("Lap FilePath: {}".format(lap_filepath))
                print("Result FilePath: {}".format(result_filepath))

                lap_df = lap_df[valid_lap_columns].reset_index(drop=True)
                lap_df.to_csv(lap_filepath, index=False)

                result_data = result_data[valid_result_columns].reset_index(drop=True)
                result_data.to_csv(result_filepath, index=False)


def convert_time_to_seconds(x):
    if x != x:
        return np.nan
    lap_time = x.split("days")[1].strip().split(":")
    total_time = (
        (float(lap_time[0]) * 3600) + (float(lap_time[1]) * 60) + (float(lap_time[2]))
    )
    return total_time


def convert_time_to_seconds(x):
    if x != x:
        return np.nan
    lap_time = x.split("days")[1].strip().split(":")
    total_time = (
        (float(lap_time[0]) * 3600) + (float(lap_time[1]) * 60) + (float(lap_time[2]))
    )
    return total_time


def get_pit_stop_duration(in_time, out_time):
    if (in_time == in_time) and (out_time == out_time):
        return abs(in_time - out_time)
    return np.nan


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


def preprocess_dataframe(data_directory):
    lap_dataframe = pd.DataFrame()
    lap_dataframe_path = os.path.join(data_directory, "LapData.csv")
    for year_name in os.listdir(data_directory):
        year_path = os.path.join(data_directory, year_name)
        year_dataframe = pd.DataFrame()
        year_dataframe_path = os.path.join(year_path, "LapData.csv")
        for grand_prix in os.listdir(year_path):
            grand_prix_path = os.path.join(year_path, grand_prix)
            lap_csv_path = os.path.join(grand_prix_path, "PreprocessedLapData.csv")
            result_csv_path = os.path.join(grand_prix_path, "Result.csv")
            lap_df = pd.read_csv(lap_csv_path)
            result_df = pd.read_csv(result_csv_path)
            result_dict = {
                row["DriverNumber"]: row["Position"]
                for idx, row in result_df.iterrows()
            }
            lap_df["FinishingPosition"] = lap_df["DriverNumber"].apply(
                lambda x: int(result_dict[x])
            )
            lap_df["LapTimeInSeconds"] = lap_df["LapTime"].apply(
                lambda x: convert_time_to_seconds(x)
            )
            lap_df["PitInTimeInSeconds"] = lap_df["PitInTime"].apply(
                lambda x: convert_time_to_seconds(x)
            )
            lap_df["PitOutTimeInSeconds"] = lap_df["PitOutTime"].apply(
                lambda x: convert_time_to_seconds(x)
            )
            lap_df["PitTimeDifference"] = lap_df.apply(
                lambda x: get_pit_stop_duration(
                    x["PitInTimeInSeconds"] - x["PitOutTimeInSeconds"]
                ),
                axis=1,
            )
            lap_df = generate_labels(lap_df)
            lap_df.to_csv(lap_csv_path, index=False)
            year_dataframe = pd.concat([year_dataframe, lap_df])
        year_dataframe = year_dataframe.reset_index(drop=True)
        year_dataframe.to_csv(year_dataframe_path, index=False)
        lap_dataframe = pd.concat([lap_dataframe, year_dataframe])
    lap_dataframe = lap_dataframe.reset_index(drop=True)
    lap_dataframe.to_csv(lap_dataframe_path, index=False)


def filter_dataframe(data_directory):
    lap_dataframe_path = os.path.join(data_directory, "LapData.csv")
    preprocessed_lap_dataframe_path = os.path.join(data_directory, "PreprocessedLapData.csv")
    dataframe = pd.read_csv(lap_dataframe_path)
    df = dataframe.loc[dataframe['LabelCompound'] != 4]
    df = df.loc[df['FinishingPosition'] <= 10]
    df = df.loc[df["PitTimeDifference"] <= 50]
    df = df.loc[df["LapTimeInSeconds"] <= 200]
    df = df.loc[(df["LapsCompleted"] >= 0.1) & (df["LapsCompleted"] <= 0.9)]
    df = df.reset_index(drop = True)
    dataframe.to_csv(preprocessed_lap_dataframe_path, index = False)