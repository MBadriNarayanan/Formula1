import fastf1
import os
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm


def create_directory(directory_path):
    if os.path.exists(directory_path):
        print("{} Directory exists!".format(directory_path))
        print("Removing contents!")
        shutil.rmtree(directory_path)
    print("Creating {} directory!".format(directory_path))
    os.makedirs(directory_path)


def get_percentage_laps(lap_number, total_laps):
    return round((lap_number / total_laps) * 100, 3)


def get_switched_compounds_data(driver_df):
    switched_compounds_data = []
    for index in range(driver_df.shape[0]):
        if index == 0 or index == 1:
            switched_compounds_data.append(np.nan)
        else:
            if (
                driver_df.iloc[index]["Compound"]
                == driver_df.iloc[index - 1]["Compound"]
            ):
                switched_compounds_data.append(False)
            else:
                switched_compounds_data.append(True)
    driver_df["SwitchedCompounds"] = switched_compounds_data
    return driver_df


def create_data_frame(
    data_directory, valid_years, valid_result_columns, valid_lap_columns
):
    for year in valid_years:
        year_directory = os.path.join(data_directory, "Year_{}".format(year))
        create_directory(year_directory)

        schedule = fastf1.get_event_schedule(year)
        for idx, row in tqdm(schedule.iterrows()):
            if row["RoundNumber"] != 0:
                event_name = row["EventName"]
                session = fastf1.get_session(year, event_name, "Race")
                session.load(telemetry=True, laps=True, weather=True)

                result_data = session.results
                lap_data = session.laps

                event_name = "".join(event_name.split())
                event_directory = os.path.join(year_directory, event_name)
                create_directory(event_directory)

                result_filepath = os.path.join(event_directory, "Result.csv")
                total_laps = session.total_laps

                print("Event Name: {}".format(event_name))
                print("Event Directory: {}".format(event_directory))
                print("Result FilePath: {}".format(result_filepath))

                result_data = result_data[valid_result_columns].reset_index(drop=True)
                result_data.to_csv(result_filepath, index=False)

                for driver_name in result_data["Abbreviation"]:
                    driver_df = lap_data.loc[
                        lap_data["Driver"] == driver_name
                    ].reset_index(drop=True)
                    driver_df = get_switched_compounds_data(driver_df)
                    driver_df["PercentageLapsCompleted"] = driver_df["LapNumber"].apply(
                        lambda x: get_percentage_laps(x, total_laps)
                    )

                    driver_filepath = os.path.join(
                        event_directory, "{}.csv".format(driver_name)
                    )
                    print("Driver Filepath: {}".format(driver_filepath))

                    driver_df = driver_df[valid_lap_columns]
                    driver_df.to_csv(driver_filepath, index=False)
