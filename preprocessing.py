import pandas as pd
import numpy as np
import glob
import tqdm
from sklearn.model_selection import train_test_split



def load_omni(omni_path):
    omni_df = []
    for file_name in tqdm.tqdm(glob.glob(omni_path), desc='Loading OMNI data'):
        file = pd.read_csv(file_name, header=0)
        omni_df.append(file)
    omni_df = pd.concat(omni_df)
    omni_df["Timestamp"] = pd.to_datetime(omni_df["Timestamp"])
    omni_df.set_index('Timestamp', inplace=True)
    # TODO omni_df = omni_df[[features we want]]

    return omni_df

def load_goes(goes_path):
    goes_df = []
    for file_name in tqdm.tqdm(glob.glob(goes_path), desc='Loading GOES data'):
        file = pd.read_csv(file_name, header=0)
        goes_df.append(file)
    goes_df = pd.concat(goes_df)
    return goes_df

def load_sat_density(sat_density_path):
    sat_density_df = []
    for file_name in tqdm.tqdm(glob.glob(sat_density_path), desc='Loading CHAMP data'):
        file = pd.read_csv(file_name, header=0)
        sat_density_df.append(file)
    sat_density_df = pd.concat(sat_density_df)
    return sat_density_df



def preprocess(omni_path, goes_path, sat_density_path):
    omni_data = load_omni(omni_path)
    goes_data = load_goes(goes_path)
    champ_data = load_sat_density(sat_density_path)

    # TODO: Change filler values to NaN

    # Interpolate over small data gaps (<10 minutes)
    omni_data.interpolate(method="linear", limit=10)
    goes_data.interpolate(method="linear", limit=10)
    champ_data.interpolate(method="linear", limit=10)

    # First, combine all three dataframes so the same data points are dropped
    all_data = pd.concat([omni_data, goes_data, champ_data], axis=1)
    # Drop nans from each dataset
    all_data.dropna(inplace=True, axis=0)
    # Do a train-valid-test split
    train_valid, test = train_test_split(all_data, test_size=0.15)
    train, valid = train_test_split(train_valid, test_size=0.18)

    # Split champ data from the rest
    champ_train = train[champ_data.columns]
    champ_valid = valid[champ_data.columns]
    champ_test = test[champ_data.columns]
    # "input" refers to both omni and goes data
    input_train = train.drop(champ_data.columns, axis=1)
    input_valid = valid.drop(champ_data.columns, axis=1)
    input_test = test.drop(champ_data.columns, axis=1)

    # TODO: Feature engineering: remove unneeded features and create new features if we want

    # TODO: Save all the input and target data to individual files, which will be loaded later by a training script

if __name__ == "__main__":
    path_start = "C:\\Users\\Raman\\Downloads\\phase_1\\"
    omni_path = path_start + "omni2\*.csv"
    sat_density_path = path_start + "sat_density\*.csv"
    goes_path = path_start + "goes\*.csv"
    preprocess(omni_path, goes_path, sat_density_path)