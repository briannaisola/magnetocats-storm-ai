import pandas as pd
import numpy as np
import glob
import tqdm


def load_omni(omni_path):
    omni_df = []
    for file_name in tqdm.tqdm(glob.glob(omni_path), desc='Loading OMNI data'):
        file = pd.read_csv(file_name, header=None)
        omni_df.append(file)
    omni_df = pd.concat(omni_df)
    return omni_df

def load_sat_density(sat_density_path):
    sat_density_df = []
    for file_name in tqdm.tqdm(glob.glob(sat_density_path), desc='Loading CHAMP data'):
        file = pd.read_csv(file_name, header=None)
        sat_density_df.append(file)
    sat_density_df = pd.concat(sat_density_df)
    return sat_density_df

def load_goes(goes_path):
    goes_df = []
    for file_name in tqdm.tqdm(glob.glob(goes_path), desc='Loading GOES data'):
        file = pd.read_csv(file_name, header=None)
        goes_df.append(file)
    goes_df = pd.concat(goes_df)
    return goes_df

def preprocess(omni_path, goes_path, sat_density_path):
    load_omni(omni_path)
    load_sat_density(sat_density_path)
    load_sat_density(goes_path)

if __name__ == "__main__":
    path_start = "C:\\Users\\Raman\\Downloads\\phase_1\\"
    omni_path = path_start + "omni2\*.csv"
    sat_density_path = path_start + "sat_density\*.csv"
    goes_path = path_start + "goes\*.csv"
    preprocess(omni_path, goes_path, sat_density_path)