import pandas as pd
import numpy as np
import glob
import tqdm


def load_omni(omni_path):
    omni_df = []
    for file_name in tdqm.tqdm(glob.glob(omni_path), desc='Loading OMNI data'):
        file = pd.read_csv(file_name, header=None)
        omni_df.append(file)
    omni_df = pd.concat(omni_df)

def load_sat_density(sat_density_path):
    sat_density_df = []
    for file_name in tdqm.tqdm(glob.glob(sat_density_path), desc='Loading CHAMP data'):
        file = pd.read_csv(file_name, header=None)
        sat_density_df.append(file)
    sat_density_df = pd.concat(sat_density_df)

def load_goes(goes_path):
    goes_df = []
    for file_name in tdqm.tqdm(glob.glob(goes_path), desc='Loading GOES data'):
        file = pd.read_csv(file_name, header=None)
        goes_df.append(file)
    goes_df = pd.concat(goes_df)

def preprocess():
    pass