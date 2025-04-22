import pandas as pd
import numpy as np
import glob
import tqdm
import pickle
from sklearn.model_selection import train_test_split
# from hapiclient import hapi
# from hapiclient  import hapitime2datetime
# from hapiclient import hapi
# from hapiclient  import hapitime2datetime
from datetime import datetime

def save_data(data, fname, fpath='data/'):
    with open(fpath+fname+'.pkl', 'wb') as file:
        pickle.dump(data, file)

# def write2df(data_dir, fname='omni', filler_map=None):
#     """
#     Read and combine csv data and pickle as pandas dataframe (df).
#     (Optional) apply filler map and replace bad values with NaNs
    
#     """
#     all_files = glob.glob(data_dir)
    
#     df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
#     if filler_map is not None:
#         # replace all filler values with NaNs
#         df = df.applymap(lambda x: np.nan if float(x) in filler_map else x)
        
#     # set timestamps as index
#     df.set_index('Timestamp', inplace=True)
    
#     # save dataset
#     df.to_pickle('data/'+fname+'.pkl')
        
# def omni_filler_cleanup(omni_df):
    
#     # get omni metadata
#     server = "https://cdaweb.gsfc.nasa.gov/hapi" # hapi server url
#     dataset = "OMNI_HRO_5MIN" # OMNI dataset
#     meta = hapi(server, dataset) # pull meta 
    
#     filler = []
#     # loop over all omni paramters in meta (ignoring 'time') and collect unique fill values
#     for meta_var in meta['parameters'][1:]:  
#         fill_val = meta_var['fill'] # get filler value (e.g. 9999.99)
        
#         # check if exists and save float to list
#         if float(fill_val) not in filler:
#             filler.append(float(fill_val))
            
#     # replace all filler values with NaNs
#     omni_df = omni_df.applymap(lambda x: np.nan if float(x) in filler else x)
    
#     return omni_df

import warnings
warnings.filterwarnings('ignore')

def create_delays(df, name, time):
    for delay in np.arange(1, int(time) + 1):
        df[name + '_%s' % delay] = df[name].shift(delay).astype('float32')


def load_initial_states(initial_states_path):
    initial_states_df = []
    for file_name in tqdm.tqdm(glob.glob(initial_states_path), desc='Loading Initial States data'):
        file = pd.read_csv(file_name, header=0, usecols=["Timestamp", "Latitude (deg)", "Longitude (deg)", "Altitude (km)"] )
        initial_states_df.append(file)
    initial_states_df = pd.concat(initial_states_df)
    initial_states_df["Timestamp"] = pd.to_datetime(initial_states_df["Timestamp"])
    initial_states_df.set_index('Timestamp', inplace=True)

    print("Created Initial States dataframe")
    return initial_states_df

def load_omni(omni_path):
    omni_df = []
    for file_name in tqdm.tqdm(glob.glob(omni_path), desc='Loading OMNI data'):
        file = pd.read_csv(file_name, header=0)
        omni_df.append(file)
    omni_df = pd.concat(omni_df)
    omni_df["Timestamp"] = pd.to_datetime(omni_df["Timestamp"])
    omni_df.set_index('Timestamp', inplace=True, drop=True)
    omni_df = omni_df[["Scalar_B_nT", "BX_nT_GSE_GSM", "BY_nT_GSM", "BZ_nT_GSM",
                       "SW_Plasma_Speed_km_s", "SW_Proton_Density_N_cm3", "SW_Plasma_Temperature_K", "E_electric_field","f10.7_index","AE_index_nT","Kp_index"]]

    print("Created OMNI dataframe")
    return omni_df

def load_goes(goes_path):
    goes_df = []
    for file_name in tqdm.tqdm(glob.glob(goes_path), desc='Loading GOES data'):
        file = pd.read_csv(file_name, header=0, usecols=["Timestamp", "xrsa_flux"])
        goes_df.append(file)
    goes_df = pd.concat(goes_df)
    goes_df["Timestamp"] = pd.to_datetime(goes_df["Timestamp"])
    goes_df.set_index('Timestamp', inplace=True, drop=True)
    print("Created GOES dataframe")
    return goes_df

def load_sat_density(sat_density_path):
    sat_density_df = []
    for file_name in tqdm.tqdm(glob.glob(sat_density_path), desc='Loading champ data'):
        file = pd.read_csv(file_name, header=0)
        sat_density_df.append(file)
    sat_density_df = pd.concat(sat_density_df)
    sat_density_df["Timestamp"] = pd.to_datetime(sat_density_df["Timestamp"])
    sat_density_df.set_index('Timestamp', inplace=True, drop=True)
    print("Created Sat Density dataframe")
    return sat_density_df

def preprocess(initial_states_path, omni_path, goes_path, sat_density_path):
    initial_states_data = load_initial_states(initial_states_path)
    omni_data = load_omni(omni_path)
    goes_data = load_goes(goes_path)
    champ_data = load_sat_density(sat_density_path)

    # TODO: Change filler values to NaN
    
    # )
    # omni_data = omni_filler_cleanup(omni_data)
    
    # replace all champ data filler values with NaNs
    # champ_filler = 9.990000e+32
    # champ_data = champ_data.applymap(lambda x: np.nan if float(x) in champ_filler else x)
    
    
    # Interpolate over small data gaps (<10 minutes)
    print("Interpolating over data gaps")
    goes_data.interpolate(method="linear", limit=10)
    champ_data.interpolate(method="linear", limit=1)  # sat_density is at 10 cadence
    
    # Drop duplicate timestamps
    print("Dropping duplicate timestamps")
    omni_data = omni_data[~omni_data.index.duplicated(keep='first')]
    goes_data = goes_data[~goes_data.index.duplicated(keep='first')]
    champ_data = champ_data[~champ_data.index.duplicated(keep='first')]

    # Upsample omni data to 10 minute intervals
    print("Upsampling OMNI data to 10 minute intervals")
    omni_data = omni_data.resample("10T").mean()
    omni_data.interpolate(method='time', limit_direction='both', inplace=True, limit=6)


    # First, combine all three dataframes so the same data points are dropped
    print("Combining all dataframes")
    all_data = pd.concat([omni_data, goes_data, champ_data], axis=1)

    print("Dropping NaNs")
    all_data.dropna(inplace=True, axis=0)

    omni_data = all_data[omni_data.columns]
    goes_data = all_data[goes_data.columns] 
    champ_data = all_data[champ_data.columns]

    del all_data

    long_history_step = 6*24*28  # 6 steps per hour * 24 hr/day * ~28 day / Carrington rotation
    short_history_step = 6*24*2  # 6 steps per hour * 24 hr/day * 2 day magnetosphere memory
    forecast_step = 3*24*6  # 6 steps per hour * 24 hr/day * 3 day forecast window
    long_delay_features = [
        "SW_Plasma_Speed_km_s",
        "SW_Proton_Density_N_cm3",
        "SW_Plasma_Temperature_K",
    ]
    short_delay_features = [
        "Scalar_B_nT",
        "BX_nT_GSE_GSM",
        "BY_nT_GSM",
        "BZ_nT_GSM",
        "E_electric_field",
        "f10.7_index",
        "AE_index_nT",
        "Kp_index",
        "xrsa_flux",
    ]

        
    
    # Create new columns in the omni_data and goes_data to represent the time history
    for feature in tqdm.tqdm(long_delay_features, desc="Creating OMNI long time history"):
        if feature in omni_data.columns:
            create_delays(omni_data, feature, long_history_step)
    omni_data.dropna(inplace=True, axis=0)

    for feature in tqdm.tqdm(short_delay_features, desc="Creating OMNI short time history"):
        if feature in omni_data.columns:
            create_delays(omni_data, feature, short_history_step)
    omni_data.dropna(inplace=True, axis=0)

    for feature in tqdm.tqdm(short_delay_features, desc="Creating GOES time history"):
        if feature in goes_data.columns:
            create_delays(goes_data, feature, short_history_step)
    goes_data.dropna(inplace=True, axis=0)
    
    # Create new columns in the target data to represent the forecast window
    print("Creating satellite density forecast window")
    create_delays(champ_data, "Orbit Mean Density (kg/m^3)", forecast_step)
    champ_data.dropna(inplace=True, axis=0)
    

    # For each row in initial_states_data, create a vector that has the initial state and concat it with omni_data, goes_data, and champ_data at the same timestamp
    input_data = pd.DataFrame(np.zeros(len(initial_states_data), len(omni_data.columns)+len(goes_data.columns)))
    target_data = pd.DataFrame(np.zeros(len(initial_states_data, len(champ_data.columns))))
    input_data.index = initial_states_data.index
    target_data.index = initial_states_data.index
    for timestamp in initial_states_data.index:
        input_data.loc[timestamp] = pd.concat([initial_states_data.loc[timestamp],
                                                omni_data.loc[timestamp],
                                                goes_data.loc[timestamp]], axis=1)
        target_data.loc[timestamp] = champ_data.loc[timestamp]
    
    del initial_states_data, omni_data, goes_data, champ_data

    combined_df = pd.concat([input_data, target_data], axis=1)
    
    # Do a train-valid-test split
    print("Doing train-test-valid split")
    train_valid, test = train_test_split(combined_df, test_size=0.15)
    train, valid = train_test_split(train_valid, test_size=0.18)


    # Split champ data from the rest
    champ_train = train[target_data.columns]
    champ_valid = valid[target_data.columns]
    champ_test = test[target_data.columns]
    # "input" refers to both omni and goes data
    input_train = train.drop(target_data.columns, axis=1)
    input_valid = valid.drop(target_data.columns, axis=1)
    input_test = test.drop(target_data.columns, axis=1)

    del champ_data, combined_df, input_data, target_data

    # save dataset
    print("Saving Data")
    
    # [X] TODO: (nearly done)  Save all the input and target data to individual files, which will be loaded later by a training script (Mule!)
    fpath='data/'
    save_data(champ_train, 'champ_train', fpath=fpath)
    save_data(champ_valid, 'champ_valid', fpath=fpath)
    save_data(champ_test, 'champ_test', fpath=fpath)
    save_data(input_train, 'input_train', fpath=fpath)
    save_data(input_valid, 'input_valid', fpath=fpath)
    save_data(input_test, 'input_test', fpath=fpath)

    


if __name__ == "__main__":
    path_start = "../data/dataset/"
    initial_states_path = path_start + "*initial_states.csv"
    omni_path = path_start + "omni2/*.csv"
    sat_density_path = path_start + "sat_density/*.csv"
    goes_path = path_start + "goes/*.csv"
    preprocess(initial_states_path, omni_path, goes_path, sat_density_path)
    # load_goes(goes_path)
    
    
    # get omni metadata
    # server = "https://cdaweb.gsfc.nasa.gov/hapi" # hapi server url
    # dataset = "OMNI_HRO_5MIN" # OMNI dataset
    # meta = hapi(server, dataset) # pull meta 
    
    # omni_filler = []
    # # loop over all omni paramters in meta (ignoring 'time') and collect unique fill values
    # for meta_var in meta['parameters'][1:]:  
    #     fill_val = meta_var['fill'] # get filler value (e.g. 9999.99)
        
    #     # check if exists and save float to list
    #     if float(fill_val) not in omni_filler:
    #         omni_filler.append(float(fill_val))
            
    # write data to pandas df files; save as .pkl
    # write2df(omni_path, fname='omni', filler_map=None)
    # write2df(sat_density_path, fname='sat', filler_map=None)
    # write2df(goes_path, fname='goes', filler_map=None)