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

# def save_data(data, fname, fpath='data/'):
#     with open(fpath+fname+'.pkl', 'wb') as file:
#         pickle.dump(data, file)

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
                       "SW_Plasma_Speed_km_s", "SW_Proton_Density_N_cm3", "SW_Plasma_Temperature_K", "Flow_pressure", "E_electric_field","f10.7_index","AE_index_nT","Kp_index"]]

    print("Created OMNI dataframe")
    return omni_df

def load_goes(goes_path):
    goes_df = []
    for file_name in tqdm.tqdm(glob.glob(goes_path), desc='Loading GOES data'):
        file = pd.read_csv(file_name, header=0, usecols=["Timestamp", "xrsa_flux", "xrsa_flag", "xrsa_flag_excluded"])
        goes_df.append(file)
    goes_df = pd.concat(goes_df)
    goes_df["Timestamp"] = pd.to_datetime(goes_df["Timestamp"])
    goes_df.set_index('Timestamp', inplace=True, drop=True)
    print("Created GOES dataframe")
    return goes_df
#xrsb_flux_electrons or xrsb_flux_observed???,

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
    omni_data.interpolate(method="linear", limit=10)
    goes_data.interpolate(method="linear", limit=10)
    champ_data.interpolate(method="linear", limit=10)
    
    omni_data = omni_data[~omni_data.index.duplicated(keep='first')]
    goes_data = goes_data[~goes_data.index.duplicated(keep='first')]
    champ_data = champ_data[~champ_data.index.duplicated(keep='first')]
    

    
    # First, combine all three dataframes so the same data points are dropped
    all_data = pd.concat([omni_data, goes_data, champ_data], axis=1)
    print(all_data)
    import pdb; pdb.set_trace()
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
    
    # set timestamps as index
    all_data.set_index('Timestamp', inplace=True)
    
    # save dataset
    print("Saving Data")
    fname = 'merged_data'
    all_data.to_pickle('data/'+fname+'.pkl')


    # TODO: Feature engineering: remove unneeded features and create new features if we want(do it better)
    

    
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