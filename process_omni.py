import csv
import pickle
import glob
import numpy as np
import pandas as pd
from hapiclient import hapi
from hapiclient  import hapitime2datetime
from datetime import datetime

# dropbox_url = "https://www.dropbox.com/scl/fo/ilxkfy9yla0z2ea97tfqv/AB9lngJ2yHvf9t5h2oQXaDc?rlkey=iju8q5b1kxol78kbt0b9tcfz3&e=2&st=vzi32eq6&dl=0"
path_start = "../data/dataset/"
omni_dir = omni_path = path_start + "omni2/*.csv"

# omni_fname = "omni2-00007-20000705_to_20000903.csv"

if __name__ == "__main__":
    
    # read csv data and save to pandas dataframe (df)
    all_files = glob.glob(omni_dir)
    
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    # get omni metadata
    server = "https://cdaweb.gsfc.nasa.gov/hapi" # hapi server url
    dataset = "OMNI_HRO_5MIN" # OMNI dataset
    meta = hapi(server, dataset) # pull meta 
    
    filler = []
    # loop over all omni paramters in meta (ignoring 'time') and collect unique fill values
    for meta_var in meta['parameters'][1:]:  
        fill_val = meta_var['fill'] # get filler value (e.g. 9999.99)
        
        # check if exists and save float to list
        if float(fill_val) not in filler:
            filler.append(float(fill_val))
            
    # replace all filler values with NaNs
    df = df.applymap(lambda x: np.nan if float(x) in filler else x)
    
    # set timestamps as index
    df.set_index('Timestamp', inplace=True)
    
    # save dataset
    df.to_pickle('data/'+'omni'+'.pkl')

    # check
    print(df.head(5))
    
    def write2df(data_dir, fname='omni', filler_map=None):
        # read csv data and save to pandas dataframe (df)
        all_files = glob.glob(data_dir)
        
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        
        if filler_map is not None:
            # replace all filler values with NaNs
            df = df.applymap(lambda x: np.nan if float(x) in filler_map else x)
            
        # set timestamps as index
        df.set_index('Timestamp', inplace=True)
        
        # save dataset
        df.to_pickle('data/'+fname+'.pkl')
        
            
            
    