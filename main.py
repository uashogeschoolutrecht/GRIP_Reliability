
#%%
'''
Main script for the GRIP HAP project.
'''
# TODO

import logging
from datetime import date
from src.ProcesRawData import process_IMU_data
from src.MergeAndCleanData import merge_and_clean_data
from src.analyseData import evaluate_models, evaluate_definitive_model, create_definitive_model
logging.basicConfig(filename=f'logging/warnings_{date.today()}.log', level=logging.WARN)

# General project settings.
# More specific settings can be found in the config_file.
settings = {
    'VERBOSE' : True,
    'VISUALISE' : True,
    'PROCESS_ALL_DATA':False, #set run_all to True to rerun all data
    'CONFIG_DIR' : "config",
    'RAW_DATA_DIR' : 'data/Raw_data',
    'CLEAN_DATA_DIR' : 'data/Clean_data',
    'MERGED_DATA_DIR' : 'data/Merged_data',
    'PEAKS_DATA_DIR' : 'data/peaks'
    }
#%%

def main():
    '''
    Uncomment the required functions
    '''
    # process_IMU_data(settings) 
    # merge_and_clean_data(settings) 
    # evaluate_models(settings)
    # evaluate_definitive_model(settings, model = 'CNN_model.keras') # Add the name of the best performing model
    create_definitive_model(settings, model = 'CNN_model.keras') # Add the name of the best performing model

if __name__ == '__main__':
    main()
 
#%%