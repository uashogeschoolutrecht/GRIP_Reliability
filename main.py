
# %%
'''
Main script for the GRIP HAP project.

- Run the script to process raw data
- Get activity scores for every 10 seconds / minute
- creates an excel with outcomes per person per measurement (day)

# TODO
# Add pain scores

'''

import logging
from datetime import date
import pandas as pd
from src.utils import *
from src.characteristics import characteristics
from src.prepare_data import prepare_data
from src.predict_data import predict_data
import os
logging.basicConfig(
    filename=f'logging/warnings_{date.today()}.log', level=logging.WARN)

# General project settings.
# More specific settings can be found in the config_file.
settings = {
    'VERBOSE': True,
    'VISUALISE': True,
    'PROCESS_ALL_DATA': False,  # set run_all to True to rerun all data
    'RAW_DATA_DIR': 'data/raw_data',
    'CONFIG_DIR': 'config'
}

# %%


def main():
    config = load_config(settings, config_name="config_file.yaml")
    files = os.listdir(settings['RAW_DATA_DIR'])

    # Load data
    for file in files:
        file_name = file.split('.')[0]

        # Load data, downsample and remove not worn
        data_df = prepare_data(file, config, settings)

        # Predict data based on a previously trained ML algorithm
        chunk_size = 6
        data = predict_data(data_df, config, settings,
                            file_name, file, chunk_size)

        results = characteristics(data, data_df, file)

        results_df = pd.DataFrame.from_dict(
            results, orient='index').transpose()
        for column in results_df.columns[4:]:
            results_df[column] = pd.to_numeric(
                results_df.loc[:, column]).round(3)

        results_df.to_excel('Results/test.xlsx', index=False)


if __name__ == '__main__':
    main()

# %%
