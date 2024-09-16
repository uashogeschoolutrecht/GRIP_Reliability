
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
from src.characteristics import characteristics, characteristics_pain
from src.prepare_data import prepare_data
from src.predict_data import predict_data
import os
from datetime import datetime


logging.basicConfig(
    filename=f'logging/warnings_{date.today()}.log', level=logging.WARN)

# General project settings.
# More specific settings can be found in the config_file.
settings = {
    'VERBOSE': True,
    'VISUALISE': False,
    'PROCESS_ALL_DATA': False,  # set run_all to True to rerun all data
    'RAW_DATA_DIR': 'data/raw_data',
    'CONFIG_DIR': 'config'
}

# %%


def main():
    config = load_config(settings, config_name="config_file.yaml")
    subjects = os.listdir(settings['RAW_DATA_DIR'])
    final_results = pd.DataFrame()

    # Load data
    for subject in subjects:

        try:
            if subject.endswith('.DS_Store'):
                continue
            painscores = pd.read_csv(
                f'data/raw_data/{subject}/pain_score/{subject}_pain_score.csv', index_col=0)
            days = os.listdir(f"{settings['RAW_DATA_DIR']}/{subject}")
            days.remove('pain_score')

            for day in days:
                if day.endswith('.csv'):
                    continue
                try:
                    results = {}

                    file_path = os.listdir(
                        f"{settings['RAW_DATA_DIR']}/{subject}/{day}")[0]
                    file = f"{subject}/{day}/{file_path}"
                    file_name = f"{subject}_{day}"

                    # Load data, downsample and remove not worn
                    data_df, not_worn_samples = prepare_data(
                        file, config, settings)

                    # Predict data based on a previously trained ML algorithm
                    chunk_size = 6
                    data = predict_data(data_df, config, settings,
                                        file_name, file, chunk_size)

                    results['subject'] = subject
                    results['day'] = day
                    results['Samples'] = len(data_df)
                    results['not_worn_samples'] = not_worn_samples
                    results['Name'] = file
                    results = characteristics(results, data)

                    # Parse the date
                    try:
                        results = characteristics_pain(
                            painscores, results, day)
                    except Exception as e:
                        logging.error(f'pain: {subject} {day} {e}')
                    results_df = pd.DataFrame.from_dict(
                        results, orient='index').transpose()
                    for column in results_df.columns[4:]:
                        results_df[column] = pd.to_numeric(
                            results_df.loc[:, column]).round(3)
                    final_results = pd.concat((final_results, results_df))
                except Exception as e:
                    logging.error(f'day: {subject} {day} {e}')
        except Exception as e:
            logging.error(f'subject: {subject} {day} {e}')

    final_results.to_excel('Results/test.xlsx', index=False)
    final_results.loc[:, 'average_activity_level':].corr().to_excel(
        'Results/correlations.xlsx')


if __name__ == '__main__':
    main()

# %%
