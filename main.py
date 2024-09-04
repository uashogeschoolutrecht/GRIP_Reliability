
#%%
'''
Main script for the GRIP HAP project.
'''
# TODO
# Detect groups where non worn values are in and drop these

import logging
from datetime import date
import pandas as pd
import numpy as np
import statistics
import yaml
from src.utils import load_config
import os
import tensorflow as tf
import matplotlib.pyplot as plt
logging.basicConfig(filename=f'logging/warnings_{date.today()}.log', level=logging.WARN)

# General project settings.
# More specific settings can be found in the config_file.
settings = {
    'VERBOSE' : True,
    'VISUALISE' : True,
    'PROCESS_ALL_DATA':False, #set run_all to True to rerun all data
    'RAW_DATA_DIR': 'data/raw_data',
    'CONFIG_DIR': 'config'
    }


#%%
def characteristics(results, data):
    # Info all data
    results['average_activity_level'] = np.mean(data)

    # Info about changes
    differences = np.diff(data)
    results['std_change'] = np.std(differences)
    indx_changes = np.where(differences != 0)[0] + 1
    results['per_change'] = len(indx_changes) / len(data) * 100 
    
    # per epoch (can be longer than ML-output length)
    epochs =   [] 
    epochs.append(data[0:indx_changes[0]])
    for num, indx in enumerate(indx_changes[:-1]):
        epochs.append(data[indx:indx_changes[num+1]])
    epochs.append(data[indx_changes[-1]:])
    
    # Set epoch per activity
    sedentairy =[]
    light =[]
    moderate =[]
    vigorous = []
    for epoch in epochs:
        if epoch[0] == 0:
            sedentairy.append(epoch)
        elif epoch[0] == 1:
            light.append(epoch)
        elif epoch[0] == 2:
            moderate.append(epoch)
        else:
            vigorous.append(epoch)
    
    # Get results per activity
    unique, counts = np.unique(data, return_counts=True)
    for i in zip(unique, counts):
        if i[0] == 0:
            activity = 'sedentairy'
        elif i[0] == 1:
            activity = 'light'
        elif i[0] == 2:
            activity = 'moderate'
        else:
            activity = 'vigorous'
        results[f'{activity}_count'] = i[1]
        results[f'{activity}_perc'] = i[1] / len(data) * 100
        
        # Percentage epochs
        activity_epochs = eval(activity)
        results[f'epochs_{activity}_perc'] = len(activity_epochs) / len(epochs) * 100
        
        # Median and average epoch length
        lengths = [len(sublist) for sublist in activity_epochs]
        results[f'epochs_{activity}_median_length'] = np.median(lengths)
        results[f'epochs_{activity}_average_length'] = np.average(lengths)
        results[f'epochs_{activity}_max_length'] = np.max(lengths)
    return results
    

def main():
    config = load_config(settings, config_name="config_file.yaml")
    files = os.listdir(settings['RAW_DATA_DIR'])
    
    # Load data
    for file in files:
        file_name = file.split('.')[0]
        data_df = pd.read_csv(f"{settings['RAW_DATA_DIR']}/{file}", header=None, skiprows=10)
        data_df = data_df.drop(data_df.iloc[:, 4:8], axis=1)  
        data_df = data_df.iloc[:, 1:]
        data_df.columns = ['acc_x', 'acc_y', 'acc_z']
        data_df = data_df.dropna() 
        
        timestamps = pd.read_csv(f"{settings['RAW_DATA_DIR']}/{file}", skiprows = 8, on_bad_lines = 'skip')
        timestamps = timestamps.dropna(subset='Unnamed: 7')
        start_time = timestamps.iloc[0,-1]
        end_time = timestamps.iloc[-1,-1]
        diff = end_time - start_time
        # Time in seconds
        diff /= 10000
        sampling_freq = len(data_df) / diff
        
        def downsample_to_frequency(df, current_freq, target_freq):
            # Calculate the ratio of current frequency to target frequency
            ratio = current_freq / target_freq
            
            # Calculate the interval to downsample
            interval = 1 / (ratio - 1)
            
            # Initialize an empty list to store the indices to keep
            indices_to_keep = []
            
            # Iterate over the DataFrame to select appropriate rows
            skip_counter = 0
            for i in range(len(df)):
                if skip_counter >= interval:
                    skip_counter = 0  # Reset skip counter
                else:
                    indices_to_keep.append(i)
                    skip_counter += 1
            
            # Select the rows based on the calculated indices
            df_downsampled = df.iloc[indices_to_keep].reset_index(drop=True)
            
            return df_downsampled
        
        if sampling_freq > config['frequency']:
            data_df = downsample_to_frequency(data_df, sampling_freq, config['frequency'])

        # Detect parts in which the sensor was not worn
        # Drop data that has more than a minute of inactivity
        data_df['Worn_sensor'] = 1
        samples_per_minute = int(config['frequency'] * 60 )
        samples = int(np.floor(len(data_df) / samples_per_minute))
        for num in range(samples):
            start = samples_per_minute * num
            end = samples_per_minute * num + samples_per_minute
            sample = data_df.iloc[start:end ,:]
            std = (np.std(sample['acc_x']) + np.std(sample['acc_y']) + np.std(sample['acc_z']) ) / 3
            if std < 10 ** (-6):
                data_df.iloc[start:end,'Worn_sensor'] = 0
        
        # Drop the first and last 30 seconds
        samples_per_halfmin = int(config['frequency'] * 30 )
        data_df = data_df.iloc[samples_per_halfmin:len(data_df)-samples_per_halfmin,:]
        
        # Predict activities per 10 seconds
        # Create a group per activity sample
        data_df = data_df.reset_index()
        data_df['group'] = np.nan
        samples_per_tensec = int(config['sample_size'] * config['frequency'])
        samples = int(np.floor(len(data_df) / samples_per_tensec))
        for num in range(samples):
            start = samples_per_tensec * num
            end = samples_per_tensec * num + samples_per_tensec - 1
            data_df.loc[start:end,'group'] = int(num)
        data_df = data_df.dropna()


        # Only include groups where the sensor is worn
        
        
        # Drop values that are
        data_input_array = {part: group[['acc_x', 'acc_y', 'acc_z']] for part, group in data_df.groupby('group')}
        length, width = data_input_array[list(data_input_array.keys())[0]].shape
        data_np = np.empty((len(data_input_array),length,width ))
        counter = 0
        for _, item in data_input_array.items():
            data_np[counter] = item.values
            counter += 1
        
        # Load model
        model = tf.keras.models.load_model('models/CNN_model.keras')

        # predict values
        # Drop 
        predicted_values = model.predict(data_np)
        predicted_values = np.argmax(predicted_values, axis=1)
        
        predicted_values_df = pd.DataFrame(predicted_values, columns = ['activities [10 sec]'])
        predicted_values_df.to_csv(f'data/ten_sec/{file_name}_10sec.csv')
        
        # visualise activities per 10 seconds
        x_axis = (np.arange(len(predicted_values)) * 10) / 60
        fig, ax = plt.subplots()

        ax.plot(x_axis, predicted_values)
        ax.set_title(f'Activities during 4 hours for file: {file} per 10 seconds')
        ax.set_ylabel('Activity')
        ax.set_xlabel('Time [minutes]')
        
        # values per minute
        # Define the chunk size
        chunk_size = 6 # 6 = 1 minute
        chunks = [predicted_values[i:i + chunk_size] for i in range(0, len(predicted_values), chunk_size)]
        def mean_value(chunk):
            return np.round(np.mean(chunk) + 0.01)
        most_common_per_chunk = [mean_value(chunk) for chunk in chunks[:-1]]
        most_common_per_chunk_df = pd.DataFrame(most_common_per_chunk, columns = ['activities [1min]'])
        most_common_per_chunk_df.to_csv(f'data/one_min/{file_name}_1min.csv')
           
        x_axis = np.arange(len(most_common_per_chunk))
        fig, ax = plt.subplots()

        ax.plot(x_axis, most_common_per_chunk)
        ax.set_title(f'Activities during 4 hours for file: {file}')
        ax.set_ylabel('Activity')
        ax.set_xlabel('Time [minutes]')
        
        data = most_common_per_chunk
        
        results = {}
        results['Name'] = file
        results['Samples'] = len(data_df)
        results['Samples_worn'] = len(data_df.loc[data_df['Worn_sensor'] == 1])
        results['Epochs of 1 minute'] = len(most_common_per_chunk)
        results = characteristics(results, data)
        results_df = pd.DataFrame.from_dict(results, orient='index').transpose()
        results_df.to_excel('Results/test.xlsx', index=False)
        
if __name__ == '__main__':
    main()
 
#%%