import pandas as pd
import numpy as np
from src.utils import *


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


def prepare_data(file, config, settings):
    data_df = pd.read_csv(
        f"{settings['RAW_DATA_DIR']}/{file}", header=None, skiprows=10)
    data_df = data_df.drop(data_df.iloc[:, 4:8], axis=1)
    data_df = data_df.iloc[:, 1:]
    data_df.columns = ['acc_x', 'acc_y', 'acc_z']
    data_df = data_df.dropna()

    timestamps = pd.read_csv(
        f"{settings['RAW_DATA_DIR']}/{file}", skiprows=8, on_bad_lines='skip')
    timestamps = timestamps.dropna(subset='Unnamed: 7')
    start_time = timestamps.iloc[0, -1]
    end_time = timestamps.iloc[-1, -1]
    diff = end_time - start_time
    # Time in seconds
    diff /= 10000
    sampling_freq = len(data_df) / diff

    if sampling_freq > config['frequency']:
        data_df = downsample_to_frequency(
            data_df, sampling_freq, config['frequency'])

    # Detect parts in which the sensor was not worn
    data_df['Worn_sensor'] = 1
    samples_per_minute = int(config['frequency'] * 60)
    samples = int(np.floor(len(data_df) / samples_per_minute))
    for num in range(samples):
        start = samples_per_minute * num
        end = samples_per_minute * num + samples_per_minute
        sample = data_df.iloc[start:end, :]
        std = (np.std(sample['acc_x']) +
               np.std(sample['acc_y']) + np.std(sample['acc_z'])) / 3
        if std < 10 ** (-6):
            data_df.iloc[start:end, 'Worn_sensor'] = 0

    # Drop data that has more than a minute of inactivity
    data_df = data_df.loc[data_df['Worn_sensor'] == 1]

    # Drop the first and last 30 seconds
    samples_per_halfmin = int(config['frequency'] * 30)
    data_df = data_df.iloc[samples_per_halfmin:len(
        data_df)-samples_per_halfmin, :]
    return data_df
