import numpy as np
from src.utils import *
import pandas as pd
import numpy as np
from src.utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def mean_value(chunk):
    return np.round(np.mean(chunk) + 0.01)


def predict_data(data_df, config, settings, file_name, file, chunk_size):
    data_df = data_df.reset_index()
    data_df['group'] = np.nan
    samples_per_tensec = int(config['sample_size'] * config['frequency'])
    samples = int(np.floor(len(data_df) / samples_per_tensec))
    for num in range(samples):
        start = samples_per_tensec * num
        end = samples_per_tensec * num + samples_per_tensec - 1
        data_df.loc[start:end, 'group'] = int(num)
    data_df = data_df.dropna()

    # Drop values that are
    data_input_array = {part: group[['acc_x', 'acc_y', 'acc_z']]
                        for part, group in data_df.groupby('group')}
    length, width = data_input_array[list(
        data_input_array.keys())[0]].shape
    data_np = np.empty((len(data_input_array), length, width))
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
    predicted_values_df = pd.DataFrame(
        predicted_values, columns=['activities [10 sec]'])
    predicted_values_df = predicted_values_df.merge(data_df.drop_duplicates(subset='group')[['time_2hours', 'group']],
                                                    left_index=True,  # Use index from predicted_values_df
                                                    right_on='group').reset_index(drop=True)
    predicted_values_df.to_csv(f'data/ten_sec/{file_name}_10sec.csv')

    # values per minute
    # Define the chunk size
    chunks = [predicted_values[i:i + chunk_size]
              for i in range(0, len(predicted_values), chunk_size)]

    most_common_per_chunk = [mean_value(chunk) for chunk in chunks[:-1]]
    most_common_per_chunk_df = pd.DataFrame(
        most_common_per_chunk, columns=['activities [1min]'])
    most_common_per_chunk_df = most_common_per_chunk_df.merge(predicted_values_df[::6].reset_index()['time_2hours'],
                                                              left_index=True,  # Merge on the index of most_common_per_chunk
                                                              right_index=True)
    most_common_per_chunk_df.to_csv(f'data/one_min/{file_name}_1min.csv')

    # visualise activities per 10 seconds
    # if settings['VISUALISE']:
    #     x_axis = (np.arange(len(predicted_values)) * 10) / 60 / 60
    #     fig, ax = plt.subplots()
    #     ax.plot(x_axis, predicted_values)
    #     ax.set_title(f'Activities for file: '
    #                  f'{file} per 10 seconds')
    #     ax.set_ylabel('Activity')
    #     ax.set_xlabel('Time [hours]')
    #     fig.savefig(f'Figures/{file}_10sec.png')

    # visualise activities per chunk size
    if settings['VISUALISE']:
        x_axis = np.arange(len(most_common_per_chunk))
        colors = ['#270452', '#A82C2C', '#D49057', '#F1EDA6']
        unique_categories = np.unique(most_common_per_chunk)
        color_map = {category: colors[i] for i, category in enumerate(unique_categories)}
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, category in enumerate(most_common_per_chunk):
            ax.vlines(x_axis[i], ymin=0, ymax=1, color=color_map[category], linewidth=5)
        ax.set_title(f'Activities for file: {file}')
        ax.set_ylabel('Activity')
        ax.set_xlabel('Time [minutes]')
        ax.set_yticks([])  # Remove y-axis ticks since this is categorical data
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        category_labels = ['Sedentair', 'Licht intensief', 'Gemiddeld intensief', 'Hoog intensief']
        legend_elements = [plt.Line2D([0], [0], color=color_map[category], lw=4, label=category_labels[i]) for i, category in enumerate(unique_categories)]
        ax.legend(handles=legend_elements, title='Categoriën', loc='upper right')
        plt.show()
        fig.savefig(f'Figures/{file_name}_1min.png')

    return most_common_per_chunk_df
