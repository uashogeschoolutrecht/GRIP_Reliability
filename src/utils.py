import yaml
import logging
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import pandas as pd

def load_config(settings, config_name):
    with open(os.path.join(settings['CONFIG_DIR'], config_name)) as params_file:
        params = yaml.safe_load(params_file)
    return params

def detect_outliers(data, raw_data_columns, settings, z_score = 4):
    outliers = np.array([])
    for catagory in data['label_converted'].unique():
        # Select all data per category
        if settings['VERBOSE']:
            print(f'catagory: {catagory}')
        cur_outliers = len(outliers)
        tmp_data = data.loc[data['label_converted'] == catagory]    
        for column in raw_data_columns:
            z_scores = np.abs(convert_to_z_score(tmp_data.groupby('unique_id')[column].mean()))
            tmp_outliers = z_scores.loc[z_scores > z_score]
            outliers = np.concatenate((outliers, tmp_outliers.index.values[:]))
            z_scores = np.abs(convert_to_z_score(tmp_data.groupby('unique_id')[column].std()))
            tmp_outliers = z_scores.loc[z_scores > z_score]
            outliers = np.concatenate((outliers, tmp_outliers.index.values[:]))
        outliers = np.unique(outliers)   
        if settings['VERBOSE']:      
            print(f'Outliers {len(outliers) - cur_outliers}')
            
    return outliers
    
def convert_to_z_score(arr):
    # Calculate z_scores
    return (arr - arr.mean()) / arr.std()

def calculate_metrics(validate_labels, validate_predictions):
    overall_accuracy = round(accuracy_score(validate_labels, validate_predictions),3)
    weighted_recall = round(recall_score(
        validate_labels, validate_predictions, average='macro'),3)
    weighted_precision = round(precision_score(
        validate_labels, validate_predictions, average='macro'),3)
    weighted_f1_score = round(f1_score(
        validate_labels, validate_predictions, average='macro'),3)
    confusion_mat = confusion_matrix(validate_labels, validate_predictions)
    normalized_confusion_mat = np.round(confusion_mat.astype(
        'float') / confusion_mat.sum(axis=1)[:, np.newaxis], 2)
    return {'overall_accuracy':overall_accuracy,
            'weighted_recall':weighted_recall,
            'weighted_precision':weighted_precision,
            'weighted_f1_score':weighted_f1_score,
            'confusion_mat':confusion_mat,
            'normalized_confusion_mat':normalized_confusion_mat}


def print_visualise_results(results,  model_name, group_name, settings):
    print(f' \n Model name {model_name} \n')
    avg_acc = round(results['overall_accuracy'].mean() * 100, 1)
    std_acc = round(results['overall_accuracy'].std() * 100, 1)
    min_acc = round(results['overall_accuracy'].min() * 100, 1)
    max_acc = round(results['overall_accuracy'].max() * 100, 1)
    avg_f1 = round(results['weighted_f1_score'].mean() * 100, 1)
    std_f1 = round(results['weighted_f1_score'].std() * 100, 1)
    min_f1 = round(results['weighted_f1_score'].min() * 100, 1)
    max_f1 = round(results['weighted_f1_score'].max() * 100, 1)
    avg_recall = round(results['weighted_recall'].mean() * 100, 1)
    std_recall = round(results['weighted_recall'].std() * 100, 1)
    min_recall = round(results['weighted_recall'].min() * 100, 1)
    max_recall = round(results['weighted_recall'].max() * 100, 1)
    avg_precision = round(results['weighted_precision'].mean() * 100, 1)
    std_precision = round(results['weighted_precision'].std() * 100, 1)
    min_precision = round(results['weighted_precision'].min() * 100, 1)
    max_precision = round(results['weighted_precision'].max() * 100, 1)
    if settings['VERBOSE']:
        print(f'Accuracy {avg_acc} ({std_acc}), [{min_acc},{max_acc}]')
        print(f'F1 {avg_f1} ({std_f1}), [{min_f1},{max_f1}]')
        print(
            f'Recall {avg_recall} ({std_recall}), [{min_recall},{max_recall}]')
        print(
            f'Precision {avg_precision} ({std_precision}), [{min_precision},{max_precision}]')
    
    results = results.dropna()
    
    results_descr = results.describe()
    
    normalized_confusion_mat = []
    for i in range(4):
        tmp = []
        for j in range(4):
            tmp.append(results_descr[f'norm_conf_mat_row_{i}_{j}']['mean'] * 100)
        normalized_confusion_mat.append(tmp)
        
    # Create a heatmap for the confusion matrix
    sns.set(font_scale=1.8)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(normalized_confusion_mat, annot=True,  cmap="Blues", cbar=False, square=True,
                xticklabels=["Sedentary", "Light intensity", "Moderate intensity",  "Vigorous intensity"],
                yticklabels=["Sedentary","Light intensity", "Moderate intensity",  "Vigorous intensity"], ax=ax)
    ax.set_xlabel("Predicted Categories")
    ax.set_ylabel("Observed Categories")

    fig.tight_layout()
    fig.savefig(f'Figures/results/Confusion_matrix_{model_name}_{group_name}.png', dpi=400)

    
def characteristics(data):
    print(f"Amount of unique individuals: {len(data['subject'].unique())}")
    print(f"Distribution activities \n {data['label_converted'].value_counts() / 125}")
    print(f"Distribution labels \n {data['label'].value_counts() / 125}")

def hotencode(data, column):
    # hot encode of the outcomes
    dummies = pd.get_dummies(data[column], dtype=int)
    data = pd.concat([data, dummies], axis=1)
    data.reset_index(inplace=True)
    return data

def prepare_data(data):
    # prepare data for ML
    data_input_array = {part: group[['acc_x', 'acc_y', 'acc_z']] for part, group in data.groupby('unique_id')}
    data_output_array = {part: group[['sedentary','light_intensity','moderate_intensity','vigorous_intensity']] for part, group in data.groupby('unique_id')}
    return data_input_array, data_output_array

def data_to_np_input(keys, data):
    length, width = data[list(data.keys())[0]].shape
    data = dict(filter(lambda item: item[0] in keys, data.items()))
    data_np = np.empty((len(data),length,width ))
    counter = 0
    for _, item in data.items():
        data_np[counter] = item.values
        counter += 1
    return data_np

def data_to_np_output(keys, data):
    _, width = data[list(data.keys())[0]].shape
    data = dict(filter(lambda item: item[0] in keys, data.items()))
    data_np = np.empty((len(data),width ))
    counter = 0
    for _, item in data.items():
        data_np[counter] = item.values[0]
        counter += 1
    return data_np    
