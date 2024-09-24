import numpy as np
import yaml
from src.utils import *
import os
import logging
from datetime import date


def initialise_project():
    check_or_create_folder('Figures')
    check_or_create_folder('data')
    check_or_create_folder('data/one_min')
    check_or_create_folder('data/raw_data')
    check_or_create_folder('data/ten_sec')
    check_or_create_folder('logging')
    check_or_create_folder('Results')
    logging.basicConfig(
        filename=f'logging/warnings_{date.today()}.log', level=logging.WARN)


def check_or_create_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it does not exist, create the folder
        os.makedirs(folder_path)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def probability(vector):
    p_vector = vector / sum(vector)
    return p_vector


def entropy(p_vector):
    p_0 = p_vector[0]*np.log(p_vector[0])
    for p in p_vector[1:]:
        p_0 = p_0 + p*np.log(p)
    return -p_0


def load_config(settings, config_name):
    with open(os.path.join(settings['CONFIG_DIR'], config_name)) as params_file:
        params = yaml.safe_load(params_file)
    return params
