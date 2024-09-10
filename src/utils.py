import numpy as np
import yaml
from src.utils import *
import os


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
