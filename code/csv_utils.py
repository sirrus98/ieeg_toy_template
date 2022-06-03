import pandas as pd
from tqdm import tqdm
import os
import sys
from tqdm import tqdm
import numpy as np

sys.path.append('../../ieegpy/ieeg')
sys.path.append('/tools')
from get_iEEG_data import get_iEEG_data

path = "/".join(os.path.abspath(os.getcwd()).split('/')[:-1])


def csv_read(filename):
    df = pd.read_csv('data.csv')
    return df


def seizure_clip_read(metadata, username, password):
    seizure_data = []
    seizure_time = []
    tot = len(metadata)
    for index, s in tqdm(metadata.iterrows(), total=tot):
        iEEG_filename = s['iEEG Filename']
        start_time_usec = (s['Seizure EEC']) * 1e6
        stop_time_usec = (s['Seizure end']) * 1e6
        data, fs = get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec)
        t_sec = np.linspace(start_time_usec, stop_time_usec, num=data.shape[0]) / 1e6
        seizure_data.append(data)
        seizure_time.append(t_sec)
    return seizure_data, seizure_time


def nonseizure_clip_read(metadata, username, password, tm=600):
    seizure_data = []
    seizure_time = []
    tot = len(metadata)
    for index, s in tqdm(metadata.iterrows(), total=tot):
        iEEG_filename = s['iEEG Filename']
        start_time_usec = (s['Seizure EEC'] - tm) * 1e6
        stop_time_usec = (s['Seizure EEC'] - tm + 50) * 1e6
        data, fs = get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec)
        t_sec = np.linspace(start_time_usec, stop_time_usec, num=data.shape[0]) / 1e6
        seizure_data.append(data)
        seizure_time.append(t_sec)
    return seizure_data, seizure_time


def dump_pickle(file, path):
    if isinstance(file[0], pd.DataFrame):
        for i in tqdm(range(len(file))):
            file[i].to_pickle(path + str(i) + '.pkl')
    if isinstance(file[0], np.ndarray):
        for i in tqdm(range(len(file))):
            file[i].dump(path + str(i) + '.pkl')
