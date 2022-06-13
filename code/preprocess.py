import numpy as np
import copy


def normalization_fit_stack(dg_data):
    """
    :param dg_data: [nd.array(n,5),nd.array(n,5),nd.array(n,5)], digital glove datas for all subjects
    :return: list of tuples, mins and maxs for the dg_data
    """
    dg_min, dg_max = np.min(dg_data), np.max(dg_data)
    return dg_min, dg_max


def normalize_stack(dg_data, dmin, dmax):
    """
    :param dg_data: [nd.array(n,5),nd.array(n,5),nd.array(n,5)], digital glove datas for all subjects
    :param mm: list of tuples, mins and maxs for the dg_data
    :return: [nd.array(n,5),nd.array(n,5),nd.array(n,5)], normalized data
    """
    dg_copy = copy.deepcopy(dg_data)
    dg_copy = (dg_copy - dmin) / (dmax - dmin)
    return dg_copy


def normalize_test(dg_data, mm, ind):
    """
    :param dg_data: [nd.array(n,5),nd.array(n,5),nd.array(n,5)], digital glove datas for all subjects
    :param mm: list of tuples, mins and maxs for the dg_data
    :return: [nd.array(n,5),nd.array(n,5),nd.array(n,5)], normalized data
    """
    dg_copy = copy.deepcopy(dg_data)
    dg_copy = (dg_copy - mm[ind][0][np.newaxis, :]) / mm[ind][1][np.newaxis, :]
    return dg_copy


def normalization_recover(dg_data_norm, mm):
    """
    :param dg_data_norm: [nd.array(n,5),nd.array(n,5),nd.array(n,5)], normalized digital glove datas for all subjects
    :param mm: list of tuples, mins and maxs for the dg_data
    :return: [nd.array(n,5),nd.array(n,5),nd.array(n,5)], digital glove data in original scale
    """
    dg_copy = copy.deepcopy(dg_data_norm)
    for i in range(len(dg_data_norm)):
        dg_copy[i] = dg_copy[i] * mm[i][1] + mm[i][0]
    return dg_copy
