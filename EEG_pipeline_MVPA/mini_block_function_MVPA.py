# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : mini_block_transformation.py
# Time       ：2021/4/8 16:17
# Author     ：ZoeDuan
# version    ：python 3.7
# Description：    divide trials into mini blocks based on category labels
                   and average data in each mini block
"""

import numpy as np
from numpy import random

# define function to find the minimal trial number for all labels
def min_trials_calculation(y):
    '''
    find the minimal trial number for all labels
    :param y: labels for all trials
    :return: the minimal trial number for all labels
    '''
    unique_labels = np.unique(y)
    num_all_labels = []
    for label in unique_labels:
        num = np.sum(y == label)
        num_all_labels.append(num)
    min_trials = min(num_all_labels)
    return min_trials

# define function to divide trials into mini blocks
def mini_block_transform(min_trials, trials_mini_block, X, y):
    '''
    divide trials into mini blocks based on category labels
    and average data in each mini block
    :param min_trials: the minimal trial number for all labels
    :param trials_mini_block: the number of trials in each mini block
    :param X: data for single trials (n_sample, n_channel, n_time)
    :param y: labels for single trials (n_sample)
    :return: X_mini, data for mini blocks
             y_mini, labels for mini blocks
    '''

    n_trial = X.shape[0]
    n_channel = X.shape[1]
    n_times = X.shape[2]

    unique_labels = np.unique(y)
    
    X_mini = np.empty([0, n_channel, n_times])
    y_mini = np.array([])

    # calculate the number of mini-block for each label
    num_mini_block = min_trials // trials_mini_block
    # calculate the number of trials for each label
    trials_each_label = int(trials_mini_block * num_mini_block)
    # check selected trials should be no more than the initial trials
    assert trials_each_label*len(unique_labels) <= n_trial

    # randomly select trials for each mini-block for each label
    # and average trials in each mini-block for each label
    for label in unique_labels:
        # find trial-index for each label
        index = np.argwhere(y == label)
        # randomly choose trials for all mini blocks
        index_mini = np.random.choice(index[:, 0], size=trials_each_label, replace=False)
        trials_mini = X[index_mini, :, :]
        # split trials into mini blocks
        trials_mini_split = np.split(trials_mini, num_mini_block)
        # average trials in each mini block
        trials_mini_mean = np.empty([0, n_channel, n_times])
        for trials in trials_mini_split:
            mean = np.mean(trials, axis=0).reshape(1, n_channel, n_times)
            trials_mini_mean = np.append(trials_mini_mean, mean, axis=0)

        labels_mini = np.repeat(label, trials_mini_mean.shape[0])

        X_mini = np.append(X_mini, trials_mini_mean, axis=0)
        y_mini = np.append(y_mini, labels_mini)
    return X_mini, y_mini