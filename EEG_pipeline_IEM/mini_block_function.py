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

# define function to find the minimal number of trials for each label
def min_trials_calculation(y):
    '''
    find the number of trials in each mini-block for all labels
    :param n_blocks: number of blocks for mini-blocks
    :return: the minimal number of trials for each label
    '''

    # find minimal trials for each label
    unique_labels = np.unique(y)
    num_all_labels = []
    for label in unique_labels:
        num = np.sum(y == label)
        num_all_labels.append(num)
    min_trials = min(num_all_labels)
    return min_trials



# define function to divide trials into mini blocks
def mini_block_assignment(min_trials, n_blocks, X, y):
    '''
    divide trials into mini blocks based on category labels
    and average data in each mini block
    :param min_trials: the minimal trial number for all labels
    :param n_blocks: the number of mini blocks
    :param X: data for single trials (n_sample, n_channel, n_time)
    :param y: labels for single trials (n_sample)
    :return: trials_index_mini, index for trials assignment
             trials_mini, data for mini blocks
             labels_mini, labels for mini blocks
             block_mini, block assignment for mini blocks
    '''

    n_trials = X.shape[0]
    n_electrodes = X.shape[1]
    n_times = X.shape[2]

    unique_labels = np.unique(y)

    # calculate the maximum # of trials for each label in each mini-block
    # such that the # of trials for each label can be equated within each block
    trials_mini_block = min_trials // n_blocks
    # calculate the # of trials for each label in all mini-blocks
    trials_each_label = int(trials_mini_block * n_blocks)
    # check selected trials should be no more than the initial trials
    assert trials_each_label * len(unique_labels) <= n_trials

    # randomly select trials for each mini-block for each label
    # record trials, trials_index, labels and blocks
    trials_index_mini = np.array([], dtype=int)
    trials_mini = np.empty([0, n_electrodes, n_times])
    labels_mini = np.array([], dtype=int)
    blocks_mini = np.array([], dtype=int)
    for label in unique_labels:
        # find trial-index for each label
        index = np.argwhere(y == label)
        # randomly choose trials for all mini blocks
        index_mini = np.random.choice(index[:, 0], size=trials_each_label, replace=False)
        # split trials into mini blocks
        index_mini_split = np.split(index_mini, n_blocks)

        for i_block in range(n_blocks):
            current_index = index_mini_split[i_block]
            current_trials = X[current_index, ]
            n_current_trials = current_index.shape[0]
            trials_index_mini = np.append(trials_index_mini, current_index, axis=0)
            trials_mini = np.append(trials_mini, current_trials, axis=0)
            labels_mini = np.append(labels_mini, np.repeat(label, n_current_trials))
            blocks_mini = np.append(blocks_mini, np.repeat(i_block, n_current_trials))

    return trials_index_mini, trials_mini, labels_mini, blocks_mini



def average_trials(trials, labels, blocks):
    '''
    average trials in each blocks based on their labels
    :param trials: EEG data, (n_trials, n_electrodes, n_times)
    :param labels: labels for EEG data, (n_times)
    :param blocks: block index for EEG data, (n_times)
    :return:
    '''

    n_trials = trials.shape[0]
    n_electrodes = trials.shape[1]
    n_times = trials.shape[2]

    unique_labels = np.unique(labels)
    unique_blocks = np.unique(blocks)

    trials_mean = np.empty([0, n_electrodes, n_times])
    labels_mean = np.array([], dtype=int)
    blocks_mean = np.array([], dtype=int)
    for i_block in unique_blocks:
        block_index = np.squeeze(np.argwhere(blocks == i_block))
        block_trials = np.squeeze(trials[block_index])
        block_labels = np.squeeze(labels[block_index])
        for i_label in unique_labels:
            label_index = np.squeeze(np.argwhere(block_labels == i_label))
            label_trials = np.squeeze(block_trials[label_index])
            mean = np.mean(label_trials, axis=0).reshape(1, n_electrodes, n_times)
            trials_mean = np.append(trials_mean, mean, axis=0)
            labels_mean = np.append(labels_mean, i_label)
            blocks_mean = np.append(blocks_mean, i_block)

    return trials_mean, labels_mean, blocks_mean