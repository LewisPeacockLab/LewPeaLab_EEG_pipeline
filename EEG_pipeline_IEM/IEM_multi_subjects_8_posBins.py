# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : IEM_multi_subjects.py
# Time       ：21/04/2021 10:07 PM
# Author     ：ZoeDuan
# version    ：python 3.7
# Description：      The pipeline of IEM decoding 8-class alpha-band power from Foster et al(2017)
#                   Data: Experiment 2a, 14 Subjects, preprocessed data (epochs)
"""

import os
import mne
import numpy as np
import math
import random
import time
import pandas as pd
import h5py
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns

from mini_block_function import *



# detect how many subjects you have
dir = os.getcwd()
data_path = dir + '/Preprocessed_data'
files = os.listdir(data_path)
sub_num = len(files)    # the number of subjects in total
print('You have %d participants in total' % sub_num)

# read all participants' data into all_data
all_data = {}
subIDs = np.array([])
for file in files:
    if not os.path.isdir(file):
        # find subID from file name
        afterInd = file.find('_')
        subID = file[0:afterInd]
        subIDs = np.append(subIDs, subID)
        data = mne.read_epochs(data_path + '/' + file, preload=False)
        all_data[subID] = data

# create file folder to save prepared data for decoding
prepared_data_dir = dir + '/prepared_data/'
exist = os.path.exists(prepared_data_dir)
if not exist:
    os.makedirs(prepared_data_dir)

# create file folder to save classification results
classification_results_dir = dir + '/classification_results_8_quadrant/'
exist = os.path.exists(classification_results_dir)
if not exist:
    os.makedirs(classification_results_dir)


# define variables
# define band-pass frequency
l_freq, h_freq = 8, 12
# define decoding timeWindow -300ms ~ 1250ms
timeWindow = [-0.3, 1.25]
# define resample numbers for each time point
n_resample = 1

# create time string
time_string = str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

# prepare data for each subject
X_all = {}
y_all = {}
time_points_resample_all = {}
for subID in subIDs:
    # read current data
    epochs = all_data[subID]
    # load current data
    epochs.load_data()
    # pick eeg channels only
    epochs.pick_types(eeg=True)
    # get sample frequency
    sfreq = epochs.info['sfreq']

    # get block_index and labels
    events_id = epochs.events[:, 2]
    blocks = events_id // 10
    labels = events_id % 10

    # get alpha-band power
    epochs_filt = epochs.copy().filter(l_freq=l_freq, h_freq=h_freq, phase='zero-double')
    epochs_filt.apply_hilbert(picks=['eeg'], envelope=True, n_fft='auto')
    power = epochs_filt.get_data()

    # extract sub-data within timeWindow
    all_time_points = epochs.times
    left_timepoint_index = int((timeWindow[0] - all_time_points[0])*sfreq)
    right_timepoint_index = int(len(all_time_points) - (all_time_points[-1] - timeWindow[1])*sfreq)
    sub_time_points = all_time_points[left_timepoint_index:right_timepoint_index]
    assert sub_time_points[0] == timeWindow[0]
    assert sub_time_points[-1] == timeWindow[1]
    # extract sub-power within timeWindow
    sub_power = power[:, :, left_timepoint_index:right_timepoint_index]

    n_trials = sub_power.shape[0]
    n_electrodes = sub_power.shape[1]
    n_times = sub_power.shape[2]

    # delete epochs to save memory
    del epochs, epochs_filt

    # down-sampling data based on n_resample
    n_times_resample = round(n_times / n_resample)
    time_points_resample = np.linspace(timeWindow[0], timeWindow[1], n_times_resample)
    # down-sampling data by combining every n_resample data
    sub_power_resample = np.zeros([n_trials, n_electrodes, n_times_resample], dtype=np.float)
    for t in range(n_times_resample):
        sub_power_resample[:, :, t] = np.average(sub_power[:, :, t * n_resample: (t + 1) * n_resample], axis=-1)
    assert sub_power_resample.shape == (n_trials, n_electrodes, n_times_resample)

    # define data for decoding
    X = np.array(sub_power_resample)
    y = np.array(labels)

    del sub_power_resample, labels

    # save each participants' alpha_band data into disk
    f = h5py.File(prepared_data_dir + '/' + subID + '_prepared_data.txt', 'w')
    f['X'] = X
    f['y'] = y
    f['time_points_resample'] = time_points_resample
    f.close()
    # read each participants' alpha_band data
    # f = h5py.File(prepared_data_dir + '/' + subID + '_prepared_data.txt', 'r')
    # X = f['X'][()]
    # y = f['y'][()]
    # time_points_resample = f['time_points_resample'][()]

    X_all[subID] = X
    y_all[subID] = y
    time_points_resample_all[subID] = time_points_resample



# define number of mini-blocks
n_blocks = 3
# define iteration times
n_iteration = 10


# Specify basis set for IEM
n_channels = 8  # of spatial channels for IEM
n_posBins = 8      # of position bins/categories
sinPower = 7
x = np.linspace(0, 2*np.pi-2*np.pi/n_posBins, n_posBins)
channel_centers = np.linspace(0, 2*np.pi-2*np.pi/n_channels, n_channels)
channel_centers = np.rad2deg(channel_centers)
# define hypothetical channel responses
resp = np.sin(0.5*x)**sinPower
# shift the initial basis function
resp = np.roll(resp, 3)
# generate circularly shifted basis functions
basis_set = np.zeros((n_channels, n_posBins))
for c in range(n_channels):
    basis_set[c, :] = np.roll(resp, c+1)


# plot basis function
fig, ax = plt.subplots()
for i in range(n_channels):
    sns.lineplot(x=channel_centers, y=basis_set[i])
ax.set_xticks(channel_centers)
ax.set_title('Spatial channels')
ax.set_xlabel('Angular position')
ax.set_ylabel('Response')

plt.savefig('Spatial channels.png')



# ----------------------- IEM decoding for all subjects ----------------------------#
# pre-allocate space for mini-block assignment for all subjects
trials_index_mini_all = {}
labels_mini_all = {}
blocks_mini_all = {}

# pre-allocate space for predicted CTF for all subjects
CTF_pred_all = np.zeros([sub_num, n_times_resample, n_channels])
# pre-allocate space for CTF slopes for all subjects
CTF_slope_all = np.zeros([sub_num, n_times_resample])

sub_index = 0
for subID in subIDs:
    # read each subject's data
    X = X_all[subID]
    y = y_all[subID]
    # find information for current data
    n_trials = X.shape[0]
    n_electrodes = X.shape[1]
    n_times_resample = X.shape[2]
    unique_labels = np.unique(y)

    # divide data into 3 blocks and average data for each label
    # find the minimal trial number among labels
    min_trials = min_trials_calculation(y)
    # calculate the maximum trial number for each label in each mini-block
    # such that the trial number for each label can be equated within each block
    trials_mini_block = min_trials // n_blocks
    # calculate the final trial number for all labels in all blocks
    n_trials_final = int(trials_mini_block*len(unique_labels)*n_blocks)
    assert n_trials_final <= n_trials

    # pre-allocate space for mini-block assignment for each iteration
    trials_index_mini_iter = np.zeros([n_iteration, n_trials_final])
    labels_mini_iter = np.zeros([n_iteration, n_trials_final])
    blocks_mini_iter = np.zeros([n_iteration, n_trials_final])
    # pre-allocate space for predicted CTF for test data
    CTF_pred = np.zeros([n_blocks, n_iteration, n_times_resample, n_channels])

    # Loop through each iteration
    for i_iter in trange(n_iteration):
        # find the minimal trials for
        # divide single trials into mini-blocks
        trials_index_mini, trials_mini, labels_mini, blocks_mini = mini_block_assignment(min_trials, n_blocks, X, y)
        # record mini-block assignment for each iteration
        trials_index_mini_iter[i_iter, ] = trials_index_mini
        labels_mini_iter[i_iter, ] = labels_mini
        blocks_mini_iter[i_iter, ] = blocks_mini

        # average trials in each mini-block for all labels
        trials_mini_mean, labels_mini_mean, blocks_mini_mean = average_trials(trials_mini, labels_mini, blocks_mini)

        # predict channel response for averaged data
        resp_mini_mean = np.zeros((len(labels_mini_mean), n_channels))
        for ind in range(len(labels_mini_mean)):
            label = labels_mini_mean[ind]
            resp_mini_mean[ind, :] = basis_set[int(label) - 1, :]

        # loop through blocks, holding each out as the test set
        for i_block in range(n_blocks):
            # find index for training and test data
            train_index = np.squeeze(np.argwhere(blocks_mini_mean != i_block))
            test_index = np.squeeze(np.argwhere(blocks_mini_mean == i_block))

            # find train and test data
            X_train = trials_mini_mean[train_index]
            y_train = labels_mini_mean[train_index]
            resp_train = resp_mini_mean[train_index]
            X_test = trials_mini_mean[test_index]
            y_test = labels_mini_mean[test_index]

            # temporal decoding for each time point
            for time_ind in range(n_times_resample):
                B1 = X_train[:, :, time_ind]  # training data
                B2 = X_test[:, :, time_ind]  # test data
                C1 = resp_train  # predicted channel outputs for training data

                # train data, estimate weight matrix
                W = np.linalg.lstsq(C1, B1, rcond=None)[0]
                # make prediction, estimate unshifted channel responses (n_posBins, n_channels)
                C2_unshift = np.linalg.lstsq(W.T, B2.T, rcond=None)[0].T

                # shift prediction to common center
                C2_shift = np.zeros(C2_unshift.shape)
                n_shift = math.ceil(n_channels / 2)
                for c in range(n_posBins):
                    shift_index = n_shift - int(y_test[c]) + 1
                    C2_shift[c, :] = np.roll(C2_unshift[c, :], shift_index)
                # average shifted channel response across channels
                C2_mean = np.mean(C2_shift, axis=0)
                CTF_pred[i_block, i_iter, time_ind, :] = C2_mean

    # average CTF prediction across three-block cross-validation
    CTF_mean = np.squeeze(np.mean(CTF_pred, axis=0))
    # average CTF prediction across iteration
    CTF_mean = np.squeeze(np.mean(CTF_mean, axis=0))

    # save averaged CTF for each subject into disk
    np.savetxt(classification_results_dir + '/' + subID + '_' + time_string + '_CTF_predictions.txt', CTF_mean, fmt='%f')

    # record averaged CTF for each subject
    CTF_pred_all[sub_index, :, :] = CTF_mean
    # quantify CTF sensitivity
    CTF_slope = np.zeros(n_times_resample)
    for time_ind in range(n_times_resample):
        data = np.squeeze(CTF_mean[time_ind, :])
        x = np.linspace(1, 5, num=5, dtype=int)
        y = [data[0], np.mean([data[1], data[7]]), np.mean([data[2], data[6]]), np.mean([data[3], data[5]]), data[4]]
        fit = np.polyfit(x, y, 1)
        CTF_slope[time_ind] = fit[0]

    # save CTF slope for each subject into disk
    np.savetxt(classification_results_dir + '/' + subID + '_' + time_string + '_CTF_slope.txt', CTF_slope, fmt='%f')

    # record CTF slope for all subject
    CTF_slope_all[sub_index, ] = CTF_slope

    # record mini-block assignment for all subjects
    trials_index_mini_all[subID] = trials_index_mini_iter
    labels_mini_all[subID] = labels_mini_iter
    blocks_mini_all[subID] = blocks_mini_iter

    sub_index += 1

# save mini-block assignment for all subjects
np.save(classification_results_dir + '/' + time_string + 'trials_index_mini_all.npy', trials_index_mini_all)
np.save(classification_results_dir + '/' + time_string + 'labels_mini_all.npy', trials_index_mini_all)
np.save(classification_results_dir + '/' + time_string + 'blocks_mini_all.npy', trials_index_mini_all)


# change time points into ms
time_points_resample = time_points_resample*1000

# plot heat-map results of CTF prediction
# average CTF_pred_all across subjects
CTF_pred_mean = np.squeeze(np.mean(CTF_pred_all, axis=0))

# # change data into DataFrame
# # flatten data into 1-D
# CTF_pred_mean_df = CTF_pred_mean.T.flatten()
# # define x-axis
# Times =time_points_resample.astype(int)
# Times_df = np.tile(Times, n_channels)
# # define y-axis
# Channel_offset = np.linspace(-180, 135, 8, endpoint=True, dtype=int)
# Channel_offset_df = np.repeat(Channel_offset, n_times_resample)
# data = {'Channel Offset': Channel_offset_df, 'Times': Times_df, 'Channel Response': CTF_pred_mean_df}
# df = pd.DataFrame(data=data)
# df = df.pivot('Channel Offset', 'Times', 'Channel Response')

fig, ax = plt.subplots(figsize=(12, 5))
# ax = sns.heatmap(df, xticklabels=200, cmap="viridis", linecolor=None, shading='gouraud', edgecolors='face')
im = ax.imshow(CTF_pred_mean.T, extent=(-300, 1250, 180, -180), interpolation='bicubic', cmap='viridis', aspect='auto', origin='upper')
ax.set_xticks(np.linspace(-300, 1200, 6, endpoint=True, dtype=int))
ax.set_yticks(np.linspace(-180, 180, 5, endpoint=True, dtype=int))
ax.set_xlabel('Times (ms)', size=13)
ax.set_ylabel('Channel offset(°)', size=13)
ax.set_title('Average alpha-band CTF', size=13)
fig.colorbar(im, ax=ax, extendrect=True)
plt.show()

plt.savefig('heatmap.png')



# statistical analysis
from scipy import stats
from mne.stats import (fdr_correction)

# change data into dataframe
# flatten data into 1-D
CTF_slope_all_df = CTF_slope_all.flatten()
sub_all = np.repeat(subIDs, n_times_resample)
timePoints_all = np.tile(time_points_resample, sub_num)

data = {'subject': sub_all, 'time': timePoints_all, 'scores': CTF_slope_all_df}
df = pd.DataFrame(data=data)

# one-sample t-test with 0
t_all = np.zeros(n_times_resample)
p_all = np.zeros(n_times_resample)
sig_all = np.zeros(n_times_resample)
for i in range(n_times_resample):
    current_time = time_points_resample[i]
    current_scores = df[df['time'].isin([current_time])]

    t, p_twoTail = stats.ttest_1samp(current_scores['scores'], 0)
    p_FDR = fdr_correction(p_twoTail)[1]

    if p_FDR <= .05:
        sig = 1
    else:
        sig = 0

    t_all[i] = t
    p_all[i] = p_FDR
    sig_all[i] = sig

# record significant time points for plot
x_sig = time_points_resample[np.nonzero(sig_all)]
y_sig = np.repeat(0.25, len(x_sig))

# plot CTF slope results
fig, ax = plt.subplots()
sns.lineplot(data=df, x='time', y='scores', ci=95, n_boot=10000, err_style='band')
sns.scatterplot(x=x_sig, y=y_sig, c=['r'], s=20, marker='_')
ax.axhline(0, color='k', linestyle='-')
ax.set_title('IEM decoding for 14 subjects')
ax.set_xlabel('Time (ms)', size=13)
ax.set_ylabel('CTF slope', size=13)
ax.text(400, 0.22, 'bootstrap=10000, 95% CI', fontsize=12)
ax.text(400, 0.20, 'FDR correction, two-tail, p<0.05', fontsize=12)
# add events
ax.axvline(.0, color='k', linestyle='--', label='stimuli onset')
ax.axvline(.1, color='b', linestyle='--', label='stimuli offset')
ax.axvline(x_sig[0], ymax=0.95, color='r', linestyle='--', label='First significant point')
ax.legend(loc='best', fontsize='xx-small')

plt.savefig('IEM_decoding.png')





# ----------------------- Permutation IEM decoding for all subjects ----------------------------#
n_permutation = 100
# pre-allocate space for predicted CTF for all subjects and permutations
CTF_pred_all_perm = np.zeros([n_permutation, sub_num, n_times_resample, n_channels])
# pre-allocate space for CTF slopes for all subjects and permutations
CTF_slope_all_perm = np.zeros([n_permutation, sub_num, n_times_resample])

for i_perm in trange(n_permutation):
    sub_index = 0
    for subID in subIDs:
        # get subject's data
        X = X_all[subID]
        # grab assigned data for all iterations
        trials_index_mini_iter = trials_index_mini_all[subID]
        labels_mini_iter = labels_mini_all[subID]
        blocks_mini_iter = blocks_mini_all[subID]

        # pre-allocate space for predicted CTF for test data
        CTF_pred = np.zeros([n_blocks, n_iteration, n_times_resample, n_channels])

        # Loop through each iteration
        for i_iter in trange(n_iteration):
            # grab assigned data for each iteration
            trials_index_mini = np.squeeze(trials_index_mini_iter[i_iter, ])
            labels_mini = np.squeeze(labels_mini_iter[i_iter, ])
            blocks_mini = np.squeeze(blocks_mini_iter[i_iter, ])
            # get trials_mini according to trials_index_mini
            trials_mini = X[trials_index_mini.astype(int)]
            # shuffle labels randomly within a block
            unique_blocks = np.unique(blocks_mini)
            for i_block in unique_blocks:
                index = np.squeeze(np.argwhere(blocks_mini == i_block))
                labels = np.squeeze(labels_mini[index])
                random.shuffle(labels)
                labels_mini[index] = labels

            # average trials in each mini-block
            trials_mini_mean, labels_mini_mean, blocks_mini_mean = average_trials(trials_mini, labels_mini,
                                                                                  blocks_mini)

            # predict channel response for averaged data
            resp_mini_mean = np.zeros((len(labels_mini_mean), n_channels))
            for ind in range(len(labels_mini_mean)):
                label = labels_mini_mean[ind]
                resp_mini_mean[ind, :] = basis_set[int(label) - 1, :]

            # loop through blocks, holding each out as the test set
            for i_block in range(n_blocks):
                # find index for training and test data
                train_index = np.squeeze(np.argwhere(blocks_mini_mean != i_block))
                test_index = np.squeeze(np.argwhere(blocks_mini_mean == i_block))

                # find train and test data and labels
                X_train = trials_mini_mean[train_index]
                resp_train = resp_mini_mean[train_index]
                X_test = trials_mini_mean[test_index]
                y_test = labels_mini_mean[test_index]

                # temporal decoding for each time point
                for time_ind in range(n_times_resample):
                    B1 = X_train[:, :, time_ind]  # training data
                    B2 = X_test[:, :, time_ind]  # test data
                    C1 = resp_train  # predicted channel outputs for training data

                    # train data, estimate weight matrix
                    W = np.linalg.lstsq(C1, B1, rcond=None)[0]
                    # make prediction, estimate unshifted channel responses (n_posBins, n_channels)
                    C2_unshift = np.linalg.lstsq(W.T, B2.T, rcond=None)[0].T

                    # shift prediction to common center
                    C2_shift = np.zeros(C2_unshift.shape)
                    n_shift = math.ceil(n_channels / 2)
                    for c in range(n_posBins):
                        shift_index = n_shift - int(y_test[c]) + 1
                        C2_shift[c, :] = np.roll(C2_unshift[c, :], shift_index)
                    # average shifted channel response across channels
                    C2_mean = np.mean(C2_shift, axis=0)
                    CTF_pred[i_block, i_iter, time_ind, :] = C2_mean

        # average CTF prediction across three-block cross-validation
        CTF_mean = np.squeeze(np.mean(CTF_pred, axis=0))
        # average CTF prediction across iteration
        CTF_mean = np.squeeze(np.mean(CTF_mean, axis=0))

        # record averaged CTF for all subjects
        CTF_pred_all_perm[i_perm, sub_index, :, :] = CTF_mean


        # quantify CTF sensitivity
        CTF_slope = np.zeros(n_times_resample)
        for time_ind in range(n_times_resample):
            data = np.squeeze(CTF_mean[time_ind, :])
            x = np.linspace(1, 5, num=5, dtype=int)
            y = [data[0], np.mean([data[1], data[7]]), np.mean([data[2], data[6]]), np.mean([data[3], data[5]]),
                 data[4]]
            fit = np.polyfit(x, y, 1)
            CTF_slope[time_ind] = fit[0]

        # record CTF slope for all permutations
        CTF_slope_all_perm[i_perm, sub_index, ] = CTF_slope

    sub_index += 1

# save averaged CTF for all permutations into disk
np.save(classification_results_dir + '/' + time_string + '_CTF_predictions_permutation.npy', CTF_pred_all_perm)
# save CTF slopes for all permutations into disk
np.save(classification_results_dir + '/' + time_string + '_CTF_slope_permutation.npy', CTF_slope_all_perm)


# permutation test
# calculate the real one-sample t-stat
mean_real = np.squeeze(np.mean(CTF_slope_all, axis=0))
se_real = np.squeeze(np.std(CTF_slope_all, axis=0)/np.sqrt(sub_num))
t_real = mean_real/se_real
# calculate one-sample t-stats for permutations
mean_perm = np.squeeze(np.mean(CTF_slope_all_perm, axis=1))
se_perm = np.squeeze(np.std(CTF_slope_all_perm, axis=1)/np.sqrt(sub_num))
t_perm = mean_perm/se_perm
# calculate p-value and find significant point
p = np.zeros(n_times_resample)
sig = np.zeros(n_times_resample)
p_threshold = 0.001
for t in range(n_times_resample):
    current_t_real = t_real[t]
    current_t_perm = t_perm[:, t]
    current_p = np.sum(current_t_real < current_t_perm)/n_permutation
    p[t] = current_p
    if current_p < p_threshold:
        sig[t] = 1
    else:
        sig[t] = 0

# record significant time points for plot
x_sig = time_points_resample[np.nonzero(sig)]
y_sig = np.repeat(0.25, len(x_sig))
# find the first significant point start from the stimuli onset
sig_index = np.where(x_sig > 0)

# plot CTF slope results with permutation results
n_boot = 10000
fig, ax = plt.subplots()
sns.lineplot(data=df, x='time', y='scores', ci=95, n_boot=n_boot, err_style='band')
sns.scatterplot(x=x_sig, y=y_sig, c=['r'], s=20, marker='_')
ax.axhline(0, color='k', linestyle='-')
ax.set_title('IEM decoding for 14 subjects')
ax.set_xlabel('Time (ms)', size=13)
ax.set_ylabel('CTF slope', size=13)
ax.text(400, 0.22, f'bootstrap={n_boot}, 95% CI', fontsize=12)
ax.text(400, 0.20, f'permutation test, one-tail, p<{p_threshold}', fontsize=12)
# add events
ax.axvline(.0, color='k', linestyle='--', label='stimuli onset')
ax.axvline(.1, color='b', linestyle='--', label='stimuli offset')
ax.axvline(x_sig[sig_index[0]], ymax=0.95, color='r', linestyle='--', label='First significant point')
ax.legend(loc='best', fontsize='xx-small')

plt.savefig('IEM_decoding_permutation.png')