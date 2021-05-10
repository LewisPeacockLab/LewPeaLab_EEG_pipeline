# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : MVPA_multi-subjects_8_posBins.py
# Time       ：2021/4/16 10:40
# Author     ：ZoeDuan
# version    ：python 3.7
# Description：      The pipeline of decoding 8-class alpha-band power from Foster et al(2017)
#                   Data: Experiment 2a, 11 Subjects, preprocessed data (epochs)
"""

import os
import mne
import numpy as np
import time
import pandas as pd
import h5py
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, Vectorizer)

from mini_block_function_MVPA import *

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
prepared_data_dir = dir + '/prepared_data_8_posBins/'
exist = os.path.exists(prepared_data_dir)
if not exist:
    os.makedirs(prepared_data_dir)

# create file folder to save classification results
classification_results_dir = dir + '/classification_results_8_posBins/'
exist = os.path.exists(classification_results_dir)
if not exist:
    os.makedirs(classification_results_dir)


# define variables
# define band-pass frequency
l_freq, h_freq = 8, 12
# define decoding timeWindow -300ms ~ 1250ms
timeWindow = [-0.3, 1.25]
# define resample numbers for each time point
n_resample = 5

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

    n_trial = sub_power.shape[0]
    n_channel = sub_power.shape[1]
    n_times = sub_power.shape[2]

    # delete epochs to save memory
    del epochs, epochs_filt

    # down-sampling data based on n_resample
    n_times_resample = round(n_times / n_resample)
    time_points_resample = np.linspace(timeWindow[0], timeWindow[1], n_times_resample)
    sub_power_resample = np.zeros([n_trial, n_channel, n_times_resample], dtype=np.float)
    # down-sampling data by combining every n_resample data
    for t in range(n_times_resample):
        sub_power_resample[:, :, t] = np.average(sub_power[:, :, t * n_resample: (t + 1) * n_resample], axis=-1)
    assert sub_power_resample.shape == (n_trial, n_channel, n_times_resample)

    # define data for decoding
    X = np.array(sub_power_resample)
    y = np.array(labels)

    del sub_power_resample, labels

    # save each participants' alpha_band data
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


# run MVPA decoding for each subject

# define classifier
clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
# define cross-validation
cv = StratifiedShuffleSplit(n_splits=1, test_size=1/3)
# define scoring
scoring = 'accuracy'
# define iteration times
n_iteration = 10
# define trial numbers for mini-block
trials_mini_block = 5
# define the Temporal decoding object
# time_decod = SlidingEstimator(clf, n_jobs=1, scoring=scoring, verbose=True)
# define the Temporal generalization object
# time_gen = GeneralizingEstimator(clf, n_jobs=1, scoring=scoring, verbose=True)


# record results for all subjects
scores_all = np.empty([0])
sub_all = np.empty([0])
timePoints_all = np.empty([0])
weights_all = {}
for subID in subIDs:
    # read each subject's data
    X = X_all[subID]
    y = y_all[subID]
    # find information for current data
    n_trial = X.shape[0]
    n_channel = X.shape[1]
    n_times_resample = X.shape[2]

    # decode across time
    # pre-allocate space for scores and channel weights for each iteration
    scores_itr = np.zeros((n_iteration, n_times_resample))
    weights_itr = np.zeros((n_iteration, n_times_resample, n_channel))

    # Loop through each iteration
    time_start = time.time()
    for i in trange(n_iteration):
        # find the minimal trial number among labels
        min_trials = min_trials_calculation(y)
        # average single trials into mini blocks
        X_mini, y_mini = mini_block_transform(min_trials, trials_mini_block, X, y)
        # temporal decoding for each iteration
        for train_ind, test_ind in cv.split(X_mini, y_mini):
            X_train = X_mini[train_ind]
            y_train = y_mini[train_ind]
            X_test = X_mini[test_ind]
            y_test = y_mini[test_ind]
            for time_ind in range(n_times_resample):
                X_train_t = X_train[:, :, time_ind]
                X_test_t = X_test[:, :, time_ind]
                scaler = StandardScaler().fit(X_train_t)
                X_train_transformed = scaler.transform(X_train_t)
                X_test_transformed = scaler.transform(X_test_t)

                clf.fit(X_train_transformed, y_train)
                y_pred = clf.predict(X_test_transformed)
                score = accuracy_score(y_test, y_pred)
                scores_itr[i, time_ind] = score
                # calculate weight for each channel
                channel_weight = clf.coef_
                weights_itr[i, time_ind, :] = np.mean(channel_weight, axis=0)


    time_end = time.time()
    print('Time cost for temporal decoding for sub %s: %f s' % (subID, time_end - time_start))

    scores_final = np.mean(scores_itr, axis=0)
    weights_final = np.mean(weights_itr, axis=0)


    # save decoding scores for each subject into disk
    np.savetxt(classification_results_dir + '/' + subID + '_' + time_string + '_decoding_results.txt', scores_final, fmt='%f')
    np.savetxt(classification_results_dir + '/' + subID + '_' + time_string + '_channel_weights.txt', weights_final, fmt='%f')

    # save all decoding scores, subject index, and time points
    subs = np.repeat(subID, scores_final.size)
    sub_all = np.append(sub_all, subs, axis=0)
    scores_all = np.append(scores_all, scores_final, axis=0)
    timePoints_all = np.append(timePoints_all, time_points_resample_all[subID], axis=0)
    # calculate averaged weights across time span for each channel
    weights_mean = np.mean(abs(weights_final), axis=0)
    weights_mean = weights_mean[None, :]
    weights_all[subID] = weights_mean


# statistical analysis
from scipy import stats
from mne.stats import (fdr_correction)

# change data into dataframe
data = {'subject': sub_all, 'time': timePoints_all, 'scores': scores_all}
df = pd.DataFrame(data=data)

# one-sample t-test with chance level (0.125)
t_all = np.zeros(n_times_resample)
p_all = np.zeros(n_times_resample)
sig_all = np.zeros(n_times_resample)
for i in range(n_times_resample):
    current_time = time_points_resample[i]
    current_scores = df[df['time'].isin([current_time])]

    t, p_twoTail = stats.ttest_1samp(current_scores['scores'], 0.125)
    p_FDR = fdr_correction(p_twoTail)[1]

    if p_FDR <= .05:
        sig = 1
    else:
        sig = 0

    t_all[i] = t
    p_all[i] = p_FDR
    sig_all[i] = sig

# record significant time points for plot
x_sig = timePoints_all[np.nonzero(sig_all)]
y_sig = np.repeat(0.45, len(x_sig))
# s_sig = t_all[np.nonzero(sig_all)]

# plot subject-wise results
fig, ax = plt.subplots()
sns.lineplot(data=df, x='time', y='scores', ci=95, n_boot=1000, err_style='band')
sns.scatterplot(x=x_sig, y=y_sig, c=['r'], s=20, marker='s')
ax.axhline(.125, color='k', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('Accuracy')
ax.set_title('temporal decoding for %d subjects_eight_posBin' % sub_num)
ax.text(0.8, 0.4, 'p < .05, 95% CI', fontsize=12)
ax.text(0.8, 0.36, 'FDR correction', fontsize=12)
# add events
ax.axvline(.0, color='k', linestyle='--', label='stimuli onset')
ax.axvline(.1, color='b', linestyle='--', label='stimuli offset')
ax.axvline(x_sig[0], ymax=0.95, color='r', linestyle='--', label='First significant point')
ax.legend(loc='best', fontsize='xx-small')

plt.savefig('decoding.png')