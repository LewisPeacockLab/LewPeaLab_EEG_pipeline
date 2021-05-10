# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Preprocessing.py
# Time       ：2021/3/5 20:05
# Author     ：ZoeDuan
# version    ：python 3.7
# Description：
                The pipeline of preprocessing EEG data from Foster et al(2017)
                Data: Experiment 2a, raw data
                Data source: https://osf.io/vw4uc/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (create_eog_epochs, find_eog_events)


# -------------------------Import data from Brain Vision EEG file------------------------- #
# input subject's ID you want to preprocess
subID = 1

dir = os.getcwd() + '/Preprocessing_data/sub_'
datapath = dir + str(subID) + '/'

raw_fname = datapath + str(subID) + '_JJF_16_1.vhdr'
raw = mne.io.read_raw_brainvision(raw_fname, eog=('HEOG', 'VEOG'), preload=True)

# save raw data
raw.save(datapath + str(subID) + '_raw.fif', overwrite=True)
# raw = mne.io.read_raw_fif(datapath + str(subID) + '_raw.fif', preload=True)


# -------------------------Check raw data------------------------- #
print(raw)
print(raw.info)
print(raw.info.keys())
# check channel information
print('channel numbers: {}'.format(raw.info['nchan']))
print('channel types: {}'.format(raw.get_channel_types()))
print('channel names: {}'.format(raw.info['ch_names']))
# check time information
print('time samples in total: {}'.format(raw.n_times))
print('time_secs: from {} s to {} s'.format(raw.times[0], raw.times[-1]))


# -------------------------Work with sensor locations------------------------- #
# check idealized montages for EEG systems included in MNE

montage_dir = os.path.join(os.path.dirname(mne.__file__),
                           'channels', 'data', 'montages')
print(sorted(os.listdir(montage_dir)))
# visualize standard_1020 montage used in Awh's paper
ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
print(ten_twenty_montage)
fig = ten_twenty_montage.plot(kind='3d')
fig.gca().view_init(azim=70, elev=15)
ten_twenty_montage.plot(kind='topomap', show_names=False)


# # visualize sensor information in our raw data

fig = plt.figure()
ax2d = fig.add_subplot(121)
ax3d = fig.add_subplot(122, projection='3d')
raw.plot_sensors(ch_type='eeg', axes=ax2d)
raw.plot_sensors(ch_type='eeg', axes=ax3d, kind='3d')
ax3d.view_init(azim=70, elev=15)


# -------------------------Work with events------------------------- #
# read events from annotations
events, event_dict = mne.events_from_annotations(raw)
print(len(events))
print(events)
print(event_dict)
# Visualize events
mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                    first_samp=raw.first_samp, event_id=event_dict)

# Visualize raw data with events
# define time periods you want to plot
tmin = events[1, 0] / raw.info['sfreq'] - 2
tmax = events[18, 0] / raw.info['sfreq'] + 5
# plot the raw data
raw_show = raw.copy().crop(tmin=tmin, tmax=tmax)
raw_show.plot(events=events, duration=6)

# check the whole data using the follows
# but be careful, since it may make your computer stuck
# raw.plot(start=event_start_time, duration=6)


# -------------------------Rereference------------------------- #
# Re-reference the algebraic average of the left(TP9) and right(TP10) mastoids
# add reference channels to data that consists of all zeros
raw_new_ref = mne.add_reference_channels(raw, ref_channels=['TP10'])
# set reference to average of ['TP9', 'TP10']
raw_new_ref.set_eeg_reference(ref_channels=['TP9', 'TP10'])

# drop referenced channels
raw_new_ref.drop_channels(['TP9', 'TP10'])
# save re-referenced data
raw_new_ref.save(datapath + str(subID) + '_newRef_raw.fif', overwrite=True)
# raw_new_ref = mne.io.read_raw_fif(datapath + str(subID) + '_newRef_raw.fif', preload=True)

print('channel numbers: {}'.format(raw_new_ref.info['nchan']))
print('channel names: {}'.format(raw_new_ref.info['ch_names']))

# visualize re-referenced data
raw_new_ref_show = raw_new_ref.copy().crop(tmin=tmin, tmax=tmax)
raw_new_ref_show.plot(events=events, duration=6)



# -------------------------discard bad channels------------------------- #
# check bad channels in raw data
print(raw_new_ref.info['bads'])

# mark bad channels manually
# by click the channel’s trace in the plot area
# raw_new_ref.plot(duration=6)

# add bad channels
# raw_new_ref.info['bads'].extend(['Fp1', 'Fp2'])

# delete bad channels
# raw_new_ref.info['bads'].pop(-1)

# check bad channels again
# print(raw_new_ref.info['bads'])


# delete bad channels, pick up rest channels
picks = mne.pick_types(raw_new_ref.info, eeg=True, eog=True, misc=True, exclude='bads')
print(np.array(raw_new_ref.ch_names)[picks])


# -------------------------Detect artifacts------------------------- #
# check for power line noise
raw_new_ref.plot_psd(fmax=250, picks=picks, average=True)

# detect EOG (Ocular artifacts)
eog_epochs = create_eog_epochs(raw_new_ref, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine='mean')
eog_epochs.average().plot_joint()


# -------------------------Deal with artifacts------------------------- #
# if EOG artifacts are too much,
# you can repair them with ICA instead of rejecting those bad spans
# however, if EOG artifacts are not too much,
# we recommend you just delete those bad EOG spans

# -------------------------Repair artifacts with ICA------------------------- #
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)

# remove slow drifts
filt_raw_new_ref = raw_new_ref.copy()
filt_raw_new_ref.filter(l_freq=1., h_freq=None)

# filt and plot the ica
ica = ICA(n_components=15, random_state=97)
ica.fit(filt_raw_new_ref)

ica.plot_sources(raw_new_ref, show_scrollbars=False)
ica.plot_components(inst=raw_new_ref)

# select component you want to exclude
ica.exclude = [0, 2, 3]

# apply ica reconstruction
raw_reconst = raw_new_ref.copy()
ica.apply(raw_reconst)
# compare data before and after ica
raw_new_ref_show = raw_new_ref.copy().crop(tmin=tmin, tmax=tmax)
raw_new_ref_show.plot(duration=6)
raw_reconst_show = raw_reconst.copy().crop(tmin=tmin, tmax=tmax)
raw_reconst_show.plot(duration=6)

raw_reconst.save(datapath + str(subID) + '_reconst_newRef_raw.fif', overwrite=True)
# raw_reconst = mne.io.read_raw_fif(datapath + str(subID) + '_reconst_newRef_raw.fif', preload=True)

# make sure two different artifact method have the same variable name
raw_artifact = raw_reconst


# -------------------------Reject bad data spans------------------------- #
# Annotate bad spans of data
# Annotate EOG programmatically, annotate from [-0.25, 0.25]
eog_events = find_eog_events(raw_new_ref)
onsets = eog_events[:, 0] / raw_new_ref.info['sfreq'] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ['bad blink'] * len(eog_events)
blink_annot = mne.Annotations(onsets, durations, descriptions,
                              orig_time=raw_new_ref.info['meas_date'])

# visualize the bad blinks
# raw_new_ref_badBlink = raw_new_ref.copy().set_annotations(blink_annot)
# raw_new_ref_show = raw_new_ref_badBlink.copy().crop(tmin=tmin, tmax=tmax)
# raw_new_ref_show.plot(duration=6)

# get raw data annotations
raw_new_ref_annot = raw_new_ref.annotations
# add EOG annotations to raw data
new_annot = raw_new_ref_annot + blink_annot
raw_with_bads = raw_new_ref.copy()
raw_with_bads.set_annotations(new_annot)

# visualize raw data with new annotations
raw_with_bads_show = raw_with_bads.copy().crop(tmin=tmin, tmax=tmax)
raw_with_bads_show.plot(duration=6)

# Annotate bad spans of data manually
# This may cost you lots of time, be careful
# fig = raw_with_bads.plot(duration=6)
# fig.canvas.key_press_event('a')

raw_with_bads.save(datapath + str(subID) + '_artifact_newRef_raw.fif', overwrite=True)
# raw_with_bads = mne.io.read_raw_fif(datapath + str(subID) + '_artifact_newRef_raw.fif', preload=True)

raw_artifact = raw_with_bads

# -------------------------Re-sample data------------------------- #
# sample rate should be 500Hz
print(raw_artifact.info['sfreq'])

# if you want, you can resample your data like this
# raw_downsampled = raw_with_bads.copy().resample(sfreq=200)


# -------------------------Filtering------------------------- #
# Notch-filter is not necessary for this data
# but if you want, you can try this

# raw_notch = raw.copy().notch_filter(freqs=60, picks=picks)
# for title, data in zip(['Unfiltered', 'Notch_Filtered'], [raw, raw_notch]):
#     fig = data.plot_psd(fmax=250, average=False)
#     fig.subplots_adjust(top=0.85)
#     fig.suptitle(title, size='xx-large', weight='bold')


# filt raw data with band-pass filter
# low cut-off=0.01 Hz, high cut-off=80Hz
raw_filt = raw_artifact.copy().filter(l_freq=0.01, h_freq=80, picks=picks)
raw_filt.save(datapath + str(subID) + '_filt_artifact_newRef_raw.fif', overwrite=True)
# raw_filt = mne.io.read_raw_fif(datapath + str(subID) + '_filt_artifact_newRef_raw.fif', preload=True)

# visualize filtered raw data
raw_filt_show = raw_filt.copy().crop(tmin=tmin, tmax=tmax)
raw_filt_show.plot()



# -------------------------Epoch data and baseline correction------------------------- #
# re-define events with behavioral data
# read behavioral data
dataFile = datapath + str(subID) + '_target_posBin.txt'
target_posBin = np.loadtxt(dataFile)

# create new events including both block and posBin information
# pick events: block, stimuli_onset
events_block = events[events[:, 2] < 21, 2]
events_new = events[events[:, 2] == 21, :]
events_new[:, 2] = events_block*10 + target_posBin

# visualize raw data with new events
raw_filt_show = raw_filt.copy().crop(tmin=tmin, tmax=tmax)
raw_filt_show.plot(events=events_new, duration=6)

# define epoch parameters
# Epochs timeWindow: -800ms ~ 1750ms, 0 = stimuli onset
# this timeWindow should be longer than the one you really want,
# which can avoid edge artifacts when performing time-frequency analysis
tmin_epoch = -0.8
tmax_epoch = 1.75
# baseline: -300ms ~ 0ms
baseline = (-0.3, 0)

epochs = mne.Epochs(raw_filt, events=events_new,
                    tmin=tmin_epoch, tmax=tmax_epoch,
                    baseline=baseline, picks=picks,
                    reject_by_annotation=True, preload=True)

# reject based on channel amplitude
reject = dict(eeg=130e-6)       # 100 µV
flat = dict(eeg=1e-6)           # 1 µV#
epochs.drop_bad(reject=reject, flat=flat)

# check epochs
epochs.plot_drop_log()
print(epochs)
print(epochs.event_id)

# visualize epochs
epochs.plot(events=events_new, picks=picks, n_epochs=5)

# save epochs
epochs.save(datapath + str(subID) + '_preprocessed_epo.fif', overwrite=True)
# epochs = mne.read_epochs('1_preprocessed_epo.fif', preload=True)

# delete unnecessary data
del raw, raw_new_ref, raw_with_bads, raw_filt


# visualize selected epochs
# epochs['11'].plot(events=events_new, picks=picks, n_epochs=5)

# visualize evoked data
evoked = epochs.average()
evoked.plot()













