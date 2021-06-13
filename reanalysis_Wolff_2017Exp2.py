#!/usr/bin/env python
# coding: utf-8

# In[135]:


"""
# File       : Wolff et al., 2017,Preprocessed
# Author     : Ziyao Zhang
# Description: decoding orientation based on mahalanobis distance
# Reference. : Wolff et al., 2017 Dynamic hidden states underlying working-memory-guided behavior Nat Neur
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import (create_eog_epochs, find_eog_events)
import scipy.io
import math
import scipy as sp
from numpy import matlib as mp
import scipy.spatial.distance
import scipy.ndimage
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf

#---------define variables------------
angspace = np.array(np.arange(-math.pi,math.pi,math.pi/6))
bin_width = math.pi/6
s_factor = 8
subID = np.linspace(1,19,19).astype(int) 
datapath = os.path.join(os.getcwd(), 'RawData')
outpath = os.getcwd() + 'Subdec/'
timepoint=300

#-----------functions to calculate mahal distance

def covdiag(x):
    t,n=x.shape
    meanx = np.mean(x,0)
    x=x-mp.repmat(meanx,t,1);

    # compute sample covariance matrix and prior
    sample = (1/t)*np.matmul(np.transpose(x),x)
    prior = np.diag(np.diag(sample))
    
    # compute shrinkage parameters
    d = 1/n * np.linalg.norm(sample-prior,'fro')**2
    y = x**2
    r2 = 1/n/t**2*sum(sum(np.matmul(np.transpose(y),y)))-1/n/t*sum(sum(sample**2))
    
    #compute the estimator
    shrinkage = np.maximum(0,min(1,r2/d))
    sigma = shrinkage*prior+(1-shrinkage)*sample
    
    return sigma

def mahalTune_func(data,theta,angspace,bin_width):
    #save variables
    d_tune = np.empty([data.shape[0],len(angspace),data.shape[2]])
    d_tune[:] = np.NaN
    cos_amp = np.empty([data.shape[0],data.shape[2]])
    cos_amp[:] = np.NaN
    trl_ind = np.arange(data.shape[0])
    
    for trl in trl_ind:
        trn_dat = data[np.where(trl_ind != trl)[0],:,:]
        trn_angle = theta[np.where(trl_ind != trl)[0]]
        m = np.empty([len(angspace),trn_dat.shape[1],trn_dat.shape[2]])
        m[:] = np.NaN
        # average the training data into orientation bins relative to the test orientation
        for b in np.arange(len(angspace)):
            m[b,:,:] = np.mean(trn_dat[np.where(np.abs(np.angle(np.exp(1j*trn_angle)/np.exp(1j*(theta[trl] - angspace[b]))))<bin_width)[0],:,:],0)
        #msg = str(np.round(trl/data.shape[0]*100,4)) + '% processed'
        #print(msg)
        
        for ti in range(data.shape[2]):
            if ~np.isnan(trn_dat[:,:,ti]).all():
                #model = GraphicalLassoCV()
                #model.fit(trn_dat[:,:,ti])
                #cov_ = model.covariance_
                #sigma = model.precision_
                #lw_cov_, _ = ledoit_wolf(trn_dat[:,:,ti])
                #sigma = np.linalg.inv(lw_cov_)
        
                # covariance matrix
                sigma = np.linalg.inv(covdiag(trn_dat[:,:,ti])) #np.cov(np.transpose(trn_dat[:,:,ti]))#
                for v in np.arange(m.shape[0]):
                    d_tune[trl,v,ti] = scipy.spatial.distance.mahalanobis(np.squeeze(m[v,:,ti]),np.squeeze(data[trl,:,ti]),sigma)
                cos_amp[trl,ti] = -(np.mean(np.multiply(np.cos(angspace),np.transpose(np.squeeze(d_tune[trl,:,ti])))))

    return cos_amp, d_tune

def norm_chan(x,dim1):
    # x: ndarray trial by channel by time
    # dim1: dimension to average, channel
    
    normed_x = np.ones(x.shape)
    normed_x[:] = np.NaN
    xmean = np.mean(x,dim1)
    for i in range(x.shape[dim1]):
        normed_x[:,i,:] = x[:,i,:]-xmean
    
    return normed_x

#---------variable to save-----------
dec_impul1_early = np.ones([len(subID),timepoint])
dec_impul1_late = np.ones([len(subID),timepoint])
dec_impul2_early = np.ones([len(subID),timepoint])
dec_impul2_late = np.ones([len(subID),timepoint])

dec_impul1_early[:] = np.NaN
dec_impul1_late[:] = np.NaN
dec_impul2_early[:] = np.NaN
dec_impul2_late[:] = np.NaN

#---------load data -----------------

for i in subID:
    print('sub:',i,'...',np.round(i/len(subID)*100,2),'%')
    fname = 'Dynamic_hidden_states_exp2_' + str(i) + '.mat'
    fpath = os.path.join(datapath,fname)
    raw = scipy.io.loadmat(fpath)
    
    rawdata = raw['exp2_data']
    mem_sess_1 = rawdata['EEG_mem_items_sess1'][0][0][0][0]
    mem_sess_2 = rawdata['EEG_mem_items_sess1'][0][0][0][0]
    impul1_sess_1 = rawdata['EEG_impulse1_sess1'][0][0][0][0]
    impul2_sess_1 = rawdata['EEG_impulse2_sess1'][0][0][0][0]
    impul1_sess_2 = rawdata['EEG_impulse1_sess2'][0][0][0][0]
    impul2_sess_2 = rawdata['EEG_impulse2_sess2'][0][0][0][0]
    beh_sess_1 = rawdata['Results_sess1'][0,0]
    beh_sess_2 = rawdata['Results_sess2'][0,0]
    beh_header = rawdata['Results_header'][0,0]
    time = impul1_sess_1['time'][0]
    
    
#--------exclude bad trials--------
    incl_impul1_sess_1 = np.logical_not(np.in1d(range(impul1_sess_1['trial'].shape[0]),(impul1_sess_1['bad_trials']-1)))
    incl_impul2_sess_1 = np.logical_not(np.in1d(range(impul2_sess_1['trial'].shape[0]),(impul2_sess_1['bad_trials']-1)))
    sel_impul1_sess1 = impul1_sess_1['trial'][incl_impul1_sess_1,:,:]
    sel_impul2_sess1 = impul2_sess_1['trial'][incl_impul2_sess_1,:,:]

    incl_impul1_sess_2 = np.logical_not(np.in1d(range(impul1_sess_2['trial'].shape[0]),(impul1_sess_2['bad_trials']-1)))
    incl_impul2_sess_2 = np.logical_not(np.in1d(range(impul2_sess_2['trial'].shape[0]),(impul2_sess_2['bad_trials']-1)))
    sel_impul1_sess2 = impul1_sess_2['trial'][incl_impul1_sess_2,:,:]
    sel_impul2_sess2 = impul2_sess_2['trial'][incl_impul2_sess_2,:,:]

#--------normalization: minus the mean across channels---------
    norm_sel_impul1_sess1 = norm_chan(sel_impul1_sess1,1)
    norm_sel_impul2_sess1 = norm_chan(sel_impul2_sess1,1)

    norm_sel_impul1_sess2 = norm_chan(sel_impul1_sess2,1)
    norm_sel_impul2_sess2 = norm_chan(sel_impul2_sess2,1)

#--------extract memory item angles------------------
    mem_angle1 = beh_sess_1[:,0:2]*2
    mem_angle2 = beh_sess_2[:,0:2]*2

#--------decode------------
    dec_impul1_early1 = mahalTune_func(norm_sel_impul1_sess1,mem_angle1[incl_impul1_sess_1,0],angspace,bin_width)
    dec_impul1_late1 = mahalTune_func(norm_sel_impul1_sess1,mem_angle1[incl_impul1_sess_1,1],angspace,bin_width)
    dec_impul2_early1 = mahalTune_func(norm_sel_impul2_sess1,mem_angle1[incl_impul2_sess_1,0],angspace,bin_width)
    dec_impul2_late1 = mahalTune_func(norm_sel_impul2_sess1,mem_angle1[incl_impul2_sess_1,1],angspace,bin_width)

    dec_impul1_early2 = mahalTune_func(norm_sel_impul1_sess2,mem_angle2[incl_impul1_sess_2,0],angspace,bin_width)
    dec_impul1_late2 = mahalTune_func(norm_sel_impul1_sess2,mem_angle2[incl_impul1_sess_2,1],angspace,bin_width)
    dec_impul2_early2 = mahalTune_func(norm_sel_impul2_sess2,mem_angle2[incl_impul2_sess_2,0],angspace,bin_width)
    dec_impul2_late2 = mahalTune_func(norm_sel_impul2_sess2,mem_angle2[incl_impul2_sess_2,1],angspace,bin_width)
    
#-------save results--------
    outfile = os.getcwd() + '/Subdec/'+str(i)+'dec'
    np.savez(outfile+'_impul1_cov_corr_early1', dec_impul1_early1_cos = dec_impul1_early1[0], d_tune=dec_impul1_early1[1])
    np.savez(outfile+'_impul1_cov_corr_late1', dec_impul1_late1_cos = dec_impul1_late1[0], d_tune=dec_impul1_late1[1])
    np.savez(outfile+'_impul2_cov_corr_early1', dec_impul2_early1_cos = dec_impul2_early1[0], d_tune=dec_impul2_early1[1])
    np.savez(outfile+'_impul2_cov_corr_late1', dec_impul2_late1_cos = dec_impul2_late1[0], d_tune=dec_impul2_late1[1])

    np.savez(outfile+'_impul1_cov_corr_early2', dec_impul1_early2_cos = dec_impul1_early2[0], d_tune=dec_impul1_early2[1])
    np.savez(outfile+'_impul1_cov_corr_late2', dec_impul1_late2_cos = dec_impul1_late2[0], d_tune=dec_impul1_late2[1])
    np.savez(outfile+'_impul2_cov_corr_early2', dec_impul2_early2_cos = dec_impul2_early2[0], d_tune=dec_impul2_early2[1])
    np.savez(outfile+'_impul2_cov_corr_late2', dec_impul2_late2_cos = dec_impul2_late2[0], d_tune=dec_impul2_late2[1])
    
#-----individal average----
    dec_impul1_early[i-1,:] = scipy.ndimage.gaussian_filter((np.mean(dec_impul1_early1[0],0) + np.mean(dec_impul1_early2[0],0))/2,s_factor,mode='wrap')
    dec_impul1_late[i-1,:] = scipy.ndimage.gaussian_filter((np.mean(dec_impul1_late1[0],0) + np.mean(dec_impul1_late2[0],0))/2,s_factor,mode='wrap')
    dec_impul2_early[i-1,:] = scipy.ndimage.gaussian_filter((np.mean(dec_impul2_early1[0],0) + np.mean(dec_impul2_early2[0],0))/2,s_factor,mode='wrap')
    dec_impul2_late[i-1,:] = scipy.ndimage.gaussian_filter((np.mean(dec_impul2_late1[0],0) + np.mean(dec_impul2_late2[0],0))/2,s_factor,mode='wrap')

outfile_all=outpath + 'allsub_dec'
np.save(outfile+'_impul1_cov_corr_early',dec_impul1_early)
np.save(outfile+'_impul1_cov_corr_late',dec_impul1_late)
np.save(outfile+'_impul2_cov_corr_early',dec_impul2_early)
np.save(outfile+'_impul2_cov_corr_late',dec_impul2_late)


# In[136]:


#-----plot dec results---------
import seaborn as sns
import pandas as pd
from scipy import stats
from mne.stats import (fdr_correction)
condition = 4;
timepoint=300;

dec_impul1_early_df = dec_impul1_early.flatten()
dec_impul1_late_df = dec_impul1_late.flatten()
dec_impul2_early_df = dec_impul2_early.flatten()
dec_impul2_late_df = dec_impul2_late.flatten()

sub_all = np.repeat(subID, timepoint)
timePoints_all = np.tile(time,len(subID))

data_impul1_early = {'subject': sub_all, 'time': timePoints_all, 'scores': dec_impul1_early_df}
df_impul1_early = pd.DataFrame(data=data_impul1_early)
data_impul1_late = {'subject': sub_all, 'time': timePoints_all, 'scores': dec_impul1_late_df}
df_impul1_late = pd.DataFrame(data=data_impul1_late)
data_impul2_early = {'subject': sub_all, 'time': timePoints_all, 'scores': dec_impul2_early_df}
df_impul2_early = pd.DataFrame(data=data_impul2_early)
data_impul2_late = {'subject': sub_all, 'time': timePoints_all, 'scores': dec_impul2_late_df}
df_impul2_late = pd.DataFrame(data=data_impul2_late)



#----one-sample t-test with 0----
t_all = np.zeros([condition,timepoint])
p_all = np.zeros([condition,timepoint])
sig_all = np.zeros([condition,timepoint])

for c in range(condition):
    for i in range(timepoint):
        current_time = time[i]
        
        if c==0:
            current_scores = dec_impul1_early[:,i]
        elif c==1:
            current_scores = dec_impul1_late[:,i]
        elif c==2:
            current_scores = dec_impul2_early[:,i]
        elif c==3:
            current_scores = dec_impul2_late[:,i]

        t, p_twoTail = stats.ttest_1samp(current_scores[:], 0)
        p_FDR = fdr_correction(p_twoTail)[1]

        if p_FDR <= .05:
            sig = 1
        else:
            sig = 0

        t_all[c,i] = t
        p_all[c,i] = p_FDR
        sig_all[c,i] = sig

#---record significant time points for plot--
x_sig_impul1_early = time[np.nonzero(sig_all[0,:])]
y_sig_impul1_early = np.repeat(0.00210, len(x_sig_impul1_early))
x_sig_impul1_late = time[np.nonzero(sig_all[1,:])]
y_sig_impul1_late = np.repeat(0.00240, len(x_sig_impul1_late))
x_sig_impul2_early = time[np.nonzero(sig_all[2,:])]
y_sig_impul2_early = np.repeat(0.00210, len(x_sig_impul2_early))
x_sig_impul2_late = time[np.nonzero(sig_all[3,:])]
y_sig_impul2_late = np.repeat(0.00240, len(x_sig_impul2_late))


# In[137]:


#-----impulse 1-----
n_boot=10000
fig, ax = plt.subplots()
sns.lineplot(data=df_impul1_early, x='time', y='scores', ci=95, n_boot=n_boot, err_style='band',color='blue',label='Tested early')
sns.scatterplot(x=x_sig_impul1_early, y=y_sig_impul1_early, c=['b'], s=20, marker='_',linewidth=5)
sns.lineplot(data=df_impul1_late, x='time', y='scores', ci=95, n_boot=n_boot, err_style='band',color='red',label='Tested late')
sns.scatterplot(x=x_sig_impul1_late, y=y_sig_impul1_late, c=['r'], s=20, marker='_',linewidth=5)
ax.axhline(0, color='k', linestyle='-')
ax.set_title('Impulse 1')
ax.set_xlabel('Time (ms)', size=13)
ax.set_ylabel('Decoding accuracy', size=13)
ax.text(0.6, 0.00240, 'bootstrap=' + str(n_boot)+', 95% CI', fontsize=12)
ax.text(0.6, 0.00220, 'FDR correction, two-tail, p<0.05', fontsize=12)
# add events
ax.axvline(.0,ymax=0.05, color='k')
ax.axvline(.1,ymax=0.05, color='k')
ax.legend(loc=2, fontsize='large',edgecolor='none',facecolor='none')
plt.xlim([time[0],time[-1]])
plt.ylim([-0.0005,0.0025])
fig.savefig('impulse1_corr2.png')


# In[138]:


#-----impulse 2-----
fig, ax = plt.subplots()
sns.lineplot(data=df_impul2_early, x='time', y='scores', ci=95, n_boot=n_boot, err_style='band',color='blue',label='Tested early')
sns.scatterplot(x=x_sig_impul2_early, y=y_sig_impul2_early, c=['b'], s=20, marker='_',linewidth=5)
sns.lineplot(data=df_impul2_late, x='time', y='scores', ci=95, n_boot=n_boot, err_style='band',color='red',label='Tested late')
sns.scatterplot(x=x_sig_impul2_late, y=y_sig_impul2_late, c=['r'], s=20, marker='_',linewidth=5)
ax.axhline(0, color='k', linestyle='-')
ax.set_title('Impulse 2')
ax.set_xlabel('Time (ms)', size=13)
ax.set_ylabel('Decoding accuracy', size=13)
ax.text(0.6, 0.00240, 'bootstrap=' + str(n_boot)+', 95% CI', fontsize=12)
ax.text(0.6, 0.00220, 'FDR correction, two-tail, p<0.05', fontsize=12)
# add events
ax.axvline(.0,ymax=0.05, color='k')
ax.axvline(.1,ymax=0.05, color='k')
ax.legend(loc=2, fontsize='large',edgecolor='none',facecolor='none')
plt.xlim([time[0],time[-1]])
plt.ylim([-0.0005,0.0025])
fig.savefig('impulse2_corr2.png')


# In[119]:


print(sel_impul1_sess1.shape)


# In[121]:


print(np.mean(sel_impul1_sess1,1).shape)


# In[124]:


print(incl_impul1_sess_1)


# In[ ]:




