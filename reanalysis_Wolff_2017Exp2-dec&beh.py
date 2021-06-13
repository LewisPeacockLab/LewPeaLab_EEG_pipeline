#!/usr/bin/env python
# coding: utf-8

# In[305]:


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
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr

#---------define variables------------
angspace = np.array(np.arange(-math.pi,math.pi,math.pi/6))
bin_width = math.pi/6
s_factor = 8
subID = np.linspace(1,19,19).astype(int)  #np.linspace(1,19,19).astype(int) 
raw_datapath = os.path.join(os.getcwd(), 'RawData')
datapath = os.path.join(os.getcwd(), 'Subdec')
outpath = os.getcwd() + 'dec&beh/'
timepoint=300
sampler = 500
timep = [-0.1,0.5]
intp = [0.1,0.5]
intpoints = np.zeros([1,2])[0]
intpoints[0] = (intp[0]-timep[0])*1000/(1000/sampler)-1
intpoints[1] = (intp[1]-timep[0])*1000/(1000/sampler)-1
inttimes = np.arange(timep[0]*1000+(intpoints[0]+1)*1000/sampler,timep[0]*1000+(intpoints[1]+1)*1000/sampler,1000/sampler)
dec_window=np.arange(intpoints[0]+1,intpoints[1]+1,1)
dec_window=dec_window.astype(int)

#----variables to save------
high_impul1_acc_early=np.ones([len(subID)])
high_impul1_acc_early[:]=np.NaN
low_impul1_acc_early=np.ones([len(subID)])
low_impul1_acc_early[:]=np.NaN

high_impul1_acc_late=np.ones([len(subID)])
high_impul1_acc_late[:]=np.NaN
low_impul1_acc_late=np.ones([len(subID)])
low_impul1_acc_late[:]=np.NaN

high_impul2_acc_early=np.ones([len(subID)])
high_impul2_acc_early[:]=np.NaN
low_impul2_acc_early=np.ones([len(subID)])
low_impul2_acc_early[:]=np.NaN

high_impul2_acc_late=np.ones([len(subID)])
high_impul2_acc_late[:]=np.NaN
low_impul2_acc_late=np.ones([len(subID)])
low_impul2_acc_late[:]=np.NaN

beta_impul1_acc_early=np.ones([len(subID)])
beta_impul1_acc_early[:]=np.NaN
beta_impul1_acc_late=np.ones([len(subID)])
beta_impul1_acc_late[:]=np.NaN
beta_impul2_acc_early=np.ones([len(subID)])
beta_impul2_acc_early[:]=np.NaN
beta_impul2_acc_late=np.ones([len(subID)])
beta_impul2_acc_late[:]=np.NaN

corr_early=np.ones([len(subID)])
corr_early[:]=np.NaN
corr_late=np.ones([len(subID)])
corr_late[:]=np.NaN
#---------load data -----------------

for i in subID:
    print('sub:',i,'...',np.round(i/len(subID)*100,2),'%')
    fname = 'Dynamic_hidden_states_exp2_' + str(i) + '.mat'
    fname_i1e1 =  np.load(datapath+'/'+str(i) + 'dec_impul1_cov_corr_early1.npz')['dec_impul1_early1_cos']
    fname_i1e2 =  np.load(datapath+'/'+str(i) + 'dec_impul1_cov_corr_early2.npz')['dec_impul1_early2_cos']
    fname_i1l1 =  np.load(datapath+'/'+str(i) + 'dec_impul1_cov_corr_late1.npz')['dec_impul1_late1_cos']
    fname_i1l2 =  np.load(datapath+'/'+str(i) + 'dec_impul1_cov_corr_late2.npz')['dec_impul1_late2_cos']
    fname_i2e1 =  np.load(datapath+'/'+str(i) + 'dec_impul2_cov_corr_early1.npz')['dec_impul2_early1_cos']
    fname_i2e2 =  np.load(datapath+'/'+str(i) + 'dec_impul2_cov_corr_early2.npz')['dec_impul2_early2_cos']
    fname_i2l1 =  np.load(datapath+'/'+str(i) + 'dec_impul2_cov_corr_late1.npz')['dec_impul2_late1_cos']
    fname_i2l2 =  np.load(datapath+'/'+str(i) + 'dec_impul2_cov_corr_late2.npz')['dec_impul2_late2_cos']
    
    
    
    fpath = os.path.join(raw_datapath,fname)
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



#--------extract behavioral results------------------
    test_early_sess1 = beh_sess_1[incl_impul1_sess_1,-2]
    test_early_sess2 = beh_sess_2[incl_impul1_sess_2,-2]
    test_late_sess1 = beh_sess_1[incl_impul2_sess_1,-1]
    test_late_sess2 = beh_sess_2[incl_impul2_sess_2,-1]
   
#--------positve area 2 sessions--------------------------------
    
    #sel_i1e1=fname_i1e1[np.logical_not(np.isnan(test_early_sess1)),:]
    #area_i1e1=np.mean(sel_i1e1[:,dec_window],1)
    #sel_i1e2=fname_i1e2[np.logical_not(np.isnan(test_early_sess2)),:]
    #area_i1e2=np.mean(sel_i1e2[:,dec_window],1)
    
    #sel_i1l1=fname_i1l1[np.logical_not(np.isnan(test_early_sess1)),:]
    #area_i111=np.mean(sel_i1l1[:,dec_window],1)
    #sel_i1l2=fname_i1l2[np.logical_not(np.isnan(test_early_sess2)),:]
    #area_i112=np.mean(sel_i1l2[:,dec_window],1)
    
    #sel_i2e1=fname_i2e1[np.logical_not(np.isnan(test_late_sess1)),:]
    #area_i2e1=np.mean(sel_i2e1[:,dec_window],1)
    #sel_i2e2=fname_i2e2[np.logical_not(np.isnan(test_late_sess2)),:]
    #area_i2e2=np.mean(sel_i2e2[:,dec_window],1)
    
    #sel_i2l1=fname_i2l1[np.logical_not(np.isnan(test_late_sess1)),:]
    #area_i211=np.mean(sel_i2l1[:,dec_window],1)
    #sel_i2l2=fname_i2l2[np.logical_not(np.isnan(test_late_sess2)),:]
    #area_i212=np.mean(sel_i2l2[:,dec_window],1)
    
    #test_early_sess1=test_early_sess1[np.logical_not(np.isnan(test_early_sess1))]
    #test_early_sess2=test_early_sess2[np.logical_not(np.isnan(test_early_sess2))]
    #test_late_sess1=test_late_sess1[np.logical_not(np.isnan(test_late_sess1))]
    #test_late_sess2=test_late_sess2[np.logical_not(np.isnan(test_late_sess2))]

    #high_impul1_acc_early[i-1]=(np.mean(test_early_sess1[area_i1e1>np.median(area_i1e1)])+np.mean(test_early_sess2[area_i1e2>np.median(area_i1e2)]))/2
    #low_impul1_acc_early[i-1]=(np.mean(test_early_sess1[area_i1e1<np.median(area_i1e1)])+np.mean(test_early_sess2[area_i1e2<np.median(area_i1e2)]))/2
    #high_impul1_acc_late[i-1]=(np.mean(test_early_sess1[area_i111>np.median(area_i111)])+np.mean(test_early_sess2[area_i112>np.median(area_i112)]))/2
    #low_impul1_acc_late[i-1]=(np.mean(test_early_sess1[area_i111<np.median(area_i111)])+np.mean(test_early_sess2[area_i112<np.median(area_i112)]))/2
    
    #high_impul2_acc_early[i-1]=(np.mean(test_late_sess1[area_i2e1>np.median(area_i2e1)])+np.mean(test_late_sess2[area_i2e2>np.median(area_i2e2)]))/2
    #low_impul2_acc_early[i-1]=(np.mean(test_late_sess1[area_i2e1<np.median(area_i2e1)])+np.mean(test_late_sess2[area_i2e2<np.median(area_i2e2)]))/2
    #high_impul2_acc_late[i-1]=(np.mean(test_late_sess1[area_i211>np.median(area_i211)])+np.mean(test_late_sess2[area_i212>np.median(area_i212)]))/2
    #low_impul2_acc_late[i-1]=(np.mean(test_late_sess1[area_i211<np.median(area_i211)])+np.mean(test_late_sess2[area_i212<np.median(area_i212)]))/2

 #--------positve area--------------------------------
    dec_impul1_early = np.concatenate((fname_i1e1,fname_i1e2))
    dec_impul1_late = np.concatenate((fname_i1l1,fname_i1l2))
    dec_impul2_early = np.concatenate((fname_i2e1,fname_i2e2))
    dec_impul2_late = np.concatenate((fname_i2l1,fname_i2l2))
    acc_early = np.concatenate((test_early_sess1,test_early_sess2)) 
    acc_late = np.concatenate((test_late_sess1,test_late_sess2))

       
    #dec_impul1_early[dec_impul1_early<0]=0
    sel_dec_impul1_early=dec_impul1_early[np.logical_not(np.isnan(acc_early)),:]
    area_impul1_early= np.trapz(sel_dec_impul1_early[:,dec_window],inttimes)#np.mean(sel_dec_impul1_early[:,dec_window],1)#np.trapz(sel_dec_impul1_early[:,dec_window],inttimes)#np.mean(sel_dec_impul1_early[:,dec_window],1)
    
    #dec_impul1_late[dec_impul1_late<0]=0
    sel_dec_impul1_late=dec_impul1_late[np.logical_not(np.isnan(acc_early)),:]
    area_impul1_late= np.trapz(sel_dec_impul1_late[:,dec_window],inttimes)#np.mean(sel_dec_impul1_late[:,dec_window],1)#np.trapz(sel_dec_impul1_late[:,dec_window],inttimes)#np.mean(sel_dec_impul1_late[:,dec_window],1)
   
    #dec_impul2_early[dec_impul2_early<0]=0
    sel_dec_impul2_early=dec_impul2_early[np.logical_not(np.isnan(acc_late)),:]
    area_impul2_early= np.trapz(sel_dec_impul2_early[:,dec_window],inttimes)#np.mean(sel_dec_impul2_early[:,dec_window],1)#np.trapz(sel_dec_impul2_early[:,dec_window],inttimes)#np.mean(sel_dec_impul2_early[:,dec_window],1)
    
    #dec_impul2_late[dec_impul2_late<0]=0
    sel_dec_impul2_late=dec_impul2_late[np.logical_not(np.isnan(acc_late)),:]
    area_impul2_late= np.trapz(sel_dec_impul2_late[:,dec_window],inttimes)#np.mean(sel_dec_impul2_late[:,dec_window],1)#np.trapz(sel_dec_impul2_late[:,dec_window],inttimes)#np.mean(sel_dec_impul2_late[:,dec_window],1)
 


#----------trial level acc------------------------------
    acc_early=acc_early[np.logical_not(np.isnan(acc_early))]
    acc_late=acc_late[np.logical_not(np.isnan(acc_late))]
    
    #clf_impul1_early = LogisticRegression(random_state=0).fit(area_impul1_early.reshape(-1, 1),acc_early)
    #beta_impul1_acc_early[i-1] = clf_impul1_early.coef_
    
    #clf_impul1_late = LogisticRegression(random_state=0).fit(area_impul1_late.reshape(-1, 1),acc_early)
    #beta_impul1_acc_late[i-1] = clf_impul1_late.coef_
    
    #clf_impul2_early = LogisticRegression(random_state=0).fit(area_impul2_early.reshape(-1, 1),acc_late)
    #beta_impul2_acc_early[i-1] = clf_impul2_early.coef_
    
    #clf_impul2_late = LogisticRegression(random_state=0).fit(area_impul2_late.reshape(-1, 1),acc_late)
    #beta_impul2_acc_late[i-1] = clf_impul2_late.coef_
    
    high_impul1_acc_early[i-1] = np.mean(acc_early[area_impul1_early>np.median(area_impul1_early)])
    low_impul1_acc_early[i-1] = np.mean(acc_early[area_impul1_early<np.median(area_impul1_early)])
    high_impul1_acc_late[i-1] = np.mean(acc_early[area_impul1_late>np.median(area_impul1_late)])
    low_impul1_acc_late[i-1] = np.mean(acc_early[area_impul1_late<np.median(area_impul1_late)])
    
    high_impul2_acc_early[i-1] = np.mean(acc_late[area_impul2_early>np.median(area_impul2_early)])
    low_impul2_acc_early[i-1] = np.mean(acc_late[area_impul2_early<np.median(area_impul2_early)])
    high_impul2_acc_late[i-1] = np.mean(acc_late[area_impul2_late>np.median(area_impul2_late)])
    low_impul2_acc_late[i-1] = np.mean(acc_late[area_impul2_late<np.median(area_impul2_late)])
    
    #high_impul1_acc_early[i-1] = np.mean(acc_early[area_impul1_early>=np.percentile(area_impul1_early,75)])
    #low_impul1_acc_early[i-1] = np.mean(acc_early[area_impul1_early<=np.percentile(area_impul1_early,25)])
    #high_impul1_acc_late[i-1] = np.mean(acc_early[area_impul1_late>=np.percentile(area_impul1_late,75)])
    #low_impul1_acc_late[i-1] = np.mean(acc_early[area_impul1_late<=np.percentile(area_impul1_late,25)])
    
    #high_impul2_acc_early[i-1] = np.mean(acc_late[area_impul2_early>=np.percentile(area_impul2_early,75)])
    #low_impul2_acc_early[i-1] = np.mean(acc_late[area_impul2_early<=np.percentile(area_impul2_early,25)])
    #high_impul2_acc_late[i-1] = np.mean(acc_late[area_impul2_late>=np.percentile(area_impul2_late,75)])
    #low_impul2_acc_late[i-1] = np.mean(acc_late[area_impul2_late<=np.percentile(area_impul2_late,25)])
    

    #comp_impul1=area_impul1_early-area_impul1_late
    #comp_impul2=area_impul2_late-area_impul2_early
    
    #--------------relationship between 2 WM representations
    corr_early[i-1], _ = pearsonr(area_impul1_early, area_impul1_late)
    corr_late[i-1], _ = pearsonr(area_impul2_early, area_impul2_late)
   

#----Acc difference between high low decoding trials---
diff_impul1_early = high_impul1_acc_early-low_impul1_acc_early
diff_impul1_late = high_impul1_acc_late-low_impul1_acc_late
diff_impul2_early = high_impul2_acc_early-low_impul2_acc_early
diff_impul2_late = high_impul2_acc_late-low_impul2_acc_late
 

#----------t-tests-----------------------------------
s_impul1_early, p_impul1_early = sp.stats.ttest_rel(high_impul1_acc_early,low_impul1_acc_early)
s_impul1_late, p_impul1_late = sp.stats.ttest_rel(high_impul1_acc_late,low_impul1_acc_late)
s_impul2_late, p_impul2_late = sp.stats.ttest_rel(high_impul2_acc_late,low_impul2_acc_late)
s_impul2_early, p_impul2_early = sp.stats.ttest_rel(high_impul2_acc_early,low_impul2_acc_early)

#s_impul1_early, p_impul1_early = sp.stats.ttest_1samp(beta_impul1_acc_early,0)
#s_impul1_late, p_impul1_late = sp.stats.ttest_1samp(beta_impul1_acc_late,0)
#s_impul2_early, p_impul2_early = sp.stats.ttest_1samp(beta_impul2_acc_early,0)
#s_impul2_late, p_impul2_late = sp.stats.ttest_1samp(beta_impul2_acc_late,0)
#s_corr_early, p_corr_early = sp.stats.ttest_1samp(corr_early,0)
#s_corr_late, p_corr_late = sp.stats.ttest_1samp(corr_late,0)


# In[306]:



print(s_impul1_early, p_impul1_early)
print(s_impul1_late, p_impul1_late)
print(s_impul2_early, p_impul2_early)
print(s_impul2_late, p_impul2_late)
print(s_corr_early, p_corr_early)
print(s_corr_late, p_corr_late)


# In[307]:




def GroupPerTest(dat,nSims):
    mdat = np.mean(dat,0)
    p = 0
    for sim in range(nSims):
        permind = np.sign(np.random.rand(len(dat))-.5);
        p = p + (np.mean(dat*permind)>=mdat);
    p = p/nSims;
    if p>0.5:
        p=1-p
    p=round(p,5)
    p=p*2
        

    return p
        
p_impul1_early_per = GroupPerTest(diff_impul1_early,50000)
p_impul1_late_per = GroupPerTest(diff_impul1_late,50000)
p_impul2_early_per = GroupPerTest(diff_impul2_early,50000)
p_impul2_late_per = GroupPerTest(diff_impul2_late,50000)

print(p_impul1_early_per)
print(p_impul1_late_per)
print(p_impul2_early_per)
print(p_impul2_late_per)


# In[290]:


diff_highlow = np.hstack((diff_impul1_early,diff_impul1_late,diff_impul2_early,diff_impul2_late))
print(diff_highlow.shape)


# In[310]:


diff_impul1 = np.hstack((diff_impul1_early,diff_impul1_late))*100
diff_impul2 = np.hstack((diff_impul2_early,diff_impul2_late))*100
WMitem = np.hstack((['Tested early']*len(subID),['Tested late']*len(subID)))
data_diff_impul1 = {'WMitem': WMitem, 'diffscores': diff_impul1}
df_diff_impul1 = pd.DataFrame(data=data_diff_impul1)
data_diff_impul2 = {'WMitem': WMitem, 'diffscores': diff_impul2}
df_diff_impul2 = pd.DataFrame(data=data_diff_impul2)

fig, ax = plt.subplots()
sns.boxplot(x="WMitem", y="diffscores", data=df_diff_impul1, palette="pastel")
sns.swarmplot(x="WMitem", y="diffscores",data=df_diff_impul1, color=".25")
ax.set_title('Impulse 1')
ax.set_xlabel('Tested items', size=13)
ax.set_ylabel('Accuracy difference', size=13)
plt.ylim([-8,8])
fig.savefig('impulse1_dec&acc.png')

fig2,ax2=plt.subplots()
sns.boxplot(x="WMitem", y="diffscores", data=df_diff_impul2, palette="pastel")
sns.swarmplot(x="WMitem", y="diffscores",data=df_diff_impul2, color=".25")
ax2.set_title('Impulse 2')
ax2.set_xlabel('Tested items', size=13)
ax2.set_ylabel('Accuracy difference', size=13)
plt.ylim([-8,8])
fig.savefig('impulse2_dec&acc.png')


# In[ ]:




