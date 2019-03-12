#!/usr/bin/env python

import pylab as plt
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import IsolationForest

########################################################
###  Analysis with entire interpolated light curve  ####
########################################################

# path to lc fitted data files
inp_files = [
    ("g'r'i", '../../data/extrapol_-20.0_100.0_g_pr,r_pr,i_pr.csv'),
    ("gri", '../../data/extrapol_-20.0_100.0_g,r,i.csv'),
    ("BRI", '../../data/extrapol_-20.0_100.0_B,R,I.csv')
]

######## user choices
# set random seed
rng = np.random.RandomState(43)

# rather to generate individual plots for detected anomalies
# False, True or 'summary', if 'summary' - generate plots only for the objs common in both analysis: full fit + LC parameters
plot = 'summary'

print('\n ****    Full interpolate LC analysis    **** \n')
print('Reading data ...')

dfs = [pd.read_csv(inp) for (_bandname, inp) in inp_files]
lcs = [np.array(df.loc[:,'g-20':'i+100']) for df in dfs]
lc_data = np.concatenate(lcs, axis=0)
lc_data_norm = np.amax(lc_data, axis=1).reshape(-1,1)
lc_norm = np.hstack([lc_data / lc_data_norm, -2.5*np.log10(lc_data_norm)])

names = [np.array(["{} ({})".format(x, bandname) for x in df.loc[:,'Name']]) for ((bandname, _filename), df) in zip(inp_files, dfs)]
lc_names = np.concatenate(names, axis=0)

print('   ... done!')

##### fit and predict from the isolation forest model ######

print('Fit isolation forest model ...') 

clf = IsolationForest(max_samples=lc_norm.shape[0], random_state=rng, contamination=0.01, n_estimators=lc_norm.shape[1])
clf.fit(lc_norm)
lc_pred = clf.predict(lc_norm)
lc_score = clf.score_samples(lc_norm)

# index of outlier objects according to the isolation forest algorithm
indx_iso_GP = sorted([i for i in range(lc_pred.shape[0]) if lc_pred[i] == -1], key=lambda x: lc_score[x])

print(   '... done!')
print('Number of outliers from isoforest + GP fit: ', str(len(indx_iso_GP)))

# save names of SNe considered anomalies
output_file1 = 'weirdSN_isoforest_GPfit.dat'

op3 = open(output_file1, 'w')
for i in range(len(indx_iso_GP)):
    op3.write(lc_names[indx_iso_GP[i]] + ", " + str(lc_score[indx_iso_GP[i]]) + '\n')
op3.close()

print('Anomalies list saved in file: ', output_file1)

if plot == True:
    for k in range(len(indx_isoGP)):
        plt.figure()
        plt.plot(np.arange(150), data_norm[indx_iso_GP[k]][3:153], label='g', lw=2)
        plt.plot(np.arange(150), data_norm[indx_iso_GP[k]][153:303], label='r', lw=2)
        plt.plot(np.arange(150), data_norm[indx_iso_GP[k]][303:453], label='i', lw=2)
        plt.legend()
        plt.show()

        plt.close('all')


########################################################
###       Analysis with GP kernel parameters        ####
########################################################

print('\n ****    Analysis based on GP parameters    **** \n')
print('Reading data ...')

thetas = [np.array(df.loc[:,'log_likehood':'theta_8']) for df in dfs]
theta_data = np.concatenate(thetas, axis=0)
theta_data_norm = np.amax(theta_data, axis=0) - np.amin(theta_data, axis=0)
param_norm = theta_data / theta_data_norm

names = [np.array(["{} ({})".format(x, bandname) for x in df.loc[:,'Name']]) for ((bandname, _filename), df) in zip(inp_files, dfs)]
param_names = np.concatenate(names, axis=0)

print('   ... done!')

##### fit and predict from the isolation forest model ######

print('Fit isolation forest model ...') 

clf_param = IsolationForest(max_samples=param_norm.shape[0], random_state=rng, contamination=0.01, n_estimators=param_norm.shape[1])
clf_param.fit(param_norm)
param_pred = clf_param.predict(param_norm)
param_score = clf_param.score_samples(param_norm)

# index of outlier objects
indx_iso_param = sorted([i for i in range(param_pred.shape[0]) if param_pred[i] == -1], key=lambda x: param_score[x])

print('   ...done!')
print('\n Number of outliers from isoforest + GP parametres: ', str(len(indx_iso_param)))

# save names of SNe considered anomalies
output_file2 = 'weirdSN_isoforest_GPparam.dat'

op4 = open(output_file2, 'w')
for i in range(len(indx_iso_param)):
    op4.write(param_names[indx_iso_param[i]] + ", " + str(param_score[indx_iso_param[i]]) + '\n')
op4.close()

print('\n Anomalies list saved in file: ', output_file2)

if plot == True:
    for k in range(len(indx_param_iso)):
        plt.figure()
        plt.plot(np.arange(150), data_norm[indx_param_iso[k]][3:153], label='g', lw=2)
        plt.plot(np.arange(150), data_norm[indx_param_iso[k]][153:303], label='r', lw=2)
        plt.plot(np.arange(150), data_norm[indx_param_iso[k]][303:453], label='i', lw=2)
        plt.legend()
        plt.show()

        plt.close('all')


# get list of common outliers
indx_out = [item for item in indx_iso_param if item in indx_iso_GP]

print('\n Found ' + str(len(indx_out)) + ' anomalies common to both analysis!')

# save names of SNe considered anomalies
output_file3 = 'weirdSN_isoforest_GPparam_GPfit.dat'

op5 = open(output_file3, 'w')
for i in range(len(indx_out)):
    op5.write(param_names[indx_out[i]] + '\n')
op5.close()

print('\n Names of anomalies common to GP fit and GP parameter analysis stored at ', output_file3)


### plot

if plot == 'summary' or plot:
    if not os.path.isdir('anomalies'):
        os.makedirs('anomalies')

    for k in range(len(indx_out)):
        plt.figure()
        plt.scatter(np.arange(121), lc_norm[indx_out[k], 0:121], s=2.0, label='g')
        plt.scatter(np.arange(121), lc_norm[indx_out[k], 122:243], s=2.0, label='r')
        plt.scatter(np.arange(121), lc_norm[indx_out[k], 243:364], s=2.0, label='i')
        plt.legend()
        plt.savefig('anomalies/' + lc_names[indx_out[k]] + '_isoforest.png')

        plt.close('all')

