import pylab as plt
import numpy as np
import os
import seaborn

from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
from itertools import groupby

########################################################
###        Analysis with tSNE parameters            ####
########################################################

# set random seed
rng = np.random.RandomState(42)

# rather to generate individual plots for detected anomalies
# False, True or 'summary', if 'summary' - generate plots only for the objs common in both analysis: full fit + LC parameters
plot = True

# number of neighbors
nn = 1000

# number of parameters to be tested
par_num = np.arange(2,10)

names = []
indx_all = []
skip = []

# create directory for plots
if not os.path.isdir('anomalies') and plot == True:
    os.makedirs('anomalies')

data_all = {}

for k in par_num:
    # path to lc fitted data files
    path_to_data = '../../data/tsne/tsne_fake_' + str(k) + '.csv'

    print('\n ****    t-SNE nalysis with ', str(k),' parameters    **** \n')
    print('Reading data ...')

    # read data for gri filters
    op1 = open(path_to_data, 'r')
    lin1 = op1.readlines()
    op1.close()

    data_str1 = [elem.split(',') for elem in lin1]
    data = [[float(item) for item in line[2:]] for line in data_str1[1:]]

    #  gather names in the correct order
    names = [line[1] for line in data_str1[1:]]

    # remove problematic objects
    for item in skip:
        data.remove(data[item])

    data = np.array(data)    
    data_all[k] = data

    print('   ... done!')

    ##### fit and predict from the Local Outlier Factor model ######

    print('Fit local outiler factor model ...') 

    clf = LocalOutlierFactor(n_neighbors=nn, contamination=0.005)
    pred = clf.fit_predict(data)

    # index of outlier objects according to the isolation forest algorithm
    indx_iso = [i for i in range(pred.shape[0]) if pred[i] == -1]
                
    print(   '... done!')
    print('Number of outliers from isoforest + tSNE_', str(k),': ', str(len(indx_iso)))

    # save names of SNe considered anomalies
    output_file1 = 'weirdSN_isoforest_tSNE_' + str(k) + 'withFake.dat'

    op3 = open(output_file1, 'w')
    for i in range(len(indx_iso)):
        op3.write(names[indx_iso[i]] + '\n')
    op3.close()

    print('Anomalies list saved in file: ', output_file1)

    if plot == True:
        normal = np.array([True if i not in indx_iso else False for i in range(data.shape[0])])
        weird = np.array([True if i in indx_iso else False for i in range(data.shape[0])])

        plt.figure(figsize=(20,20))
        for l1 in range(data.shape[1]):            
            for l2 in range(data.shape[1]):
                plt.subplot(data.shape[1], data.shape[1], l1*data.shape[1] + l2 + 1)
                if l1 == l2:
                    plt.hist(data[normal][:,l1], facecolor='gray')
                    plt.hist(data[weird][:,l1], facecolor='blue')
                    plt.xlabel('p' + str(l1))
                else:
                    plt.scatter(data[normal][:,l1], data[normal][:,l2], label='normal', marker='x', color='black')
                    plt.scatter(data[weird][:,0], data[weird][:,1], label='weird', marker='o', color='red')
                    plt.xlabel('p' + str(l1))
                    plt.ylabel('p' + str(l2))

        plt.tight_layout()
        plt.savefig('anomalies/tSNE_' + str(k) + '_iso_withFake.png')
        plt.close('all')

    indx_all += indx_iso

indx_counter = Counter(indx_all)
indx_groups = groupby(indx_counter.most_common(), lambda x: x[1])

for (c,it) in indx_groups:
    itl = list(it)
    print('Anomalies common for {} cases ({} total):'.format(c, len(itl)))
    for x in itl:
        print('    {}'.format(names[x[0]]))

#print('\n There are ', str(len(indx_all)), ' common anomalies.')
#print([names[item] for item in indx_all])

# save names of SNe considered anomalies in all configurations
#output_file3 = 'weirdSN_isoforest_tSNE_all.dat'

#op5 = open(output_file3, 'w')
#for i in range(len(indx_all)):
#    op5.write(names[indx_all[i]] + '\n')
#op5.close()

#print('\n Names of anomalies common to all tSNE analysis stored at ', output_file3)


### plot
if plot == 'summary' or plot == True and len(indx_all) > 0:

    for k in par_num:
        print('Plotting summary for ' + str(k) + ' parameters.')

        normal_all = np.array([True if i not in indx_all else False for i in range(data_all[k].shape[0])])
        weird_all = np.array([True if i in indx_all else False for i in range(data_all[k].shape[0])])

        plt.figure(figsize=(10,10))
        for l1 in range(data_all[k].shape[1]):            
            for l2 in range(data_all[k].shape[1]):
                plt.subplot(data_all[k].shape[1], data_all[k].shape[1], l1*data_all[k].shape[1] + l2 + 1)
                if l1 == l2:
                    plt.hist(data_all[k][normal_all][:,l1], facecolor='gray')
                    plt.hist(data_all[k][weird_all][:,l1], facecolor='blue')
                    plt.xlabel('p' + str(l1))
                else:
                    plt.scatter(data_all[k][normal_all][:,l1], data_all[k][normal_all][:,l2], label='normal', marker='x', color='black')
                    plt.scatter(data_all[k][weird_all][:,l1], data_all[k][weird_all][:,l2], label='weird', marker='o', color='red')
                    plt.xlabel('p' + str(l1))
                    plt.ylabel('p' + str(l2))

        plt.tight_layout()
        plt.savefig('anomalies/tSNE_' + str(k) + '_iso_withFake_ALL.png')
        plt.close('all')


