import pylab as plt
import numpy as np
import os

from sklearn.neighbors import LocalOutlierFactor

########################################################
###  Analysis with entire interpolated light curve  ####
########################################################

# path to lc fitted data files
path_to_data = '../../data/gri_renorm.csv'
path_to_data_pr = '../../data/gri_pr_renorm.csv'

######## user choices
# set random seed
rng = np.random.RandomState(43)

# rather to generate individual plots for detected anomalies
# False, True or 'summary', if 'summary' - generate plots only for the objs common in both analysis: full fit + LC parameters
plot = 'summary'

# contamination level
cont = 0.01

# number of neighbors
nn = 20

print('\n ****    Full interpolate LC analysis    **** \n')
print('Reading data ...')

# read data for gri filters
op1 = open(path_to_data, 'r')
lin1 = op1.readlines()
op1.close()

data_str1 = [elem.split(',') for elem in lin1]
data_lc1 = [[float(item) for item in line[9:]] for line in data_str1[1:]]

# read data for g'r'i' filters
op1b = open(path_to_data_pr, 'r')
lin1b = op1b.readlines()
op1b.close()

data_str2 = [elem.split(',') for elem in lin1b]
data_lc2 = [[float(item) for item in line[9:]] for line in data_str2[1:]]

# join lc and gather names in the correct order
lc_norm = np.array(data_lc1 + data_lc2)
lc_names = [line[0] for line in data_str1[1:]] + [line[0] for line in data_str2[1:]]

print('   ... done!')

##### fit and predict from the isolation forest model ######

print('Fit Local Outlier Factor model ...') 

clf = LocalOutlierFactor(n_neighbors=nn, contamination=cont)
lc_pred = clf.fit_predict(lc_norm)

# index of outlier objects according to the Local Outlier Factor algorithm
indx_lof_GP = [i for i in range(lc_pred.shape[0]) if lc_pred[i] == -1]

print(   '... done!')
print('Number of outliers from Local Outlier Factor + GP fit: ', str(len(indx_lof_GP)))

# save names of SNe considered anomalies
output_file1 = 'weirdSN_LocalOutlierFactor_GPfit.dat'

op3 = open(output_file1, 'w')
for i in range(len(indx_lof_GP)):
    op3.write(lc_names[indx_lof_GP[i]] + '\n')
op3.close()

print('Anomalies list saved in file: ', output_file1)

if plot == True:
    for k in range(len(indx_lofGP)):
        plt.figure()
        plt.plot(np.arange(150), data_norm[indx_lof_GP[k]][3:153], label='g', lw=2)
        plt.plot(np.arange(150), data_norm[indx_lof_GP[k]][153:303], label='r', lw=2)
        plt.plot(np.arange(150), data_norm[indx_lof_GP[k]][303:453], label='i', lw=2)
        plt.legend()
        plt.show()

        plt.close('all')


########################################################
###       Analysis with GP kernel parameters        ####
########################################################

print('\n ****    Analysis based on GP parameters    **** \n')
print('Reading data ...')

# path to data
path_to_param = '../../theta.csv'
path_to_param2 = '../../theta_pr.csv'

# read data for gri filters
op2 = open(path_to_param, 'r')
lin2 = op2.readlines()
op2.close()

param_str1 = [elem.split(',') for elem in lin2]
param1 = [[float(line[i]) for i in range(2, len(line))] for line in param_str1[1:]]

# read data for g'r'i' filters
op3 = open(path_to_param2, 'r')
lin3 = op3.readlines()
op3.close()

param_str2 = [elem.split(',') for elem in lin3]
param2 = [[float(line[i]) for i in range(2, len(line))] for line in param_str2[1:]]

print('   ... done!')
print('Normalizing light curves ...')

# join lc and gather names in the correct order
param = np.array(param1 + param2)
param_names = [line[1] for line in param_str1[1:]] + [line[1] for line in param_str2[1:]]

# check if order of names is the same as in the light curve files
check_names = [lc_names[i] == param_names[i] for i in range(param.shape[0])]
if False in check_names:
    raise NameError('List of names in parameter files does not correspond to light curve files!')

# normalize parameters
param_norm = np.array([[param[i][j]/(max(param[:,j]) - min(param[:,j])) for j in np.arange(param.shape[1])] for i in np.arange(param.shape[0])])

print('   ... done!')

##### fit and predict from the loflation forest model ######

print('Fit Local Outlier Factor model ...') 

clf_param = LocalOutlierFactor(n_neighbors=nn, contamination=cont)
param_pred = clf_param.fit_predict(param_norm)

# index of outlier objects
indx_lof_param = [i for i in range(param_pred.shape[0]) if param_pred[i] == -1]

print('   ...done!')
print('\n Number of outliers from Local Outlier Factor + GP parametres: ', str(len(indx_lof_param)))

# save names of SNe considered anomalies
output_file2 = 'weirdSN_LocalOutlierFactor_GPparam.dat'

op4 = open(output_file2, 'w')
for i in range(len(indx_lof_param)):
    op4.write(param_names[indx_lof_param[i]] + '\n')
op4.close()

print('\n Anomalies list saved in file: ', output_file2)

if plot == True:
    for k in range(len(indx_param_lof)):
        plt.figure()
        plt.plot(np.arange(150), data_norm[indx_param_lof[k]][3:153], label='g', lw=2)
        plt.plot(np.arange(150), data_norm[indx_param_lof[k]][153:303], label='r', lw=2)
        plt.plot(np.arange(150), data_norm[indx_param_lof[k]][303:453], label='i', lw=2)
        plt.legend()
        plt.show()

        plt.close('all')


# get list of common outliers
indx_out = [item for item in indx_lof_param if item in indx_lof_GP]

print('\n Found ' + str(len(indx_out)) + ' anomalies common to both analysis!')

# save names of SNe considered anomalies
output_file3 = 'weirdSN_LocalOutlierFactor_GPparam_GPfit.dat'

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
        plt.scatter(np.arange(150), lc_norm[indx_out[k]][3:153], s=2.0, label='g')
        plt.scatter(np.arange(150), lc_norm[indx_out[k]][153:303], s=2.0, label='r')
        plt.scatter(np.arange(150), lc_norm[indx_out[k]][303:453], s=2.0, label='i')
        plt.legend()
        plt.savefig('anomalies/' + lc_names[indx_out[k]] + '_LocalOutlierFactor.png')

        plt.close('all')

