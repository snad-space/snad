#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

_, output, dim = sys.argv

if output is None:
	raise ValueError()

if dim is None:
	dim = 9

dim = int(dim)

def do_metric(X1, X2):
	mask = np.invert((X1 == 0.0) | (X2 == 0.0))
	z = X1[mask] - X2[mask]
	return np.inner(z,z)

inp_files = [
	('../data/gri_pr.csv', '../data/theta_pr.csv'),
	('../data/gri.csv', '../data/theta.csv'),
]

dfs = [(pd.read_csv(lc), pd.read_csv(theta)) for lc, theta in inp_files]

df_lc, df_theta = zip(*dfs)
df_lc_join = pd.concat(df_lc, axis=0)
df_theta_join = pd.concat(df_theta, axis=0)

lc_data = np.array(df_lc_join.loc[:,'g-050':])
lc_data_norm = np.amax(lc_data, axis=1).reshape(-1,1)
lc_data_normed = np.hstack([lc_data / lc_data_norm, lc_data_norm])

theta_data = np.array(df_theta_join.iloc[:,2:])

data = np.hstack([lc_data_normed, theta_data])
data_norm = np.amax(data, axis=0)
data = data / data_norm

n_features = dim
method = 'exact'

t = TSNE(n_components=n_features, method=method, metric=do_metric)
new_data = t.fit_transform(data)

sn_name = df_lc_join[['SN']]

new_df = pd.concat([sn_name.reset_index(drop=True), pd.DataFrame(new_data)], axis=1)
new_df.to_csv(output)
