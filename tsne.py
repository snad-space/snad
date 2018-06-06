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

df_gri = pd.read_csv('gri_pr.csv')
df_theta = pd.read_csv('theta_pr.csv')

df = pd.concat([df_gri.reset_index(drop=True),df_theta.iloc[:,2:]], axis=1)

data = df.loc[:,'g-050':]
norm = np.amax(data, axis=0)

data = data / norm

n_features = dim
method = 'exact'

t = TSNE(n_components=n_features, method=method, metric=do_metric)
new_data = t.fit_transform(data)

sn_name = df[['SN']]

new_df = pd.concat([sn_name.reset_index(drop=True), pd.DataFrame(new_data)], axis=1)
new_df.to_csv(output)
