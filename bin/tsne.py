#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE



def do_metric(X1, X2):
	z = X1 - X2
	return np.inner(z,z)

_, output, dim = sys.argv

if dim is None:
	dim = 9

dim = int(dim)

inp_files = [
	("g'r'i'", '../data/extrapol_-20.0_100.0_g_pr,r_pr,i_pr.csv'),
	("gri", '../data/extrapol_-20.0_100.0_g,r,i.csv'),
	("BRI", '../data/extrapol_-20.0_100.0_B,R,I.csv')
]

dfs = [pd.read_csv(inp) for (_name, inp) in inp_files]
lcs = [np.array(df.loc[:,'g-20':'i+100']) for df in dfs]
lc_data = np.concatenate(lcs, axis=0)
lc_data_norm = np.amax(lc_data, axis=1).reshape(-1,1)
lc_data_normed = np.hstack([lc_data / lc_data_norm, lc_data_norm])

thetas = [np.array(df.loc[:,'log_likehood':'theta_8']) for df in dfs]
theta_data = np.concatenate(thetas, axis=0)
theta_data_norm = np.amax(theta_data, axis=0)
theta_data_normed = theta_data / theta_data_norm

names = [np.array(["{} ({})".format(x, bandname) for x in df.loc[:,'Name']]) for ((bandname, _filename), df) in zip(inp_files, dfs)]
name_data = np.concatenate(names, axis=0).reshape(-1,1)

data = np.hstack([lc_data_normed, theta_data_normed])

n_features = dim
method = 'exact'

t = TSNE(n_components=n_features, method=method, metric=do_metric, random_state=42)
new_data = t.fit_transform(data)

print(name_data.shape)
print(new_data.shape)

out_data = np.hstack([name_data, new_data])

out_df = pd.DataFrame(out_data)
out_df.to_csv(output, index=False)
