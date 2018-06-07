#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

_, inp, output = sys.argv

if output is None:
	raise ValueError()

df_lc = pd.read_csv(inp)
lc_data = np.array(df_lc.loc[:,'g-050':])
lc_data_norm = np.amax(lc_data, axis=1).reshape(-1,1)
lc_data_normed = np.hstack([lc_data / lc_data_norm, lc_data_norm])
meta_data = df_lc.iloc[:,0:9]
new_df = pd.concat([meta_data.reset_index(drop=True), pd.DataFrame(lc_data_normed)], axis=1)
new_df.columns = list(df_lc.columns) + ["LC_norm",]
new_df.to_csv(output)
