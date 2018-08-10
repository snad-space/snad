Isoforest
=========

This files came from `isolation_forest_GP.py` and `isolation_forest_tSNE.py` scripts.

`weirdSN_isoforest_tSNE_X.dat`
------------------------------

`weirdSN_isoforest_tSNE_X.dat` is obtained from `data/tsne.X.csv` by running `isolation_forest_tSNE.py`.
`weirdSN_isoforest_tSNE_X.dat` contains outliers from `data/tsne.X.csv` found by isoforest.

`weirdSN_isoforest_tSNE_common_X.dat`
-------------------------------------

`weirdSN_isoforest_tSNE_common_X.dat` contains outliers which appeared in `X` different `weirdSN_isoforest_tSNE_Y.dat` files.

For instance if you have `weirdSN_isoforest_tSNE_1.dat` with two lines `A` and `B`, and `weirdSN_isoforest_tSNE_2.dat` with `A`, `B` and `C`. Then corresponding `weirdSN_isoforest_tSNE_common_1.dat` will contain `C` and `weirdSN_isoforest_tSNE_common_2.dat` will contain `A` and `B`.

`weirdSN_isoforest_GPfit.dat`
-----------------------------
All `weirdSN_isoforest_GPfit.dat`, `weirdSN_isoforest_GPparam.dat`, `weirdSN_isoforest_GPparam_GPfit.dat` are done from `data/extrapol_-20.0_100.0_g_pr,r_pr,i_pr.csv` and `data/extrapol_-20.0_100.0_g,r,i.csv` by `isolation_forest_GP.py`.

`weirdSN_isoforest_GPfit.dat` have outliers produced by applying isoforest to SN light curve only + normalization.
Mind that we do per-row light curve normalization and save maximum magnitude as separate extra column.

`weirdSN_isoforest_GPparam.dat`
-------------------------------
This is the same as above but produced by applying isoforest to interpolation parameters: `theta_0`...`theta_8` and inerpolation `likelyhood`.

`weirdSN_isoforest_GPparam_GPfit.dat`
-------------------------------------
This is intersection between `weirdSN_isoforest_GPfit.dat` and `weirdSN_isoforest_GPparam.dat`.
