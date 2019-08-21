# SNAD - SuperNova Anomaly Detection


[SNAD](https://snad.space/) is a project devoted to the anomaly detection problem in [the Open Supernova Catalog](https://sne.space/). The results of this work can be found in [Pruzhinskaya et al., 2019](https://arxiv.org/abs/1905.11516). Here, we present all the code and the data from the SNAD analysis.

---------------------------------------------------------------------------------------------------

# Overview 

`data/`


Lists of supernovae that pass our selection criteria, i.e. have 3 observations in each passband. These objects are used 
for interpolation/extrapolation. 3-day bin width is applied: 

  * *min3obs_B,R,I.csv*
  
  * *min3obs_g,r,i.csv*
    
  * *min3obs_g_pr,r_pr,i_pr.csv*


Files that contain photometry in the range of [-20:100] days relative to the maximum in *r/r'* band of the SNe
extrapolated light curves in *gri*, *g'r'i'*, and *BRI* (transformed to *gri*) passbands. The results of the extrapolation were not checked by eye:

  * *extrapol_-20.0_100.0_B,R,I_uncut.csv*
  
  * *extrapol_-20.0_100.0_g,r,i_uncut.csv*
  
  * *extrapol_-20.0_100.0_g_pr,r_pr,i_pr.csv*


Files that contain photometry in the range of [-20:100] days relative to the maximum in *r/r'* band of the SNe
extrapolated light curves in *gri*, *g'r'i'*, and *BRI* (transformed to *gri*) passbands. 
The results of the extrapolation were checked by eye. These files are mainly used for the ML analysis:

  * *extrapol_-20.0_100.0_B,R,I.csv*

  * *extrapol_-20.0_100.0_g,r,i.csv*

  * *extrapol_-20.0_100.0_g_pr,r_pr,i_pr.csv*


`tsne/` contains files with dimensionality-reduced data sets corresponding to 2 to 9 t-SNE features.


`isolation_forests/` contains results of isolation forest algorithm run on 10 datasets:

  * *weirdSN_isoforest_GPfit.dat*

    data set of 364 photometric characteristics (121×3 normalized fluxes, the LC flux maximum)

  * *weirdSN_isoforest_GPparam.dat*

    data set of 10 parameters of the Gaussian process (9 fitted parameters of the kernel, the log-likelihood of the fit)

  * *weirdSN_isoforest_tSNE_\*.dat* 

    8 data sets obtained by reducing 374 features to 2-9 t-SNE dimensions


---------------------------------------------------------------------------------------------------

`fig/` contains plots of the supernova light curves in *gri*, *g'r'i'*, and *BRI* passbands together with their Gaussian processes approximation. 


---------------------------------------------------------------------------------------------------

# Feedback

You can send feedback via [e-mail](mailto:malanchev@physics.msu.ru) or via [Github](https://github.com/sai-msu/snad/issues). 
