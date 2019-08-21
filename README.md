# SNAD - SuperNova Anomaly Detection

---------------------------------------------------------------------------------------------------

[SNAD](https://snad.space/) is a project devoted to the anomaly detection in [the Open Supernova Catalog](https://sne.space/). The results of this work can be found in [Pruzhinskaya et al., 2019](https://arxiv.org/abs/1905.11516). Here, we present all the code and the data from the SNAD analysis.

---------------------------------------------------------------------------------------------------

# data/ 

*min3obs_B,R,I.csv*
  
*min3obs_g,r,i.csv*
    
*min3obs_g_pr,r_pr,i_pr.csv*

the lists of supernovae that pass our selection criteria, i.e. have 3 observations in each passband. These objects are used 
for interpolation/extrapolation. 3-day bin width is applied.

-----------------

*extrapol_-20.0_100.0_B,R,I_uncut.csv*
  
*extrapol_-20.0_100.0_g,r,i_uncut.csv*
  
*extrapol_-20.0_100.0_g_pr,r_pr,i_pr.csv*

These files contain photometry in the range of [-20:100] days relative to the maximum in r/r' band of the SNe
extrapolated light curves in gri, g'r'i', and BRI (transformed to gri) passbands. The results of the extrapolation were not checked by eye.

-----------------

*extrapol_-20.0_100.0_B,R,I.csv*

*extrapol_-20.0_100.0_g,r,i.csv*

*extrapol_-20.0_100.0_g_pr,r_pr,i_pr.csv*

These files contain photometry in the range of [-20:100] days relative to the maximum in r/r' band of the SNe
extrapolated light curves in gri, g'r'i', and BRI (transformed to gri) passbands. 
The results of the extrapolation were checked by eye. These files are mainly used for the ML analysis.


# fig/

Contains the plots with found anomalies.

# Feedback

You can send feedback via [e-mail](mailto:) or via [Github](https://github.com/sai-msu/snad/issues). 
