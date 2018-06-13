#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:26:56 2018

@author: florian
"""

from __future__ import print_function

import numpy as np
import interpolate 
import matplotlib.pyplot as plt
from curves import OSCCurve
from sklearn.gaussian_process import kernels
import band_transformation as bt



def plot_lc_trans(sn_list, init_bands):
    """
    Plot the light curves after transformation for a Sn using both system of filter (Jhson and gri or g'r'i')

    Parameters
    ----------
    sn_list : list contains Sn to test the transformation
    init_bands : list contains initial bands used to transform to the new system
    """
    for sn_name in sn_list:
        bands_jhonson = init_bands.split(',')
        bands_gri = "g,r,i".split(',')
        k1 = kernels.RBF(length_scale_bounds=(1e-4, 1e4))
        k2 = kernels.RBF(length_scale_bounds=(1e-4, 1e4))
        # k2 = kernels.WhiteKernel()
        # k3 = kernels.ConstantKernel(constant_value_bounds='fixed')
        k3 = kernels.WhiteKernel()
    
        m = np.array([[1, 0, 0],
                      [0.5, 1, 0],
                      [0.5, 0.5, 1]])
        m_bounds = (np.array([[1e-4, 0, 0],
                              [-1e2, -1e3, 0],
                              [-1e2, -1e2, -1e3]]),
                    np.array([[1e4, 0, 0],
                              [1e2, 1e3, 0],
                              [1e2, 1e2, 1e3]]))
#        m = np.zeros((2,2)); m[0, 0] = 1
#        m = np.array([[1, 0],
#                      [0.5, 1]])
#        m_bounds = (np.array([[1e-4, 0],
#                              [-1e2, -1e3]]),
#                    np.array([[1e4, 0],
#                              [1e2, 1e3]]))
#        
        colors = {'B':'b','V':'y' ,'R':'r', 'I':'brown',"g'": 'g', "r'": 'r', "i'": 'brown','g':'g','r':'r','i': 'brown'}
        
        curve = OSCCurve.from_name(sn_name, bands=bands_jhonson).binned(bin_width=1, discrete_time=True).filtered(sort='filtered')
        x_ = np.linspace(curve.X[:,1].min(), curve.X[:,1].max(), 101)
        interpolator = interpolate.GPInterpolator(
            curve, (k1, k2, k3), m, m_bounds,
            optimize_method=None,  #'trust-constr',
            n_restarts_optimizer=0,
            random_state=0,
            add_err=10,
            raise_on_bounds=False
        )
        
        msd = interpolator(x_)
        
        for i, band in enumerate(bands_jhonson):
            plt.subplot(2, 2, i+1)
            blc = curve[band]
            plt.errorbar(blc['x'], blc['y'], blc['err'], marker='x', ls='', color=colors[band])
            plt.plot(msd.odict[band].x, msd.odict[band].y, color=colors[band], label=band)
            plt.grid()
            plt.legend()
            pdffile = '../plot/lc_BVRI_'+sn_name+'.pdf'
            plt.savefig(pdffile, bbox_inches='tight') 
        plt.show()
#        if 'V' in bands_jhonson:
#            new_msd = bt.VR_to_gri(msd)
#        else:
#            new_msd = bt.BR_to_gri(msd)
          
        new_msd = bt.BRI_to_gri(msd)
        curve2 = OSCCurve.from_name(sn_name, bands=bands_gri).binned(bin_width=1, discrete_time=True).filtered(sort='filtered')
        print(new_msd.odict.keys())
        for i, band in enumerate(bands_gri):
            plt.subplot(2, 2, i+1)
            blc = curve2[band]
            plt.errorbar(blc['x'], blc['y'], blc['err'], marker='x', ls='', color=colors[band])
            plt.plot(new_msd.odict[band].x, new_msd.odict[band].y, color=colors[band], label=band)
            plt.grid()
            plt.title(sn_name)
            pdffile = '../plot/lc_BVRI_gri_'+sn_name+'.pdf'
            plt.savefig(pdffile, bbox_inches='tight')     
            plt.legend()
        plt.show()

    
       
if __name__=='__main__':        
        
        
        
#    sn_list = ['SN2013fs', 'LSQ13zm', 'SN2013ej','SN2004dt', 'SN2009dc', 'SN2009dc'] #BR gri
#    sn_list = ['SN2013gh', 'ASASSN-13ax', 'SN2004gv','SN2016eay','iPTF13bvn','SN2004eo','SN2005el','SN2006hb'] #VR gri
#    sn_list = ['SN2011bm' ,'SN2011hs' ,'SN2010gx' ,'SN2009ib' ,'SN2007fr' ,'SN2006fo' ,'SN2008ax' ,'SN2009N' ,'SN2005hk' ] #BR g'r'i'
#    sn_list = ['SN2015bn','SN2011bm','SN2011hs','SN2013bh','SN2009ib','SN2006fo','SN2008ax','SN2009N','SN2005hk'] #VR g'r'i'
    
    sn_list =['SN2007bc','ASASSN-13ax','SN2013ej','SN2005am','SN2005M','SN2013fs'] #BRI gri
    init_bands = 'B,R,I'
    plot_lc_trans(sn_list, init_bands)