#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################################################################

import os.path
import glob
import matplotlib.pyplot as plt
import pylab as plb
import matplotlib as mpl
import pyart
import numpy as np
import scipy as sp
import numpy.ma as ma
import re
import pickle

from pylab import *
from scipy import ndimage

#############################################################################################################################
## FUNCTIONS ################################################################################################################
#############################################################################################################################

def dummy_cols(data, kernel, val='nan'):
    
    # Image boundaries are wrapped for calculation of convolution
    # NA values (zeroes/nan) need to be added in range
    
    c = (np.asarray(kernel.shape)-1)/2
    add_cols = ceil(c[1])
    dummy_cols = np.zeros((data.shape[0], add_cols.astype(int)))
    
    if val=='nan':
        dummy_cols[:] = np.NAN
    else:
        dummy_cols[:] = val
        
    # Add dummy columns
    data_out = np.hstack((data, dummy_cols))
    
    return add_cols, data_out


#############################################################################################################################

def local_valid(mask, kernel=np.ones((3,3))):
    
    ## Calculate number of local neighbours with valid value ##
    
    # Modified mask (NA values addition)
    mask_tmp = (~mask).astype(int)
    ncols, mask_tmp = dummy_cols(mask_tmp, kernel, val=0)
    
    # Convolve with kernel to calculate number of valid neighbours
    valid_tmp = ndimage.convolve(mask_tmp, kernel, mode='wrap')
    
    # Remove added values
    valid = valid_tmp[:, : int(valid_tmp.shape[1] - ncols)]
    
    return valid.astype(int)

#############################################################################################################################

def primary_vel(dim, Nprf, prf_flag=None, prf_odd=None):
    
    ## Construct array with the dual-PRF factor corresponding to each gate ##
    # Flag 0 indicates low PRF and flag 1 indicates high PRF
    
    flag_vec = np.ones(dim[0])
    
    if (prf_flag is None) & (prf_odd is not None) :
        if prf_odd==0:
            flag_vec[::2] = 0
        elif prf_odd==1:
            flag_vec[1::2] = 0
    else:
        flag_vec = prf_flag
        
    flag_vec = flag_vec - 1
    flag_vec[flag_vec<0] = 1
        
    flag_arr = np.transpose(np.tile(flag_vec, (dim[1], 1)))
    Nprf_arr = flag_arr + Nprf
    
    return Nprf_arr

#############################################################################################################################

def ref_val(data, mask, kernel, method='mean'):
    
    dummy_val=0
    Nval_arr = np.ones(mask.shape)
    
    if method=='mean':
        Nval_arr = local_valid(mask, kernel=kernel)
        dummy_data = data*(~mask).astype(int)
        
    if method=='median':
        dummy_val='nan'
        dummy_data = np.where(np.logical_not(mask), data, np.nan)
        

    ncols, data_conv = dummy_cols(dummy_data, kernel, val=dummy_val)
    
    if method=='mean':
        conv_arr = ndimage.convolve(data_conv, weights=kernel, mode='wrap')
        
    if method=='median':
        conv_arr = ndimage.generic_filter(data_conv, np.nanmedian, footprint=kernel, mode='wrap')
    
    # Remove added columns and divide by weight
    conv_arr = conv_arr[:, : int(conv_arr.shape[1] - ncols)]
    ref_arr = conv_arr/Nval_arr
    
    return ref_arr

#############################################################################################################################

def outlier_detector_cmean(np_ma, Vny, Nprf_arr, Nmin=2):
    
    data = np_ma.data
    mask = np_ma.mask
    
    f_arr = np.ones(Nprf_arr.shape)
    f_arr[np.where(Nprf_arr==np.min(Nprf_arr))] = -1
    Vny_arr = Vny/Nprf_arr
    
    kH, kL = np.zeros((5,5)), np.zeros((5,5))
    kH[1::2] = 1
    kL[::2] = 1
     
    # Array with the number of valid neighbours at each point
    Nval_arr_H = local_valid(mask, kernel=kH)
    Nval_arr_L = local_valid(mask, kernel=kL)
    
    # Convert to angles and calculate trigonometric variables
    ang_ma = (np_ma*pi/Vny)
    cos_ma = ma.cos(ang_ma*Nprf_arr)
    sin_ma = ma.sin(ang_ma*Nprf_arr)
    
    # Average trigonometric variables in local neighbourhood
    dummy_cos = cos_ma.data*(~mask).astype(int)
    dummy_sin = sin_ma.data*(~mask).astype(int)
    
    ncols, cos_conv = dummy_cols(dummy_cos, kH, val=0)
    ncols, sin_conv = dummy_cols(dummy_sin, kH, val=0)
    
    cos_sumH = ndimage.convolve(cos_conv, weights=kH, mode='wrap')
    cos_sumL = ndimage.convolve(cos_conv, weights=kL, mode='wrap')
    
    sin_sumH = ndimage.convolve(sin_conv, weights=kH, mode='wrap')
    sin_sumL = ndimage.convolve(sin_conv, weights=kL, mode='wrap')
    
    # Remove added columns
    cos_sumH = cos_sumH[:, : int(cos_sumL.shape[1] - ncols)]
    cos_sumL = cos_sumL[:, : int(cos_sumL.shape[1] - ncols)]
    sin_sumH = sin_sumH[:, : int(sin_sumL.shape[1] - ncols)]
    sin_sumL = sin_sumL[:, : int(sin_sumL.shape[1] - ncols)]
    
    # Average angle in local neighbourhood
    cos_avgH_ma = ma.array(data=cos_sumH, mask=mask)/Nval_arr_H
    cos_avgL_ma = ma.array(data=cos_sumL, mask=mask)/Nval_arr_L
    sin_avgH_ma = ma.array(data=sin_sumH, mask=mask)/Nval_arr_H
    sin_avgL_ma = ma.array(data=sin_sumL, mask=mask)/Nval_arr_L
      
    BH = ma.arctan2(sin_avgH_ma, cos_avgH_ma)
    BL = ma.arctan2(sin_avgL_ma, cos_avgL_ma)
    
    # Average velocity ANGLE of neighbours (reference ANGLE for outlier detection):
    angref_ma = f_arr*(BL-BH)
    angref_ma[angref_ma<0] = angref_ma[angref_ma<0] + 2*pi
    angref_ma[angref_ma>pi] = - (2*pi - angref_ma[angref_ma>pi])
    angobs_ma = ma.arctan2(ma.sin(ang_ma), ma.cos(ang_ma))
    
    # Detector array (minimum ANGLE difference between observed and reference):
    diff = angobs_ma - angref_ma
    det_ma = (Vny/pi)*ma.arctan2(ma.sin(diff), ma.cos(diff))
    
    out_mask = np.zeros(det_ma.shape)
    out_mask[abs(det_ma)>0.8*Vny_arr] = 1
    out_mask[(Nval_arr_H<Nmin)|(Nval_arr_L<Nmin)] = 0
    
    # CORRECTION (2 STEP)
    
    # Convolution kernel
    kernel = np.ones(kH.shape)
    
    new_mask = (mask) | (out_mask.astype(bool))
    
    # Array with the number of valid neighbours at each point (outliers removed)
    Nval_arr = local_valid(new_mask, kernel=kernel)
    
    out_mask[Nval_arr<Nmin] = 0
    
    ref_arr = ref_val(data, new_mask, kernel, method='median')
    ref_ma = ma.array(data=ref_arr, mask=mask)
    
    return ref_ma, out_mask
    
#############################################################################################################################

def correct_dualPRF_cmean(radar, field='velocity', Nprf=3, 
                   corr_method='median', Nmin=2):
    
    v_ma = radar.fields[field]['data']
    vcorr_ma = v_ma.copy()
    out_mask = np.zeros(v_ma.shape)
    
    # Dual-PRF parameters
    Vny = radar.instrument_parameters['nyquist_velocity']['data'][0]
    prf_flag = radar.instrument_parameters['prf_flag']['data']
    
    # Array with the primary Nyquist velocity corresponding to each bin
    Nprf_arr = primary_vel(v_ma.shape, Nprf, prf_flag=prf_flag)
    Vny_arr = Vny/Nprf_arr
    
    for nsweep, sweep_slice in enumerate(radar.iter_slice()):
        
        v0 = v_ma[sweep_slice] # velocity field
        vp = Vny_arr[sweep_slice] # primary velocities
        nprfp = Nprf_arr[sweep_slice] # dual-PRF factors
        
        ref, out_mask[sweep_slice] = outlier_detector_cmean(v0, Vny, nprfp, Nmin=Nmin)
                 
        # Convert non-outliers to zero for correction procedure  
        v0_out = v0*out_mask[sweep_slice]
        vp_out = vp*out_mask[sweep_slice]
        ref_out = ref*out_mask[sweep_slice]
        vp_out_L = vp_out.copy() # Only low PRF outliers
        vp_out_L[nprfp==Nprf] = 0
        
        dev = ma.abs(v0_out-ref_out)
        nuw = np.zeros(v0.shape) # Number of unwraps (initialisation)
        
        for ni in range(-Nprf, (Nprf+1)):
            
            # New velocity values for identified outliers
            if abs(ni)==Nprf:
                vcorr_out = v0_out + 2*ni*vp_out_L
            else:
                vcorr_out = v0_out + 2*ni*vp_out
            
            # New deviation for new velocity values
            dev_tmp = ma.abs(vcorr_out-ref_out)
            # Compare with previous
            delta = dev-dev_tmp
            # Update unwrap number
            nuw[delta>0] = ni
            # Update corrected velocity and deviation
            vcorr_out_tmp = v0_out + 2*nuw*vp_out
            dev = ma.abs(vcorr_out_tmp-ref_out)
            
        # Corrected velocity field
        vcorr_ma[sweep_slice] = v0 + 2*nuw*vp
        
    return vcorr_ma, out_mask
        
            
#############################################################################################################################
## PROCESS FILES ############################################################################################################
#############################################################################################################################

data_path = '/home/pav/repos/vilabella_140530/RAW/'
out_path = '/home/pav/repos/vilabella_140530/RESULTS_pkl/'

for f in glob.glob(data_path + '*.RAW*'):
    
    plt.close('all')
    file_name = f[-23:-10]
    outpkl = out_path + file_name + '_corr.pkl'

    radar = pyart.io.read(f)
    fac = radar.instrument_parameters['prt_ratio']['data'][0]
    N = int(round(1/(fac-1)))

    [vcorr_ma, out_mask] = correct_dualPRF_cmean(radar, 'velocity', Nprf=N, Nmin=3)
    

    radar.add_field_like('velocity','velocity_corr_cmean', vcorr_ma, replace_existing = True)
    radar.fields['velocity_corr_cmean']['standard_name'] = 'Corrected Velocity'
    radar.fields['velocity_corr_cmean']['units'] = 'm/s'
    radar.fields['velocity_corr_cmean']['long_name'] = 'dualPRF filtered Velocity'
        
    with open(outpkl, 'wb') as output:
            pickle.dump(radar, output, -1)


#############################################################################################################################
## EOF ######################################################################################################################
#############################################################################################################################
