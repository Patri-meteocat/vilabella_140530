#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################################################################
# Load libraries
import os.path
import glob
import matplotlib.pyplot as plt
import pylab as plb
import matplotlib as mpl
import pyart
import numpy as np
import scipy as sp
import numpy.ma as ma
import pickle

from pylab import *
from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

#############################################################################################################################

# Set input and output directories
data_path = '/home/pav/repos/vilabella_140530/RESULTS_pkl/'
out_path = '/home/pav/repos/vilabella_140530/PLOTS/dualpanel_zv/'

# AQUIIIII!!!! (todo en [km])
lims_x = [-80, 20]
lims_y = [-80, 20]
ring_step = 20 

# Custom colormap for velocity
cmap = plt.get_cmap('jet',31)
cmaplist = [cmap(i) for i in list(range(1,14))+list(range(19,31))]
cmaplist[12] = (1,1,1,1)
# create the new map
cmap_vel = cmap.from_list('Custom cmap', cmaplist, cmap.N)
cmap_vel.set_bad('lightgrey',1.)

# Loop for all pickle files in input directory
for f in glob.glob(data_path + '*.pkl'):

    # Open pickle file and load pyart radar object
    with open(f, "rb") as i_file:
        radar = pickle.load(i_file)

    file_name = f[-22:-9]
    Vny = radar.instrument_parameters['nyquist_velocity']['data'][0]
    
    # SPECKLE FILTER: Boundary management for speckle filter is not working (not applied: delta=0)
    speckle_z = pyart.correct.despeckle_field(radar, 'reflectivity', delta=0)
    mask_z = speckle_z.gate_excluded
    speckle_z._merge(mask_z, 'or', True)
    radar.fields['reflectivity']['data'].mask = mask_z

    speckle_v = pyart.correct.despeckle_field(radar, 'velocity_corr_cmean', delta=0)
    mask_v = speckle_v.gate_excluded
    speckle_v._merge(mask_v, 'or', True)
    radar.fields['velocity_corr_cmean']['data'].mask = mask_v

    # Loop for all sweeps in radar object
    for nsw, sweep_slice in enumerate(radar.iter_slice()):

        # Corresponding elevation and plot title
        el = radar.fixed_angle['data'][nsw]
        title = file_name + ' el=' + '%2.2f' % el + 'deg'

        # Out plot (png) filename
        out_pdf = out_path + 'el' + '%2.2f' % el + '_' + file_name + '_' + '_corr.png'

        # Start plot
        display = pyart.graph.RadarDisplay(radar)
        fig = plt.figure(figsize=(16, 6.5))
        
        ax=plt.subplot(121)
        # Display corrected velocity, (vmin, vmax) are colorbar limits
        display.plot('velocity_corr_cmean', nsw, ax=ax, vmin=-Vny, vmax=Vny, mask_outside=False, cmap=cmap_vel, colorbar_flag=True, title_flag=False, colorbar_label='[m/s]')
        # Display range rings
        display.plot_range_rings(list(range(max([lims_x[0], lims_y[0]]), max([lims_x[1], lims_y[1]])+ring_step, ring_step)), lw=0.5, ls=':', ax=ax)
        # Display cross in the radar location        
        display.plot_cross_hair(0.5, ax=ax)
        # Set plot limits
        ax.set_xlim((lims_x[0], lims_x[1]))
        ax.set_ylim((lims_y[0], lims_y[1]))
        # Set tick positions in axes
        ax.set_xticks(np.arange(lims_x[0]+20, lims_x[1], 20))
        ax.set_yticks(np.arange(lims_y[0]+20, lims_y[1], 20))
        # Set axes' title
        ax.set_xlabel('Along beam distance East-West (km)')
        ax.set_ylabel('Along beam distance North-South (km)')

        # Title size
        ax.set_title(title, fontsize=14, fontweight='bold')


        ax1=plt.subplot(122)
        # Display corrected velocity, (vmin, vmax) are colorbar limits
        display.plot('reflectivity', nsw, ax=ax1, vmin=0, vmax=60., mask_outside=False, colorbar_flag=True, title_flag=False, colorbar_label='[dBZ]')
        # Display range rings
        display.plot_range_rings(list(range(max([lims_x[0], lims_y[0]]), max([lims_x[1], lims_y[1]])+ring_step, ring_step)), lw=0.5, ls=':', ax=ax1)
        # Display cross in the radar location        
        display.plot_cross_hair(0.5, ax=ax1)
        # Set plot limits
        ax1.set_xlim((lims_x[0], lims_x[1]))
        ax1.set_ylim((lims_y[0], lims_y[1]))
        # Set tick positions in axes
        ax1.set_yticks(np.arange(lims_x[0]+20, lims_x[1], 20))
        ax1.set_xticks(np.arange(lims_y[0]+20, lims_y[1], 20))
        # Set axes' title
        ax1.set_xlabel('Along beam distance East-West (km)')
        ax1.set_ylabel('Along beam distance North-South (km)')

        # Title size
        ax1.set_title(title, fontsize=14, fontweight='bold')


        # Colorbar label and size
        cbar = display.cbs
        cbar[0].set_label('Dual-PRF radial velocity [m/s]', fontsize=14)
        cbar[0].ax.tick_params(labelsize=16)
        
        cbar = display.cbs
        cbar[1].set_label('Reflectivity [dBZ]', fontsize=14)
        cbar[1].ax.tick_params(labelsize=16)

        plt.tight_layout()

        fig.savefig(out_pdf)
        plt.close("all")

