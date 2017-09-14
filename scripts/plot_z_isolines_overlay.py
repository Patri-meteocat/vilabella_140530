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

#############################################################################################################################

# SETTINGS

# Input and output directories
data_path = '/home/pav/repos/vilabella_140530/RESULTS_pkl/'
out_path = '/home/pav/repos/vilabella_140530/PLOTS/overlay_z_isolines/'

# plot limits
lims_x = [-80, 20]
lims_y = [-80, 20]
ring_step = 20 

# Contour levels (reflectivity)
levels = np.arange(5, 55, 10)

# Figure size (inches)
fig_size = [12, 10]

#############################################################################################################################

# Custom colormap for velocity
cmap = plt.get_cmap('jet',31)
cmaplist = [cmap(i) for i in list(range(1,14))+list(range(19,31))]
cmaplist[12] = (1,1,1,1)
# create the new map
cmap_vel = cmap.from_list('Custom cmap', cmaplist, cmap.N)
cmap_vel.set_bad('lightgrey',1.)

#############################################################################################################################

# Loop for all pickle files in input directory
for f in glob.glob(data_path + '*.pkl'):

    with open(f, "rb") as i_file:
        radar = pickle.load(i_file)
    
    file_name = f[-22:-9]
    v_ny = radar.instrument_parameters['nyquist_velocity']['data'][0]

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
        out_fig = out_path + 'el' + '%2.2f' % el + '_' + file_name + '_' + '.png'

        # get data
        data = radar.get_field(nsw, 'reflectivity')

        # smooth out the lines
        data = sp.ndimage.gaussian_filter(data, sigma=1.1)
        
        # Sweep gate coordinates (in km)
        x, y, z = radar.get_gate_x_y_z(nsw, edges=False)
        x /= 1000.0
        y /= 1000.0
        z /= 1000.0

        display = pyart.graph.RadarDisplay(radar)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        display.plot('velocity_corr_cmean', sweep=nsw, vmin=-v_ny, vmax=v_ny, fig=fig,
                     ax=ax, cmap=cmap_vel, colorbar_label='Velocity (m/s)')
        display.set_limits(xlim=lims_x, ylim=lims_y)

        # adds coutours to plot
        contours = ax.contour(x, y, data, levels, linewidths=1.5, colors='k',
                              linestyles='solid', antialiased=True)
        # adds contour labels (fmt= '%r' displays 10.0 vs 10.0000)
        plt.clabel(contours, levels, fmt='%r', inline=True, fontsize=10)

        
        # Display range rings
        display.plot_range_rings(list(range(max([lims_x[0], lims_y[0]]), 
                                            max([lims_x[1], lims_y[1]]) + ring_step, ring_step)), 
                                 lw=0.5, ls=':', ax=ax)

        # Display cross in the radar location        
        display.plot_cross_hair(0.5, ax=ax)

        # Set plot limits
        ax.set_xlim((lims_x[0], lims_x[1]))
        ax.set_ylim((lims_y[0], lims_y[1]))
        # Set tick positions in axes
        ax.set_yticks(np.arange(lims_x[0]+20, lims_x[1], 20))
        ax.set_xticks(np.arange(lims_y[0]+20, lims_y[1], 20))
        # set axes' title
        ax.set_xlabel('Along beam distance E-W (km)')
        ax.set_ylabel('Along beam distance N-S (km)')

        # Title size
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Colorbar label and size
        cbar = display.cbs
        cbar[0].set_label('Dual-PRF radial velocity [m/s]', fontsize=14)
        cbar[0].ax.tick_params(labelsize=16)

        plt.tight_layout()

        fig.savefig(out_fig)
        plt.close("all")

