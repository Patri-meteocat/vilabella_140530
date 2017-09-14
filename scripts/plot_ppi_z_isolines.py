#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################################################################
# Load libraries
import os.path
import glob
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyart
import pickle

from pylab import *
from scipy import ndimage

#############################################################################################################################

# SETTINGS

# Input and output directories
data_path = '/home/pav/repos/vilabella_140530/RESULTS_pkl/'
out_path = '/home/pav/repos/vilabella_140530/PLOTS/ppi_z_isolines/'

shp_path = '/home/pav/repos/vilabella_140530/SHP/'
shp_file_name = 'comarques'
shp_file = shp_path + shp_file_name

# Contour levels (reflectivity)
levels = np.arange(5, 55, 10)

# plot limits
lat_lims = [41.1, 42.2]
lon_lims = [0.45, 1.65]
lat_step = 0.2
lon_step = 0.5

range_rings = list(range(20, 160, 20))

# Figure size (inches)
fig_size = [12, 10.5]

#############################################################################################################################

# Custom colormap for velocity
cmap = plt.get_cmap('jet',31)
cmaplist = [cmap(i) for i in list(range(1,14))+list(range(19,31))]
cmaplist[12] = (1,1,1,1)
# create the new map
cmap_vel = cmap.from_list('Custom cmap', cmaplist, cmap.N)
cmap_vel.set_bad('lightgrey',1.)

#############################################################################################################################

lat_ticks = np.arange(np.floor(lat_lims[0]), np.ceil(lat_lims[1]), lat_step)
lon_ticks = np.arange(np.floor(lon_lims[0]), np.ceil(lon_lims[1]), lon_step)

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
    for nsw, sw_slice in enumerate(radar.iter_slice()):

        # Corresponding elevation and plot title
        el = radar.fixed_angle['data'][nsw]
        title = file_name + ' el=' + '%2.2f' % el + 'deg'

        # Out plot (png) filename
        out_fig = out_path + 'el' + '%2.2f' % el + '_' + file_name + '_' + '.png'

        # get data
        data = radar.get_field(nsw, 'reflectivity')

        # smooth out the lines
        data = sp.ndimage.gaussian_filter(data, sigma=1.1)
        
        # Sweep gate coordinates
        lat = radar.gate_latitude['data'][sw_slice]
        lon = radar.gate_longitude['data'][sw_slice]

        display = pyart.graph.RadarMapDisplay(radar)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

        display.plot_ppi_map('velocity_corr_cmean', nsw, vmin=-v_ny, vmax=v_ny, projection='lcc',
resolution='h', fig=fig, ax=ax, cmap=cmap_vel,
min_lon=lon_lims[0], max_lon=lon_lims[1], min_lat=lat_lims[0], max_lat=lat_lims[1],
lat_0=radar.latitude['data'][0], lon_0=radar.longitude['data'][0])


        # Display cross in the radar location        
        display.plot_point(radar.longitude['data'][0], radar.latitude['data'][0], symbol='+', color='k')

        x, y = display.basemap(lon, lat)
        contours = display.basemap.contour(x, y, data, levels, colors='k', linewidths=1.)

        # adds contour labels (fmt= '%r' displays 10.0 vs 10.0000)
        plt.clabel(contours, levels, fmt='%r', inline=True, fontsize=8)

        # plot range rings
        display.plot_range_rings(range_rings, ax=None, col='k', ls=':', lw=0.8)

        # Indicate the radar location with a point
        display.basemap.readshapefile(shp_file, 'comarques', color='grey')

        display.basemap.drawparallels(lat_ticks, labels=[True, False, False, False], linewidth=0.01)
        display.basemap.drawmeridians(lon_ticks, labels=[False, False, False, True], linewidth=0.01)

        # Title size
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')

        # Colorbar label and size
        cbar = display.cbs
        cbar[0].set_label('Dual-PRF radial velocity [m/s]', fontsize=14)
        cbar[0].ax.tick_params(labelsize=16)
        
        #fig.tight_layout(pad=1)

        fig.savefig(out_fig, bbox_inches='tight')
        plt.close("all")

