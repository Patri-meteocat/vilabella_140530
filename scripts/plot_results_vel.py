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
import cPickle as pickle

from pylab import *
from scipy import ndimage
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec

#############################################################################################################################

# Set input and output directories
data_path = '/Users/patriciaaltube/Desktop/Vilabella_Joan/results_pkl/'
out_path = '/Users/patriciaaltube/Desktop/Vilabella_Joan/plots_vel_corr/'

# Custom colormap for velocity
cmap = plt.get_cmap('jet',31)
cmaplist = [cmap(i) for i in range(1,14)+range(19,31)]
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

	# Loop for all sweeps in radar object
	for nsw, sweep_slice in enumerate(radar.iter_slice()):

		# Corresponding elevation and plot title
		el = radar.fixed_angle['data'][nsw]
		title = file_name + ' el=' + '%2.2f' % el + 'deg'

		# Out plot (png) filename
		out_pdf = out_path + file_name + '_' + str(nsw) + '_corr.png'

		# Start plot
		display = pyart.graph.RadarDisplay(radar)
		fig = plt.figure(figsize=(8, 6.5))
		ax=plt.subplot(111)
		# Display corrected velocity, (vmin, vmax) are colorbar limits
 	   	display.plot('velocity_corr_cmean', nsw, ax=ax, vmin=-Vny, vmax=Vny, mask_outside=False, cmap=cmap_vel, colorbar_flag=True, title_flag=False, colorbar_label='[m/s]')
		# Display range rings
  	  	display.plot_range_rings(range(20, 160, 20), lw=0.5, ls=':', ax=ax)
		# Display cross in the radar location		
		display.plot_cross_hair(0.5, ax=ax)
		# Set plot limits
		ax.set_xlim((-150, 150))
		ax.set_ylim((-150, 150))
		# Set tick positions in axes
		ax.set_yticks(np.arange(-120, 160, 40))
		ax.set_xticks(np.arange(-120, 160, 40))
		# Set axes' title
		ax.set_xlabel('Along beam distance East-West (km)')
		ax.set_ylabel('Along beam distance North-South (km)')

		# Title size
		ax.set_title(title, fontsize=14, fontweight='bold')

		# Colorbar label and size
		cbar = display.cbs
		cbar[0].set_label('Dual-PRF corrected radial velocity [m/s]', fontsize=14)
		cbar[0].ax.tick_params(labelsize=16)

		plt.tight_layout()

		fig.savefig(out_pdf)
