# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import linregress
import time
import matplotlib as mpl 
from tqdm import tqdm 
from matplotlib import gridspec
import multiprocessing
import mpl_toolkits
from mpl_toolkits.basemap import Basemap

# above 66.5°N, starting 1979-01-01
era_arctic = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                             + "-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11"
                             + "d905ef570a.nc")
    
plt.figure(figsize=(9, 5))

map = Basemap(projection='npstere', 
              boundinglat=np.nanmin(np.array(file['latitude'][:])) - 2, 
              lon_0=0, resolution='l')      

map.drawcoastlines(linewidth=.5)  # draws coastline 

parallels = np.arange(-80, 81, 10.)  # make latitude lines ever 30 degrees from 30N-50N
meridians = np.arange(-180, 181, 20.)  # make longitude lines every 60 degrees from 95W to 70W

map.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.2, fontsize=8)
map.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.2, fontsize=8)

lons, lats = np.meshgrid(np.array(file['longitude'][:]), 
                         np.array(file['latitude'][:]))  # 2D lat lon to plot contours
x, y = map(lons, lats)

if type(lims) == bool:
    clevsf = np.arange(np.nanmin(variable), np.nanmax(variable), step)
else:
    clevsf = np.arange(lims[0], lims[1], step) 

colormap = 'coolwarm'

csf = map.contourf(x, y, variable, clevsf, extend='both', cmap=colormap)  # filled contour
cb = map.colorbar(csf, "right", extend='both', size="3%", pad="1%")
plt.show()

# plt.savefig(path+'Plots/'+name+'_'+sign+'.png', dpi=600)


visualisation(ratios, era_arctic, lims=[-0.1, 3])
visualisation(rvalues, era_arctic, lims=[-0.05, 0.3], step=0.001)
visualisation(CCs, era_500hpa_file, step=0.01)

