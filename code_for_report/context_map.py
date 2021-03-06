# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import glob

# above 66.5°N, starting 1979-01-01
era_arctic = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                             + "-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11"
                             + "d905ef570a.nc")

# %% visualisation
plt.figure()
ax = plt.subplot(111, projection=ccrs.NorthPolarStereo())

ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
# ax.gridlines()
ax.set_global()

ax.set_extent([-180, 180, 64.1, 90], ccrs.PlateCarree())

station_filenames = glob.glob('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'
                              + 'raw_GISTEMP_csv_data/*.csv')

# get station coordinates from metadata file
stations_metadata = pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/'
                                + 'climatic-modes-arctic/GHCNv4_stations.txt',
                                delimiter=r"\s+")

for station_filename in station_filenames:
    
    # read station data
    station_ID = station_filename.split(os.sep)[-1].split('.')[0]
    station_data = pd.read_csv(station_filename, 
                               na_values=999.9, index_col=0).iloc[:, :12]
    
    station_data.index = pd.to_datetime(station_data.index)
    
    station_spec = stations_metadata[stations_metadata.ID == station_ID]
    station_name = station_spec.Station.iloc[0]
    
    plt.scatter(station_spec.Lon.iloc[0], station_spec.Lat.iloc[0], s=100, 
                transform=ccrs.PlateCarree(), color='orange', edgecolors='k', 
                linewidth=0.5, zorder=10)
    
    if station_spec.Lat.iloc[0] < 66.5:
        
        print(station_ID, station_spec)

gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='red')
gl.xlabels_top = True
gl.ylabels_left = True
gl.ylabels_right=True
gl.xlines = True
gl.xlocator = mticker.FixedLocator([])
gl.ylocator = mticker.FixedLocator([66.5])

ax2 = plt.subplot(111, projection=ccrs.NorthPolarStereo())
gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.3, linestyle='-', zorder=10)
# gl2.xlabels_top = False
gl2.ylabels_left = True
# gl2.xlines = False
gl2.xlocator = mticker.FixedLocator(np.arange(-180, 180, 60))
gl2.ylocator = mticker.FixedLocator(np.arange(55, 85+5, 5))
gl2.xformatter = LONGITUDE_FORMATTER
gl2.yformatter = LATITUDE_FORMATTER
gl2.xlabel_style = {'size': 15, 'color': 'gray'}
gl2.xlabel_style = {'color': 'black'}  # 'weight': 'bold'}
gl2.xlabel_style = {'zorder': 12}

plt.savefig('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'
            + 'code_for_report/context_v1.pdf', bbox_inches='tight')

    
