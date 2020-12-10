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
import glob

# above 66.5°N, starting 1979-01-01
era_arctic = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                             + "-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11"
                             + "d905ef570a.nc")

# %% visualisation
ax = plt.subplot(111, projection=ccrs.NorthPolarStereo())

ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
ax.gridlines()
ax.set_global()

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
    
    plt.scatter(station_spec.Lon.iloc[0], station_spec.Lat.iloc[0], color='m')

