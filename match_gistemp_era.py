# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, Jason Box, GEUS (Geological Survey of Denmark and Greenland)

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import datetime
from scipy.spatial import cKDTree
import glob
import os
# from tqdm import tqdm
import matplotlib.pyplot as plt


# %% set path to Github repository

path = 'C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/raw_GISTEMP_csv_data/'

# %% get centroid coordinates of ERA cells

# above 66.5°N, starting 1979-01-01
era = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                      + "-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11"
                      + "d905ef570a.nc")

era_time = pd.to_datetime(np.array(era['time']), format='%Y-%*-%dT00:00:00.000000000')
    
lon = np.array(era['longitude'])
lat = np.array(era['latitude'])
    
lon_mat, lat_mat = np.meshgrid(lon, lat)

rows, cols = np.meshgrid(np.arange(np.shape(lat_mat)[0]), 
                         np.arange(np.shape(lat_mat)[1]))

era_positions = pd.DataFrame({'row': rows.ravel(),
                              'col': cols.ravel(),
                              'lon': lon_mat.ravel(),
                              'lat': lat_mat.ravel()})

era_gdfpoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(era_positions.lon, 
                                                             era_positions.lat))


# %% get closest points from two GeoDataFrame

def ckdnearest(nA, nB):
    
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    
    gdf_A = station_gdfpoint.reset_index(drop=True)['geometry']
    gdfmatching_B = era_gdfpoints.loc[idx].reset_index(drop=True)['geometry']
    
    res = pd.DataFrame({'gistemp_station': gdf_A, 
                        'era_cell': gdfmatching_B,
                        'distance': pd.Series(dist)})
    
    return res, idx


# %% match GISTEMP and ERA data



def gistemp_era_match(station_filename, metadata_filename):
    
    
    # read station data
    station_ID = station_filename.split('/')[-1].split('.')[0]
    station_data = pd.read_csv(station_filename, 
                               na_values=999.9, index_col=0).iloc[:, :12]
    
    station_data = data.stack(dropna=False).reset_index()
    station_data.rename(columns={'level_1': 'MONTH', 0: 'Temperature_C'}, 
                        inplace=True)
    
    station_data.index = pd.to_datetime(station_data.YEAR.astype(str) 
                                        + station_data.MONTH, format='%Y%b')
    
    station_data = station_data[((station_data.YEAR >= 1979) 
                                & (station_data.YEAR <= 2020))]
    
    station_data.drop(['YEAR', 'MONTH'], axis=1, inplace=True)
    
    # get station coordinates from metadata file
    stations_metadata = pd.read_csv(metadata_filename, delimiter=r"\s+")
    station_spec = stations_metadata[stations_metadata.ID==station_ID]
    station_geometry = gpd.points_from_xy(station_spec.Lon, 
                                          station_spec.Lat)
    
    station_gdfpoint = gpd.GeoDataFrame(geometry=station_geometry)
    
    station_point = np.vstack((station_spec.Lon.iloc[0], 
                               station_spec.Lat.iloc[0])).T
        
    # get closest ERA5 point to station location
    era_matching_cell, idx = ckdnearest(station_point, era_points)
    era_matching_rowcol = era_positions.iloc[idx[0]]
    
    # target time series at the point of interest
    era_point_timeseries = np.array(era.t2m[:, 0, int(era_matching_rowcol.row), 
                                    int(era_matching_rowcol.col)]) - 273.15
    
    era_point_timeseries_df = pd.DataFrame({'era_temperature': era_point_timeseries}, 
                                           index=era_time)
    
    # merge GISTEMP and ERA data
    merged_gistemp_era=pd.merge_asof(station_data, era_point_timeseries_df, 
                                     left_index=True, right_index=True)
    
    return merged_gistemp_era, station_ID


# %% run for all stations

plt=0

station_filenames=glob.glob(path+'raw_GISTEMP_csv_data/*.csv')

n=len(station_filenames)

for i,sfn in enumerate(station_filenames):
# for sfn in station_filenames:
    results,name=gistemp_era_match(station_filename=sfn)
    print('station ',i+1,name,'remaining:',n-i,'of',n)    
    
    # print('mean GISTEMP-ERA bias: %.3f' %np.nanmean(results.gistemp_temperature-results.era_temperature))
    
    if plt:
        plt.figure()
        plt.plot(results.gistemp_temperature-results.era_temperature,'o-', color='darkorange')
        plt.xlabel('time (years)',fontsize=18)
        plt.ylabel('GISTEMP minus ERA temperature (°C)',fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()
    results.to_csv(path+"results/"+name+'.csv')