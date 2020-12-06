# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, Jason Box, GEUS (Geological Survey of Denmark and Greenland)

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import distance
from shapely.geometry import Point
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# %% set required paths

# path to Github repository
gr_path = 'C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'

# path to GHCNv4 datasets
dataset_path = 'C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/' \
    + 'raw_GISTEMP_csv_data/'


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

era_points = np.vstack((era_positions.lon.ravel(), 
                        era_positions.lat.ravel())).T

era_gdfpoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(era_positions.lon, 
                                                             era_positions.lat))


# %% get ERA cell corresponding to station position 

def match(station_point, era_points):
    
    dists = distance.cdist(station_point, era_points)
    
    dist = np.nanmin(dists)
    
    idx = dists.argmin()
    
    station_gdfpoint = Point(station_point[0, 0], station_point[0, 1])
    matching_era_cell = era_gdfpoints.loc[idx]['geometry']
    
    res = gpd.GeoDataFrame({'gistemp_station': [station_gdfpoint], 
                            'era_cell': [matching_era_cell],
                            'distance': pd.Series(dist)})
    
    return res, idx


# %% match GISTEMP and ERA data


def gistemp_era_match(station_filename, 
                      metadata_filename=gr_path + 'GHCNv4_stations.txt'):
    
    # read station data
    station_ID = station_filename.split(os.sep)[-1].split('.')[0]
    station_data = pd.read_csv(station_filename, index_col=0)
    station_data.index = pd.to_datetime(station_data.index)
    station_data = station_data[((station_data.index.year >= 1979) 
                               & (station_data.index.year <= 2020))]
    
    # get station coordinates from metadata file
    stations_metadata = pd.read_csv(metadata_filename, delimiter=r"\s+")
    station_spec = stations_metadata[stations_metadata.ID == station_ID]
    
    station_point = np.vstack((station_spec.Lon.iloc[0], 
                               station_spec.Lat.iloc[0])).T
        
    # get ERA5 cell matching station location
    era_matching_cell, idx = match(station_point, era_points)
    era_matching_rowcol = era_positions.iloc[idx]
    
    # target time series at the point of interest
    era_point_timeseries = np.array(era.t2m[:, 0, int(era_matching_rowcol.row), 
                                    int(era_matching_rowcol.col)]) - 273.15
    
    era_point_timeseries_df = pd.DataFrame({'era_temperature': era_point_timeseries}, 
                                           index=era_time)
    
    # merge GISTEMP and ERA data
    merged_gistemp_era = pd.merge_asof(station_data, era_point_timeseries_df, 
                                       left_index=True, right_index=True)
    
    return merged_gistemp_era, station_ID


# %% run for all stations

station_filenames = glob.glob(dataset_path + '*.csv')

results = {}

for i, sfn in tqdm(enumerate(station_filenames)):
    
    station_results, station_name = gistemp_era_match(station_filename=sfn) 
    
    results[station_name] = station_results 
    