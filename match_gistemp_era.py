# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, Jason Box, GEUS (Geological Survey of Denmark and Greenland)

Match GHCNv4 meteorological station datasets with corresponding ERA5 cells.

"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.spatial import distance as dist
from shapely.geometry import Point
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# %% set required paths

# path to Github repository
gr_path = 'C:/Users/Pascal/Desktop/GEUS_2019/GISTEMP_analysis/'

# path to folder containing GHCNv4 datasets
dataset_path = 'C:/Users/Pascal/Desktop/GEUS_2019/GISTEMP_analysis/raw_GISTEMP_csv_data/'

# filename of era5 dataset
era5_filename = "C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal" \
            + "-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11" \
            + "d905ef570a.nc"


# %% get centroid coordinates of ERA5 cells

# open ERA5 dataset
era5 = xr.open_dataset(era5_path)

# convert time to datetime
era5_time = pd.to_datetime(np.array(era5['time']), 
                           format='%Y-%*-%dT00:00:00.000000000')

# extract coordinates
lon = np.array(era5['longitude'])
lat = np.array(era5['latitude'])
lon_mat, lat_mat = np.meshgrid(lon, lat)

# save corresponding matrix positions
rows, cols = np.meshgrid(np.arange(np.shape(lat_mat)[0]), 
                         np.arange(np.shape(lat_mat)[1]))

era5_positions = pd.DataFrame({'row': rows.ravel(),
                               'col': cols.ravel(),
                               'lon': lon_mat.ravel(),
                               'lat': lat_mat.ravel()})

era5_points = np.vstack((era5_positions.lon.ravel(), 
                        era5_positions.lat.ravel())).T

# create GeoDataFrame from points
era5_gdfpoints = gpd.GeoDataFrame(geometry=gpd.points_from_xy(era5_positions.lon, 
                                                              era5_positions.lat))


# %% get ERA5 cell corresponding to station position 

def data_match(station_point, era5_points):
    '''
    Find closest ERA5 cell centroid to given GHCNv4 station.
    
    INPUTS:
        station_point: station position in (lon, lat) [array]
        era5_points: centroids of ERA5 cells in (lon, lat) [array]
        
    OUTPUTS:
        results: shapely points of GHCNv4 station and closest ERA5 cell 
                 centroid and associated distance [GeoDataFrame]
        era5_cell: index of the matching ERA5 cell [int]
    '''
    distances = dist.cdist(station_point, era5_points)
    
    distance = np.nanmin(distances)
    
    era5_cell = distances.argmin()
    
    station_gdfpoint = Point(station_point[0, 0], station_point[0, 1])
    matching_era5_cell = era5_gdfpoints.loc[era5_cell]['geometry']
    
    results = gpd.GeoDataFrame({'GHCNv4_station': [station_gdfpoint], 
                                'ERA5_cell': [matching_era5_cell],
                                'distance': pd.Series(dist)})
    
    return results, era5_cell


# %% match GHCNv4 and ERA5 data


def GHCNv4_ERA5_merger(station_filename, 
                       metadata_filename=gr_path + 'GHCNv4_stations.txt'):
    '''
    Read and prepare given a GHCNv4 time series before merging with the 
    corresponding ERA5 cell time series. 
    
    INPUTS:
        station_filename: file name of GHCNv4 station [string]
        metadata_filename: file name of GHCNv4 metadata [string]
    
    OUTPUTS:
        merged_ghcnv4_era5: combination of GHCNv4 and ERA5 time series for a
                            given station [DataFrame]
        station_ID: ID of selected GHCNv4 station [string]
        station_name: name of selected GHCNv4 station [string]
    '''
    # read station data
    station_ID = station_filename.split(os.sep)[-1].split('.')[0]
    station_data = pd.read_csv(station_filename, 
                               na_values=999.9, index_col=0).iloc[:, :12]
    
    station_data = station_data.stack(dropna=False).reset_index()
    station_data.rename(columns={'level_1': 'MONTH', 0: 'GHCNv4_temperature'}, 
                        inplace=True)
    
    station_data.index = pd.to_datetime(station_data.YEAR.astype(str) 
                                        + station_data.MONTH, format='%Y%b')
    
    station_data.drop(['YEAR', 'MONTH'], axis=1, inplace=True)
    
    # get station coordinates from metadata file
    stations_metadata = pd.read_csv(metadata_filename, delimiter=r"\s+")
    station_spec = stations_metadata[stations_metadata.ID == station_ID]
    station_name = station_spec.Station.iloc[0]
    
    station_point = np.vstack((station_spec.Lon.iloc[0], 
                               station_spec.Lat.iloc[0])).T
        
    # get ERA5 cell matching station location
    era5_matching_cell, idx = data_match(station_point, era5_points)
    era5_matching_rowcol = era5_positions.iloc[idx]
    
    # target time series at the point of interest
    era5_point_timeseries = np.array(era5.t2m[:, 0, int(era5_matching_rowcol.row), 
                                     int(era5_matching_rowcol.col)]) - 273.15
    
    era5_point_timeseries_df = pd.DataFrame({'ERA5_temperature': era5_point_timeseries}, 
                                            index=era5_time)
    
    # merge GHCNv4 and ERA5 data
    merged_ghcnv4_era5 = pd.merge_asof(station_data, era5_point_timeseries_df, 
                                       left_index=True, right_index=True)
    
    return merged_ghcnv4_era5, station_ID, station_name


# %% run for all stations

# options
visualisation = False
save = False

station_filenames = glob.glob(dataset_path + '*.csv')

# store in dict to keep initial (variable) GHCNv4 temporal coverage
results = {}

for i, sfn in tqdm(enumerate(station_filenames)):
    
    st_results, st_ID, st_name = GHCNv4_ERA5_merger(station_filename=sfn) 
    
    results[st_ID] = st_results 
    
    if visualisation:
        
        plt.figure()
        plt.plot(st_results.GHCNv4_temperature - st_results.ERA5_temperature,
                 'o-', color='darkorange')
        plt.ylabel('GHCNv4 minus ERA5 temperature (°C)', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.axvline(0, LineStyle='--', color='darkgray')
        plt.title('%s (%s)' %(st_name, st_ID), fontsize=20)
        
if save:
    
