# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

#https://www.ncdc.noaa.gov/teleconnections/ao/
#https://cds.climate.copernicus.eu/
#ERA5 monthly averaged data on single levels from 1979 to present

ao_idx=pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/AO_index_monthly.txt')

ao_idx['Time']=pd.to_datetime(ao_idx.Date,format='%Y%m')


era = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11d905ef570a.nc")