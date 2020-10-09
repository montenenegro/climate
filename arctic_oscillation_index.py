# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#https://www.ncdc.noaa.gov/teleconnections/ao/
#https://cds.climate.copernicus.eu/
#ERA5 monthly averaged data on single levels from 1979 to present

ao_idx=pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/AO_index_monthly.txt')

ao_idx['Time']=pd.to_datetime(ao_idx.Date,format='%Y%m')

