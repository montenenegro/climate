# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from xarray import apply_ufunc
from scipy.stats import linregress

#https://www.ncdc.noaa.gov/teleconnections/ao/
#https://cds.climate.copernicus.eu/
#ERA5 monthly averaged data on single levels from 1979 to present

ao_idx=pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/AO_index_monthly.txt')

ao_idx['Time']=pd.to_datetime(ao_idx.Date,format='%Y%m')

ao_idx=ao_idx[348:]


#above 66.5°N, starting 1979-01-01
era = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11d905ef570a.nc")

u10=era.u10[:,0,:,:]


plt.figure()
plt.imshow(era.u10[0,0,:,:], aspect='auto')

i=0

def linreg(A):
    print(i,'/',501)
    results=linregress(A,ao_idx.Value)
    corrcoeff=results.rvalue
    i+=1
    return corrcoeff

CCs=np.apply_along_axis(linreg, 0, u10)