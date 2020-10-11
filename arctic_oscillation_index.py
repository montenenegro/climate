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
#https://www.ncdc.noaa.gov/teleconnections/ao/
#https://cds.climate.copernicus.eu/
#ERA5 monthly averaged data on single levels from 1979 to present

ao_idx=pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/AO_index_monthly.txt')

ao_idx['Time']=pd.to_datetime(ao_idx.Date,format='%Y%m')

ao_idx=ao_idx[348:]

nao_idx=pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/NAO_index_monthly_tab.txt', sep='  ', header=None)

nao_idx_arr=nao_idx.iloc[:,1:].values.flatten()

nao_idx_arr=nao_idx_arr[348:-3]


#above 66.5°N, starting 1979-01-01
era = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11d905ef570a.nc")

era_var=era.t2m[:,0,:,:]

def linreg(A):
    A_arr=np.array(A)
    #ao_arr=np.array(ao_idx.Value)
    ao_arr=np.array(nao_idx_arr)
    mask=[(~np.isnan(A_arr)) & (~np.isnan(ao_arr))]
    results=linregress(A_arr[mask],ao_arr[mask])
    corrcoeff=results.rvalue
    return corrcoeff


start_time = time.time()
start_local_time = time.ctime(start_time)
    
CCs=np.apply_along_axis(linreg, 0, era_var)

end_time = time.time()
end_local_time = time.ctime(end_time)
print("--- Processing time: %.2f minutes ---" % ((end_time - start_time)/60))
print("--- Start time: %s ---" % start_local_time)
print("--- End time: %s ---" % end_local_time)


# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    



plt.subplot(projection="polar")
plt.imshow(CCs, vmin=np.nanmin(CCs), vmax=np.nanmax(CCs), norm=MidpointNormalize(np.nanmin(CCs),np.nanmax(CCs), 0.),
           cmap='bwr', aspect='auto')
plt.colorbar()

# plt.pcolormesh(np.indices((np.shape(CCs)[0],np.shape(CCs)[1]))[0], 
#                np.indices((np.shape(CCs)[0],np.shape(CCs)[1]))[1], 
#                CCs)
