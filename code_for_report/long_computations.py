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
# from mpl_toolkits.basemap import Basemap

#  https://www.ncdc.noaa.gov/teleconnections/ao/
#  https://cds.climate.copernicus.eu/
#  ERA5 monthly averaged data on single levels from 1979 to present

ao_idx = pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'
                     + 'AO_index_monthly.txt')

ao_idx['Time'] = pd.to_datetime(ao_idx.Date, format='%Y%m')

ao_idx = ao_idx[348:]

ao_idx_arr = np.array(ao_idx.Value)

ao_idx.index = ao_idx.Time
ao_idx.drop(columns=['Date', 'Time'], inplace=True)
ao_idx_annual = ao_idx.resample('1Y').mean()
ao_idx_arr_annual = np.array(ao_idx_annual.Value)

nao_idx = pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'
                      + 'NAO_index_monthly_tab.txt', sep='  ', header=None)

nao_idx_arr = nao_idx.iloc[:, 1:].values.flatten()

nao_idx_arr = nao_idx_arr[348:-3]

nao_idx_df = pd.DataFrame(nao_idx_arr)
nao_idx_df.index = ao_idx.index
nao_idx_annual = nao_idx_df.resample('1Y').mean()
nao_idx_arr_annual = np.array(nao_idx_annual.iloc[:,0])

# above 66.5°N, starting 1979-01-01
era_arctic = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                             + "-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11"
                             + "d905ef570a.nc")

# 500hPa, northern hemisphere, starting 1979-01-01
era_500hpa_file = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                                  + "-1606980488.2916174-29195-17-742f648e-b0f6-4780-92d0"
                                  + "-888ed5090d2f.nc")
era_500hpa = era_500hpa_file.z_0001[:-1, :, :]

# 1000hPa, northern hemisphere, starting 1979-01-01
era_1000hpa_file = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                                   + "-1606980517.300832-6782-15-5386caa7-724d-4db7-9349"
                                   + "-4d72949f9cee.nc")
era_1000hpa = era_1000hpa_file.z_0001[:-1, :, :]

era_arctic_temp = np.array(era_arctic.t2m[:, 0, :, :])

era_time = pd.to_datetime(np.array(era_arctic['time']), format='%Y-%*-%dT00:00:00.000000000')

# above 0°N, starting 1979-01-01
era_mid = xr.open_dataset('H:/adaptor.mars.internal-1607766295.7032957-886-22'
                          +'-0fc4ca2c-a399-4fbe-8133-1e259b447826.nc')

# lat_mask = (era_mid.latitude > 20) & (era_mid.latitude < 66.5)

era_mid_temp = np.array(era_mid.t2m[:, 0, :, :])
    

start_time = time.time()
start_local_time = time.ctime(start_time)
print('ANNUAL RESAMPLING')
    
era_500hpa_annual = era_500hpa_file.z_0001[:-1, :, :].resample(time='1Y').mean()
era_1000hpa_annual = era_1000hpa_file.z_0001[:-1, :, :].resample(time='1Y').mean()

end_time = time.time()
end_local_time = time.ctime(end_time)
print("--- Processing time: %.2f minutes ---" % ((end_time - start_time) / 60))
print("--- Start time: %s ---" % start_local_time)
print("--- End time: %s ---" % end_local_time)


def linreg_idx(A_arr, cm):
    
    mask = [(~np.isnan(A_arr)) & (~np.isnan(cm))]
    results = linregress(A_arr[mask], cm[mask])
    corrcoeff = results.rvalue
    
    return corrcoeff


def linregress_time(A_arr):
    
    x = np.arange(0, len(A_arr))
    
    mask = [(~np.isnan(A_arr)) & (~np.isnan(x))]
    results = linregress((A_arr - 273.15)[mask], x[mask])
    y = results.slope * x + results.intercept
    
    rvalue = results.rvalue 
    
    residuals = np.nanmean(np.abs(y - A_arr))
    
    variation = y[-1] - y[0]
    
    ratio = variation / residuals 
    
    return ratio, variation, rvalue


# %% relation tempeature and mode of climate variability
# start_time = time.time()
# start_local_time = time.ctime(start_time)

# CCs_AO_mo = np.apply_along_axis(linreg_idx, 0, era_1000hpa, ao_idx_arr)
# print('AO M')
# np.save('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/CC_1000hpa_AO_m.npy', 
#         CCs_AO_mo)

# print('NAO M')
# CCs_NAO_mo = np.apply_along_axis(linreg_idx, 0, era_500hpa, nao_idx_arr)
# np.save('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/CC_500hpa_NAO_m.npy', 
#         CCs_NAO_mo)

# print('AO A')
# CCs_AO_an = np.apply_along_axis(linreg_idx, 0, era_1000hpa_annual, 
#                                 ao_idx_arr_annual)
# np.save('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/CC_1000hpa_AO_a.npy', 
#         CCs_AO_an)

# print('NAO A')
# CCs_NAO_an = np.apply_along_axis(linreg_idx, 0, era_500hpa_annual, 
#                                  nao_idx_arr_annual)
# np.save('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/CC_500hpa_NAO_a.npy', 
#         CCs_NAO_an)

# end_time = time.time()
# end_local_time = time.ctime(end_time)
# print("--- Processing time: %.2f minutes ---" % ((end_time - start_time) / 60))
# print("--- Start time: %s ---" % start_local_time)
# print("--- End time: %s ---" % end_local_time)


# %% temperature trends

start_time = time.time()
start_local_time = time.ctime(start_time)

print('ARCTIC')
ratios_arc, variations_arc, rvalues_arc = np.apply_along_axis(linregress_time, 
                                                              0, era_arctic_temp)

print('MID LATITUDES')
ratios_mid, variations_mid, rvalues_mid = np.apply_along_axis(linregress_time, 
                                                              0, era_mid_temp)

end_time = time.time()
end_local_time = time.ctime(end_time)
print("--- Processing time: %.2f minutes ---" % ((end_time - start_time) / 60))
print("--- Start time: %s ---" % start_local_time)
print("--- End time: %s ---" % end_local_time)



