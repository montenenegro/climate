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
from mpl_toolkits.basemap import Basemap

#  https://www.ncdc.noaa.gov/teleconnections/ao/
#  https://cds.climate.copernicus.eu/
#  ERA5 monthly averaged data on single levels from 1979 to present

ao_idx = pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'
                     + 'AO_index_monthly.txt')

ao_idx['Time'] = pd.to_datetime(ao_idx.Date, format='%Y%m')

ao_idx = ao_idx[348:]

nao_idx = pd.read_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'
                      + 'NAO_index_monthly_tab.txt', sep='  ', header=None)

nao_idx_arr = nao_idx.iloc[:, 1:].values.flatten()

nao_idx_arr = nao_idx_arr[348:-3]


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


def annual_resampling(A_arr):
    
    A_df = pd.DataFrame(A_arr)
    A_df.index = era_time
    
    A_AR = A_df.resample('1Y').mean()
    
    return A_AR
    

# start_time = time.time()
# start_local_time = time.ctime(start_time)
    
# annual_era_var = np.apply_along_axis(annual_resampling, 0, era_arctic_temp)

# end_time = time.time()
# end_local_time = time.ctime(end_time)
# print("--- Processing time: %.2f minutes ---" % ((end_time - start_time) / 60))
# print("--- Start time: %s ---" % start_local_time)
# print("--- End time: %s ---" % end_local_time)


def linreg_idx(A_arr):
    
    mask = [(~np.isnan(A_arr)) & (~np.isnan(nao_idx_arr))]
    results = linregress(A_arr[mask], nao_idx_arr[mask])
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


start_time = time.time()
start_local_time = time.ctime(start_time)
    
# CCs = np.apply_along_axis(linreg_idx, 0, era_arctic_temp)
CCs = np.apply_along_axis(linreg_idx, 0, era_500hpa)

# print('linreg_idx done')

ratios, variations, rvalues = np.apply_along_axis(linregress_time, 0, era_arctic_temp)


end_time = time.time()
end_local_time = time.ctime(end_time)
print("--- Processing time: %.2f minutes ---" % ((end_time - start_time) / 60))
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
    

# plt.subplot(projection="polar")
plt.figure()
plt.imshow(CCs, vmin=np.nanmin(CCs), vmax=np.nanmax(CCs), 
           norm=MidpointNormalize(np.nanmin(CCs), np.nanmax(CCs), 0.),
           cmap='bwr', aspect='auto')
plt.title('Correlation ERAtemp NAO', fontsize=20)
plt.colorbar()

plt.figure()
plt.imshow(ratios, vmin=np.nanmin(ratios), vmax=np.nanmax(ratios), 
           norm=MidpointNormalize(np.nanmin(ratios), np.nanmax(ratios), 0.),
           cmap='bwr', aspect='auto')
plt.title('Ratio temperature variations/residuals', fontsize=20)
plt.colorbar()

plt.figure()
plt.imshow(rvalues, vmin=np.nanmin(rvalues), vmax=np.nanmax(rvalues), 
           norm=MidpointNormalize(np.nanmin(rvalues), np.nanmax(rvalues), 0.),
           cmap='bwr', aspect='auto')
plt.title('ERAtemp Correlation coefficient with time', fontsize=20)
plt.colorbar()

plt.figure()
plt.imshow(variations, vmin=np.nanmin(variations), vmax=np.nanmax(variations), 
           norm=MidpointNormalize(np.nanmin(variations), np.nanmax(variations), 0.),
           cmap='bwr', aspect='auto')
plt.title('ERAtemp linregress difference', fontsize=20)
plt.colorbar()


# %% correlation per month

monthly_CCs = {}

for month in tqdm(range(1, 13)):
    
    month_selection = ao_idx.Time.dt.month == month
    
    era_month = era_arctic_temp[month_selection, :, :]
    
    ao_month = nao_idx_arr[month_selection]
    
    idx_arr = ao_month 
    
    monthly_CCs[month] = np.apply_along_axis(linreg_idx, 0, era_month)

# %% map correlation per month

min_mCC = np.nanmin([np.nanmin(monthly_CCs[m]) for m in monthly_CCs])
max_mCC = np.nanmax([np.nanmax(monthly_CCs[m]) for m in monthly_CCs])

for month in monthly_CCs:
    
    plt.figure()
    
    CC = monthly_CCs[month]
    
    plt.imshow(CC, vmin=min_mCC, vmax=max_mCC, 
               norm=MidpointNormalize(min_mCC, max_mCC, 0.),
               cmap='bwr', aspect='auto')
    plt.colorbar()

# %% extract average values per month

monthly_CCs_flat = {}

for month in monthly_CCs:
    
    monthly_CCs_flat[month] = monthly_CCs[month].flatten()

plt.figure()
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 
ax2 = plt.subplot(gs[1])
annual_res = plt.boxplot(CCs.flatten())
annual_median = [item.get_ydata() for item in annual_res['medians']][0][0]
ax2.axhline(annual_median, color='darkblue', LineStyle='--', alpha=0.5)

ax = plt.subplot(gs[0])
ax.boxplot(monthly_CCs_flat.values())
ax.set_xticklabels(monthly_CCs_flat.keys())
ax.axhline(annual_median, color='darkblue', LineStyle='--', alpha=0.5)
    

# %% multiprocessing on np.apply_along_axis()

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    
    
# %% Nicole's visualisation

def visualisation(variable, file, lims=False, step=0.1):
    
    plt.figure(figsize=(9, 5))
    
    map = Basemap(projection='npstere', 
                  boundinglat=np.nanmin(np.array(file['latitude'][:])) - 2, 
                  lon_0=0, resolution='l')      
    
    map.drawcoastlines(linewidth=.5)  # draws coastline 
    
    parallels = np.arange(-80, 81, 10.)  # make latitude lines ever 30 degrees from 30N-50N
    meridians = np.arange(-180, 181, 20.)  # make longitude lines every 60 degrees from 95W to 70W
    
    map.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.2, fontsize=8)
    map.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.2, fontsize=8)
    
    lons, lats = np.meshgrid(np.array(file['longitude'][:]), 
                             np.array(file['latitude'][:]))  # 2D lat lon to plot contours
    x, y = map(lons, lats)
    
    if type(lims) == bool:
        clevsf = np.arange(np.nanmin(variable), np.nanmax(variable), step)
    else:
        clevsf = np.arange(lims[0], lims[1], step) 
    
    colormap = 'coolwarm'
    
    csf = map.contourf(x, y, variable, clevsf, extend='both', cmap=colormap)  # filled contour
    cb = map.colorbar(csf, "right", extend='both', size="3%", pad="1%")
    plt.show()

# plt.savefig(path+'Plots/'+name+'_'+sign+'.png', dpi=600)


visualisation(ratios, era_arctic, lims=[-0.1, 3])
visualisation(rvalues, era_arctic, lims=[-0.05, 0.3], step=0.001)
visualisation(CCs, era_500hpa_file, step=0.01)

