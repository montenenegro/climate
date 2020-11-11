# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import time
import matplotlib as mpl 
from tqdm import tqdm 
import pymannkendall as mk


# above 66.5°N, starting 1979-01-01
era = xr.open_dataset("C:/Users/Pascal/Desktop/UGAM2/CIA/adaptor.mars.internal"
                      + "-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11d905ef570a.nc")

era_var = np.array(era.t2m[:, 0, :, :])


def MannKendall_test(A_arr):
    
    result = mk.original_test(A_arr).h
        
    return result


start_time = time.time()
start_local_time = time.ctime(start_time)
    
TCs = np.apply_along_axis(MannKendall_test, 0, era_var)

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
plt.imshow(CCs, vmin=np.nanmin(CCs), vmax=np.nanmax(CCs), 
           norm=MidpointNormalize(np.nanmin(CCs), np.nanmax(CCs), 0.),
           cmap='bwr', aspect='auto')
plt.colorbar()

