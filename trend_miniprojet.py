# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:03:36 2020

@author: NicoleMV
"""


import numpy as np 
from scipy import stats
from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt #python plotting package
import os
os.environ["PROJ_LIB"] = 'C:\\Users\\NicoleMV\\Anaconda3\\Library\\share'
import mpl_toolkits
from mpl_toolkits.basemap import Basemap


def l_trend(var,lon,lat,time,sig=True):
    nlon=len(lon)
    nlat=len(lat)
    nt=len(time)
    vart=np.zeros(nlat*nlon)
    varp=np.zeros(nlat*nlon)
    
    if len(var.shape)== 3:        
        var=np.reshape(var,(nt,nlat*nlon)) 
        #print('l_trend: assuming variable as 3D [time,lat,lon]')
        for i in range(nlat*nlon):
            v=var[:,i]  
            vart[i], intercept, r_value, varp[i], std_err=stats.linregress(time,v)
            
        vart=np.reshape(vart,(nlat,nlon))
        varp=np.reshape(varp,(nlat,nlon))
        return (vart,varp)
        
    else:
        raise ValueError('Variable shape is not 2D or 3D. plese instert variable \
                         in this format var[time,lat,lon] or var[time,lon*lat]')
    
    if sig==False:
        return (vart, varp) 
    else:
        for i in range(nlat):
            for j in range (nlon):
                if varp[i,j]>sig:
                  vart[i,j]=np.nan
        return (vart, varp)

# after reading the SLP file

path ='C:/Users/NicoleMV/Desktop/Master_ACSC/Semetre_3/Climate/Mini_projet/'
file=nc(path + '-1602255451.139694-24165-26-eecb89cc-17e1-4466-b8a2-11d905ef570a.nc')
file

t2m=file.variables['t2m'][:-9]
t2m=t2m[:,0,:,:]
t2m.shape


lon=file.variables['longitude'][:]
lat=file.variables['latitude'][:]
time=file.variables['time'][:-9]

# dimension

nlon=len(lon)
nlat=len(lat)
ntime=len(time)

# Here my SLP data is in monthly time dimension. 
#I will take average over the months to make years
mo=12
yr=ntime//mo
year=np.linspace(1979,2019,41)


t2m=np.reshape(t2m,(yr,mo,nlat,nlon))

t2m=(np.nanmean(t2m,1))  # taking mean over month diension

# Now lets calculate linear trend pr year

t2m_trend, t2m_p = l_trend(t2m,lon,lat,year,sig=0.05)