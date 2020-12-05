

import numpy as np 
from scipy import stats
from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt #python plotting package
import os
os.environ["PROJ_LIB"] = 'C:\\Users\\NicoleMV\\Anaconda3\\Library\\share'
import mpl_toolkits
from mpl_toolkits.basemap import Basemap


#### CHOISIR "name" 

name='emergence_2000'# Plot Emergence date with Tmax 1979-2000 and Trend 2000-2019
#name='trend'# Plot trend between 1979-2019
#name='trend_2000'# Plot trend between 2000-2019

sign = 'sign'#'0#if =='sign' Plot only that has significant trend (Emergence or Trend)

def l_trend(var,lon,lat,time,sig=True):
    nlon=len(lon)
    nlat=len(lat)
    nt=len(time)
    vart=np.zeros(nlat*nlon)
    varp=np.zeros(nlat*nlon)
    intercept=np.zeros(nlat*nlon)
    
    if len(var.shape)== 3:        
        var=np.reshape(var,(nt,nlat*nlon)) 
        #print('l_trend: assuming variable as 3D [time,lat,lon]')
        for i in range(nlat*nlon):
            v=var[:,i]  
            vart[i], intercept[i], r_value, varp[i], std_err=stats.linregress(time,v)
            
        vart=np.reshape(vart,(nlat,nlon))
        varp=np.reshape(varp,(nlat,nlon))
        intercept=np.reshape(intercept,(nlat,nlon))
        return (vart,varp, intercept)
        
    else:
        raise ValueError('Variable shape is not 2D or 3D. plese instert variable \
                         in this format var[time,lat,lon] or var[time,lon*lat]')
    
    if sig==False:
        return (vart, varp, intercept) 
    else:
        for i in range(nlat):
            for j in range (nlon):
                if varp[i,j]>sig:
                  vart[i,j]=np.nan
                  intercept[i,j]=np.nan
        return (vart, varp, intercept)

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
len(time)


# dimension

nlon=len(lon)
nlat=len(lat)
ntime=len(time)
ntime
# Here my SLP data is in monthly time dimension. 
#I will take average over the months to make years
mo=12
yr=ntime//mo
year=np.linspace(1979,2019,41)

t2m.shape
t2m=np.reshape(t2m,(yr,mo,nlat,nlon))
t2m.shape


t2m=(np.nanmean(t2m,1))  # taking mean over month diension
t2m.shape
# Now lets calculate linear trend pr year

t2m_trend, t2m_p, offset = l_trend(t2m,lon,lat,year,sig=0.05)

## EMERGENCE DATE
t2m_max = np.nanmax(t2m[0:21,:,:],0)
t2m_2000 = t2m[21:,:,:]
year2000 = np.linspace(2000,2019,20)
t2m_trend2000, pvalue2000, off_2000 = l_trend(t2m_2000,lon,lat,year2000,sig=0.05)

emergence = (t2m_max - off_2000)/t2m_trend2000
#emergence[pvalue2000>0.05]=np.nan
#t2m_trend2000[pvalue2000>0.05]=np.nan

#np.shape(t2m_trend)
#t2m_trend(t2m_p)

### PLOT


if name=='emergence_2000':
    a=emergence
    pval = pvalue2000
    colormap = 'YlOrRd_r'
    limf = [2000,2050,5]
    lims = [2000,2050,20]
    title = 'Emergence date'
    
elif name=='trend_2000':
    a=t2m_trend2000
    pval = pvalue2000
    colormap = 'coolwarm'
    limf = [-0.25,0.25,0.025]
    lims = [-0.25,0.25,0.025]
    title = 'Trend'
else:
    a=t2m_trend
    pval = t2m_p
    colormap = 'coolwarm'
    limf = [-0.25,0.25,0.025]
    lims = [-0.25,0.25,0.025]
    title = 'Trend'

if sign=='sign':
    a[pval>0.05]=np.nan



plt.figure(figsize=(9,5)) #setting the figure size

#map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='l')
map = Basemap(projection='npstere',boundinglat=65,lon_0=0,resolution='l')      
#This like sets the lat lon of the plot. Projection cylinder. 

map.drawcoastlines(linewidth=.5)  #draws coastline 

#parallels = np.arange(-90,91,30.) # make latitude lines ever 30 degrees from 30N-50N
#meridians = np.arange(-180,180,60.) # make longitude lines every 60 degrees from 95W to 70W
parallels = np.arange(-80,81,10.) # make latitude lines ever 30 degrees from 30N-50N
meridians = np.arange(-180,181,20.) # make longitude lines every 60 degrees from 95W to 70W

#labelling the lat and lon dimesion

map.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.2,fontsize=8)
map.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.2,fontsize=8)

lons,lats= np.meshgrid(lon,lat) #2D lat lon to plot contours
x,y = map(lons,lats)

clevsf = np.arange(limf[0], limf[1], limf[2]) 
clevs = np.arange(lims[0], lims[1], lims[2])

#clevs and clevsf sets the contour interval of contour and filled contour. 
#if you don't set it, it will plot default values.

csf = map.contourf(x,y,a,clevsf,extend='both',cmap=colormap) #filled contour
cb = map.colorbar(csf,"right", extend='both',size="3%", pad="1%")
#cs = map.contour(x,y,a,clevs,colors='k',linewidths=0.3)

#plt.clabel(cs, inline=True, fmt='%1.0f', fontsize=3, colors='k')
plt.title(title)
plt.show()

plt.savefig(path+'Plots/'+name+'_'+sign+'.png', dpi=600)
#plt.savefig(path+name+'_2000_sign0.05.png', dpi=600)

#plt.savefig(path+'test_slp.eps', format='eps', dpi=1000) #saving figure
#plt.savefig(path+'test_slp.png', dpi=600) #saving figure


#https://www.afahadabdullah.com/blog/linear-regression-in-python







