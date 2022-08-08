# Plots climatological ky anisotropy metrics

import numpy               as np
import xarray              as xr
import dim
import const
from fxn_misc              import get_season     
from matplotlib            import pyplot as plt

# INPUT -----------------------------------------------------------   
datadir          = '/nird/home/edu061/w21/data/'
figdir           = '/nird/home/edu061/w21/fig/'
year             = np.arange(1979,2018,1)
var              = 'alpha'
wavenumber       = np.arange(1,14,1)
season           = 'ANNUAL'
no_stw_flag      = 0
outputdata       = True
# ----------------------------------------------------------------- 

# read data
if no_stw_flag == 1:
    filename = datadir + 'yk_anisotropy_metrics_without_stationary_wave_clim_' + season + '_' + str(year[0]) + '-' + str(year[-1]) + '.nc' 
else:
    filename = datadir + 'yk_anisotropy_metrics_clim_' + season + '_' + str(year[0]) + '-' + str(year[-1]) + '.nc' 
ds   = xr.open_dataset(filename).sel(wavenumber=wavenumber)
data = ds[var].values
ds.close()

# calculate isolines of constant length scale
k0          = np.array([2,4,6,8,10,12,14,16,18])
lat0        = 0
lats        = np.arange(lat0,91,1)
wavenumbers = np.zeros((k0.size,lats.size))
for i in range(0,k0.size,1): 
    for j in range(0,lats.size,1):
        wavelength = 2*np.pi*const.Re*np.cos(np.radians(lat0))/k0[i]
        wavenumbers[i,j] = 2*np.pi*const.Re*np.cos(np.radians(lats[j]))/wavelength

# Plotting
fontsize = 12
figsize  = np.array([7.245,4.5])

if var == 'K':
    cmap     = plt.cm.get_cmap("Reds")
    cmap.set_under('white')
    clabel   = r'10$^{12}$ Jm$^{-2}$'
    clevs    = np.arange(0.5,6.5,0.5)
    cskip    = 1
    data     = data/10**12
elif var == 'M':
    cmap     = plt.cm.get_cmap("RdBu_r")
    clabel   = r'10$^{12}$ Jm$^{-2}$'
    clevs    = np.arange(-6,6.5,0.5)
    cskip    = 2
    data     = data/10**12
elif var == 'N':
    cmap     = plt.cm.get_cmap("RdBu_r")
    clabel   = r'10$^{11}$ Jm$^{-2}$'
    clevs    = np.arange(-14,16,2)
    cskip    = 1
    data     = data/10**11
if var == 'alpha':
    cmap     = plt.cm.get_cmap("RdBu_r")
    clabel   = r'unitless'
    clevs    = np.arange(-1,1.2,0.2)
    cskip    = 1


[fig,ax] = plt.subplots()
fig.set_size_inches(figsize[0],figsize[1])

plt.contour(wavenumber,dim.lat,data,levels=clevs,colors='grey',linewidths=1.5,linestyles='-')
plt.contourf(wavenumber,dim.lat,data,levels=clevs,cmap=cmap,extend='both')
for i in range(0,k0.size,1):
    plt.plot(wavenumbers[i,:],lats,'--k',linewidth=0.75)

plt.yticks(np.arange(0,100,10),np.arange(0,100,10),fontsize = fontsize)
plt.xticks(wavenumber,wavenumber,fontsize = fontsize)
plt.xlabel('zonal wavenumber (k)',fontsize = fontsize)
plt.ylabel('latitude (deg)',fontsize = fontsize)
cb = plt.colorbar(ticks=clevs[0::cskip])
cb.set_label(clabel,fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize,size=0)

ax.set_ylim([0,90])
ax.set_xlim([wavenumber[0],wavenumber[-1]])
plt.tight_layout()
if outputdata:
    if no_stw_flag == 1:
        outputfilename = figdir + 'yk_' + var + '_' + 'no_stw' + '_' + season + '_' + \
                         str(year[0]) + '-' + str(year[-1]) + '.pdf'
    else:
        outputfilename = figdir + 'yk_' + var + '_' + season + \
                         '_' + str(year[0]) + '-' + str(year[-1]) + '.pdf'
    plt.savefig(outputfilename)
plt.show()
