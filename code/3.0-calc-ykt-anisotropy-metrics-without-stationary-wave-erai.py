"""
Removes monthly mean stationary wave component from daily anisotropy metrics
decomposed by wavenumber. 
NOTE: You can see why this is true by decomposing a flux e.g. QV by time-mean/transient +
zonal-mean/eddy
"""

import numpy               as np
import xarray              as xr
import dim
import const
from fxn_misc import  dayofyear_to_month 


# INPUT -----------------------------------------------------------
datadir          = '/Data/skd/scratch/edu061/ERAI/POSTPROCESSED/'
year             = np.arange(1979,2018,1)
outputdata       = False
# -----------------------------------------------------------------  
nyr  = year[-1]-year[0]+1
nday = nyr*365
# -----------------------------------------------------------------

# read data  
filename1  = datadir + 'ykt_anisotropy_metrics_stationary_wave_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
ds1        = xr.open_dataset(filename1)
K_STW      = ds1.K.values
M_STW      = ds1.M.values
N_STW      = ds1.N.values
wavenumber = ds1.wavenumber.values
ds1.close()

filename2 = datadir + 'ykt_anisotropy_metrics_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
ds2       = xr.open_dataset(filename2)
K         = ds2.K.values
M         = ds2.M.values
N         = ds2.N.values
ds2.close()

# remove time-mean stationary wave component for each month
K = np.reshape(K,(nyr,365,dim.lat.size,wavenumber.size))
M = np.reshape(M,(nyr,365,dim.lat.size,wavenumber.size))
N = np.reshape(N,(nyr,365,dim.lat.size,wavenumber.size))
for yr in range(0,nyr):
    print(year[yr])
    for t in range(0,365):
        K[yr,t,:,:] = K[yr,t,:,:] - K_STW[dayofyear_to_month(t)-1,:,:]
        M[yr,t,:,:] = M[yr,t,:,:] - M_STW[dayofyear_to_month(t)-1,:,:]
        N[yr,t,:,:] = N[yr,t,:,:] - N_STW[dayofyear_to_month(t)-1,:,:]
K = np.reshape(K,(nyr*365,dim.lat.size,wavenumber.size))
M = np.reshape(M,(nyr*365,dim.lat.size,wavenumber.size))
N = np.reshape(N,(nyr*365,dim.lat.size,wavenumber.size))

if outputdata:
    output = xr.Dataset(data_vars={'K':  (('day','lat','wavenumber'), K.astype(np.float32)),
                                   'M':  (('day','lat','wavenumber'), M.astype(np.float32)),
                                   'N':  (('day','lat','wavenumber'), N.astype(np.float32))},
                        coords={'day': np.arange(1,nday+1,1),'lat': dim.lat,'wavenumber': wavenumber})
    output.K.attrs['units'] = 'J/m'
    output.M.attrs['units'] = 'J/m'
    output.N.attrs['units'] = 'J/m'
    outputfilename          = 'ykt_anisotropy_metrics_without_stationary_wave_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
    output.to_netcdf(datadir + outputfilename)
