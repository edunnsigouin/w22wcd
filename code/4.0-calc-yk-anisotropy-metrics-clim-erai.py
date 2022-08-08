# calculates climatological ky anisometry metrics 

import numpy               as np
import xarray              as xr
import dim
import const
from fxn_misc              import get_season     

# INPUT -----------------------------------------------------------   
datadir          = '/nird/projects/NS9625K/edu061/processed/erai/'
datadir2         = '/nird/home/edu061/w21/data/'
year             = np.arange(1979,2018,1)
wavenumber       = np.arange(1,14,1)
season           = 'ANNUAL'
no_stw_flag      = 1
outputdata       = False
# ----------------------------------------------------------------- 

# read data
if no_stw_flag == 1:
    filename = datadir + 'ykt_anisotropy_metrics_without_stationary_wave_' + str(year[0]) + '-' + str(year[-1]) + '.nc' 
else:
    filename = datadir + 'ykt_anisotropy_metrics_' + str(year[0]) + '-' + str(year[-1]) + '.nc' 
ds  = xr.open_dataset(filename).sel(wavenumber=wavenumber)
K   = ds.K.values
M   = ds.M.values
N   = ds.N.values

# seasonal climatology
K = get_season(K,season)
M = get_season(M,season)
N = get_season(N,season)
K = np.mean(K,axis=0)
M = np.mean(M,axis=0)
N = np.mean(N,axis=0)

# define alpha
alpha = M/K

ds.close()

# write to file      
if outputdata:
    output = xr.Dataset(data_vars={'K':  (('lat','wavenumber'), K.astype(np.float32)),
                                   'M':  (('lat','wavenumber'), M.astype(np.float32)),
                                   'N':  (('lat','wavenumber'), N.astype(np.float32)),
                                   'alpha':  (('lat','wavenumber'), alpha.astype(np.float32))},
			coords={'lat': dim.lat,'wavenumber': wavenumber})
    output.K.attrs['units']     = 'J/m'
    output.M.attrs['units']     = 'J/m'
    output.N.attrs['units']     = 'J/m'
    output.alpha.attrs['units'] = 'unitless'

    if no_stw_flag == 1:
        outputfilename = 'yk_anisotropy_metrics_without_stationary_wave_clim_' + season + '_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
    else:
        outputfilename = 'yk_anisotropy_metrics_clim_' + season + '_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
    output.to_netcdf(datadir2 + outputfilename)
