"""                                    
Calculates the zonally and vertically integrated anisotropy metrics K,M,N and alpha
discussed in Hoskins et al. 1983 JAS as a function of latitude and wavenumber for 
ERA-interim reanalysis data.  
The calculation is similar to the once used in Graversen and Burtu 2016 QJRMS
where the mass flux is integrated into one of the zonal-wavenumber decomposed terms 
e.g., v'T' ~ vg'T' where vg = v*dp/g
NOTE: this version of the code is for calculating the stationary wave part only                                                                 
"""


import numpy         as np
import xarray        as xr
import const
import dim
from   fxn_misc      import tic,toc,compute_dp_ml,compute_fft_amp
from matplotlib      import pyplot as plt

# INPUT -----------------------------------------------------------   
datadir1   = '/Data/skd/scratch/edu061/ERAI/model_level/'
datadir2   = '/Data/skd/scratch/edu061/ERAI/sfc/'
datadir3   = '/Data/skd/scratch/edu061/ERAI/POSTPROCESSED/'
year       = np.arange(1979,2018,1)
month      = np.arange(1,13,1)
wavenumber = np.arange(0,41,1)
outputdata = False
# ----------------------------------------------------------------- 

# initialize
V     = np.zeros((12,dim.lev.size,dim.lat.size,dim.lon.size))
U     = np.zeros((12,dim.lev.size,dim.lat.size,dim.lon.size))
Vstar = np.zeros((12,dim.lev.size,dim.lat.size,dim.lon.size))
Ustar = np.zeros((12,dim.lev.size,dim.lat.size,dim.lon.size))

# calculate monthly mean V and U
for iyear in range(0,year.size):
    for imonth in range(0,month.size):

        print(year[iyear], month[imonth])

        tic()

        filename1 = datadir1 + 'v/' + 'v_' + str(year[iyear]) + '-' + format(month[imonth], "02") + '.nc'
        filename2 = datadir1 + 'u/' + 'u_' + str(year[iyear]) + '-' + format(month[imonth], "02") + '.nc'
        filename3 = datadir2 + 'lnsp/' + 'lnsp_' + str(year[iyear]) +'-' + format(month[imonth], "02") + '.nc'
        da1       = xr.open_dataset(filename1)
        da2       = xr.open_dataset(filename2)
        da3       = xr.open_dataset(filename3)

        PS          = np.exp(da3.lnsp.values)
        dp          = compute_dp_ml(PS,dim)        
        dim.lev     = da1.lev.values
        dim.ilev    = da1.ilev.values
        dim.hyai    = da1.hyai.values
        dim.hybi    = da1.hybi.values
        V_short     = da1.v.values
        U_short     = da2.u.values
        Vstar_short = da1.v.values*dp/const.go
        Ustar_short = da2.u.values*dp/const.go

        da1.close()
        da2.close()
        da3.close()

        V[imonth,:,:,:]     = V[imonth,:,:,:]     + V_short.mean(axis=0)/year.size
        U[imonth,:,:,:]     = U[imonth,:,:,:]     + U_short.mean(axis=0)/year.size
        Vstar[imonth,:,:,:] = Vstar[imonth,:,:,:] + Vstar_short.mean(axis=0)/year.size
        Ustar[imonth,:,:,:] = Ustar[imonth,:,:,:] + Ustar_short.mean(axis=0)/year.size

        toc()

# ---- Zonal wavenumber decomposition of transport ----                                                        
# zonal fourier amplitudes                                                                                             
V_A     = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))
V_B     = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))
U_A     = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))
U_B     = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))
Vstar_A = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))
Vstar_B = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))
Ustar_A = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))
Ustar_B = np.zeros((12,dim.lev.size,dim.lat.size,wavenumber.size))

[V_A,V_B]         = compute_fft_amp(V,3,wavenumber)
[U_A,U_B]         = compute_fft_amp(U,3,wavenumber)
[Vstar_A,Vstar_B] = compute_fft_amp(Vstar,3,wavenumber)
[Ustar_A,Ustar_B] = compute_fft_amp(Ustar,3,wavenumber)

# vertically and zonally integrated anisotropy metrics M,K,N                                                          
K = 0.5*(U_A*Ustar_A + U_B*Ustar_B + V_A*Vstar_A + V_B*Vstar_B)
M = 0.5*(U_A*Ustar_A + U_B*Ustar_B - V_A*Vstar_A - V_B*Vstar_B)
N = U_A*Vstar_A + U_B*Vstar_B
K = K.sum(axis=1)
M = M.sum(axis=1)
N = N.sum(axis=1)
d = 2*np.pi*const.Re*np.cos(np.radians(dim.lat))
for j in range(0,dim.lat.size):
    K[:,j,0]  = K[:,j,0]*d[j]/4 # zonal-mean                                                  
    M[:,j,0]  = M[:,j,0]*d[j]/4
    N[:,j,0]  = N[:,j,0]*d[j]/4

    K[:,j,1:] = K[:,j,1:]*d[j]/2 # eddies 
    M[:,j,1:] = M[:,j,1:]*d[j]/2
    N[:,j,1:] = N[:,j,1:]*d[j]/2

# write to file                           
if outputdata:
    output = xr.Dataset(data_vars={'K':  (('month','lat','wavenumber'), K.astype(np.float32)),
                                   'M':  (('month','lat','wavenumber'), M.astype(np.float32)),
                                   'N':  (('month','lat','wavenumber'), N.astype(np.float32))},
                        coords={'month': np.arange(1,13,1),'lat': dim.lat,'wavenumber': wavenumber})
    output.K.attrs['units'] = 'J/m'
    output.M.attrs['units'] = 'J/m'
    output.N.attrs['units'] = 'J/m'
    outputfilename = 'ykt_anisotropy_metrics_stationary_wave_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
    output.to_netcdf(datadir3 + outputfilename)

