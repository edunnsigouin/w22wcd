"""
Calculates the zonally and vertically integrated anisotropy metrics K,M,N and alpha
discussed in Hoskins et al. 1983 JAS as a function of latitude and wavenumber for
ERA-interim reanalysis data.
The calculation is similar to the once used in Graversen and Burtu 2016 QJRMS 
where the mass flux is integrated into one of the zonal-wavenumber decomposed terms
e.g., v'T' ~ vg'T' where vg = v*dp/g
"""

import numpy         as np
import xarray        as xr
import const
import dim
from   fxn_misc      import tic,toc,leap_year_test,compute_fft_amp,compute_dp_ml

# INPUT -----------------------------------------------------------   
datadir1       = '/Data/skd/scratch/edu061/ERAI/model_level/'
datadir2       = '/Data/skd/scratch/edu061/ERAI/sfc/'
datadir3       = '/Data/skd/scratch/edu061/ERAI/POSTPROCESSED/'
year           = np.arange(1979,2018,1)
month          = np.arange(1,13,1)
wavenumber     = np.arange(0,41,1)
trop_only_flag = 0
outputdata     = False
# ----------------------------------------------------------------- 
nday           = year.size*365
# -----------------------------------------------------------------

# Initialize
K = np.zeros((nday,dim.lat.size,wavenumber.size))
M = np.zeros((nday,dim.lat.size,wavenumber.size))
N = np.zeros((nday,dim.lat.size,wavenumber.size))
c = 0 # counter

for iyear in range(0,year.size):
    for imonth in range(0,month.size):

        print(year[iyear], month[imonth])

        tic()

        # read data for each month 
        timestamp = str(year[iyear]) + '-' + format(month[imonth], "02")
        filename1 = datadir1 + 'v/' + 'v_' + timestamp + '.nc'
        filename2 = datadir1 + 'u/' + 'u_' + timestamp + '.nc'
        filename3 = datadir2 + 'lnsp/' + 'lnsp_' + timestamp + '.nc'
        if trop_only_flag == 1:
            ds1       = xr.open_dataset(filename1).sel(lev=slice(287.6383,1012.0494),ilev=slice(272.05,1013.25))
            ds2       = xr.open_dataset(filename2).sel(lev=slice(287.6383,1012.0494),ilev=slice(272.05,1013.25))
            ds3       = xr.open_dataset(filename3)
        else:
            ds1       = xr.open_dataset(filename1)
            ds2       = xr.open_dataset(filename2)
            ds3       = xr.open_dataset(filename3)

        day       = ds1.day.values
        dim.hyai  = ds1.hyai.values
        dim.hybi  = ds1.hybi.values
        dim.lev   = ds1.lev.values
        dim.ilev  = ds1.ilev.values
        V         = ds1.v.values
        U         = ds2.u.values
        PS        = np.exp(ds3.lnsp.values)
        ds1.close()
        ds2.close()
        ds3.close()


        # remove leap year day         
        if (leap_year_test(year[iyear]) == 'leap') & (month[imonth] == 2):
            V    = V[:28,:,:,:]
            U    = U[:28,:,:,:]
            PS   = PS[:28,:,:]
            day  = day[:28]
        
        # mass at each level
        dp    = compute_dp_ml(PS,dim)
        
        # incorporate mass weighting into calculation
        Vstar = V*dp/const.go
        Ustar = U*dp/const.go

        # zonal fourier amplitudes                                                                                          
        V_A            = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))
        V_B            = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))
        U_A            = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))
        U_B            = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))
        Vstar_A        = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))
        Vstar_B        = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))
        Ustar_A        = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))
        Ustar_B        = np.zeros((day.size,dim.lev.size,dim.lat.size,wavenumber.size))

        [V_A,V_B]         = compute_fft_amp(V,3,wavenumber)
        [U_A,U_B]         = compute_fft_amp(U,3,wavenumber)
        [Vstar_A,Vstar_B] = compute_fft_amp(Vstar,3,wavenumber)
        [Ustar_A,Ustar_B] = compute_fft_amp(Ustar,3,wavenumber)

        # zonally and vertically integrated K,M & N
        K_short = 0.5*(U_A*Ustar_A + U_B*Ustar_B + V_A*Vstar_A + V_B*Vstar_B)
        M_short = 0.5*(U_A*Ustar_A + U_B*Ustar_B - V_A*Vstar_A - V_B*Vstar_B)
        N_short = U_A*Vstar_A + U_B*Vstar_B

        # Sum over all levels
        K_short = K_short.sum(axis=1)                            
        M_short = M_short.sum(axis=1)
        N_short = N_short.sum(axis=1)
        d       = 2*np.pi*const.Re*np.cos(np.radians(dim.lat))
        for j in range(0,dim.lat.size):
            K_short[:,j,0]  = K_short[:,j,0]*d[j]/4 # zonal-mean (k=0)                      
            M_short[:,j,0]  = M_short[:,j,0]*d[j]/4
            N_short[:,j,0]  = N_short[:,j,0]*d[j]/4

            K_short[:,j,1:] = K_short[:,j,1:]*d[j]/2 # eddies (k>=1)                            
            M_short[:,j,1:] = M_short[:,j,1:]*d[j]/2
            N_short[:,j,1:] = N_short[:,j,1:]*d[j]/2

        # dump computation into long time series array 
        for t in range(0,day.size):
            K[c,:,:] = K_short[t,:,:]
            M[c,:,:] = M_short[t,:,:]
            N[c,:,:] = N_short[t,:,:]
            c        = c + 1

        toc()

# write to file
if outputdata:
    output = xr.Dataset(data_vars={'K':  (('day','lat','wavenumber'), K.astype(np.float32)),
                                   'M':  (('day','lat','wavenumber'), M.astype(np.float32)),
                                   'N':  (('day','lat','wavenumber'), N.astype(np.float32))},
                        coords={'day': np.arange(1,nday+1,1),'lat': dim.lat,'wavenumber': wavenumber})
    output.K.attrs['units'] = 'J/m'
    output.M.attrs['units'] = 'J/m'
    output.N.attrs['units'] = 'J/m'
    if trop_only_flag == 1:
        outputfilename = 'ykt_anisotropy_metrics_troposphere_only_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
    else:
        outputfilename = 'ykt_anisotropy_metrics_' + str(year[0]) + '-' + str(year[-1]) + '.nc'
    output.to_netcdf(datadir3 + outputfilename)




