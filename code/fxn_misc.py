# collection of useful miscellaneous functions

def tic():
    # Homemade version of matlab tic function   
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
     # Homemade version of matlab tic function 
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def leap_year_test(year):
    """                             
    Determine if year is a leap year   
    output type of calendar (leap or noleap)  
    """
    leap = 'noleap'
    if (year % 4 == 0):
        leap = 'leap'
    elif (year % 100 == 0) and (year % 400 != 0):
        leap = 'noleap'
    return leap


def compute_dp_ml(sp,dim):
    """   
    calculates the difference in pressure across a hybrid-sigma model level 
    as a function of model level,lat,lon and time
    output dp in Pa units 
    """
    import numpy as np

    if np.ndim(sp) == 2:
        # compute pressure on half levels
        ph = np.zeros((sp.shape[0],dim.ilev.size,dim.lat.size))
        dp = np.zeros((sp.shape[0],dim.lev.size,dim.lat.size))

        for k in range(0,dim.ilev.size):
            ph[:,k,:] = dim.hyai[k] + dim.hybi[k]*sp[:,:]

            # take the difference between half pressure levels = full level thickness 
            for k in range(0,dim.ilev.size):
                if k > 0:
                    dp[:,k-1,:] = ph[:,k,:] - ph[:,k-1,:]

    if np.ndim(sp) == 3:
        # compute pressure on half levels
        ph = np.zeros((sp.shape[0],dim.ilev.size,dim.lat.size,dim.lon.size))
        dp = np.zeros((sp.shape[0],dim.lev.size,dim.lat.size,dim.lon.size))

        for k in range(0,dim.ilev.size):
            ph[:,k,:,:] = dim.hyai[k] + dim.hybi[k]*sp[:,:,:]

            # take the difference between half pressure levels = full level thickness 
            for k in range(0,dim.ilev.size):
                if k > 0:
                    dp[:,k-1,:,:] = ph[:,k,:,:] - ph[:,k-1,:,:]

    return dp


def compute_fft_amp(data,ax,wavenumber):
    """                                                   
    Calculates sin and cos fourier amplitude coefficients                        
    """
    import numpy as np

    xxx = np.fft.fft(data,axis=ax)/data.shape[ax]
    A   = 2*np.real(xxx)
    B   = -2*np.imag(xxx)

    if (data.ndim == 4) & (ax == 3):
        A = A[:,:,:,wavenumber]
        B = B[:,:,:,wavenumber]

    return A,B


def dayofyear_to_month(dayofyear):
    """          
    maps day of year (1-365) onto a months (1-12)            
    """
    import numpy as np
    daysinmonths = np.array([31,28,31,30,31,30,31,31,30,31,30,31])

    dayofyear = dayofyear + 1 # convert to dayofyear (1-365) from (0-364)                

    if dayofyear > 0 and dayofyear <= 31:
        month = 1
    else:
        for i in range(1,12):
            if (dayofyear > np.sum(daysinmonths[0:i])) and (dayofyear <= np.sum(daysinmonths[0:i+1])) :
                month = i + 1

    return month




def months_to_dayofyear_index(month):

    # input month string and output dayofyear indicies
    # corresponding to those months
    
    import numpy as np

    dayofyear = np.arange(0,365,1)

    if month == 'JAN':
        index = dayofyear[0:31]
    elif month == 'FEB':
        index = dayofyear[31:59]
    elif month == 'MAR':
        index = dayofyear[59:90]
    elif month == 'DEC':
        index = dayofyear[334:]
    elif month == 'DJFM':
        temp = np.roll(dayofyear,31,axis=0)
        index = temp[0:121]
    elif month == 'JJA':
        index = dayofyear[151:243]
    elif month == 'JJAS':
        index = dayofyear[151:243+30]
    elif month == 'SON':
        index = dayofyear[243:334]
    elif month == 'MAM':
        index = dayofyear[59:151]
    elif month == 'JFM':
        index = dayofyear[0:90]
    elif month == 'DJF':
        temp = np.roll(dayofyear,31,axis=0)
        index = temp[0:90]
    elif month == 'ANNUAL':
        index = dayofyear

    return index


def get_season(data,season):
    """
    extracts season from long time series reanlaysis data
    note: time dimension must be first one
    """
    import numpy as np

    dayofyear = np.arange(0,365,1)

    if season == 'JAN':
        index = dayofyear[0:31]
    elif season == 'FEB':
        index = dayofyear[31:59]
    elif season == 'MAR':
        index = dayofyear[59:90]
    elif season == 'DEC':
        index = dayofyear[334:]
    elif season == 'DJFM':
        temp  = np.roll(dayofyear,31,axis=0)
        index = temp[0:121]
    elif season == 'NDJFM':
        temp  = np.roll(dayofyear,61,axis=0)
        index = temp[0:151]
    elif season == 'MJJAS':
        index = dayofyear[120:273]
    elif season == 'JJA':
        index = dayofyear[151:243]
    elif season == 'JJAS':
        index = dayofyear[151:243+30]
    elif season == 'SON':
        index = dayofyear[243:334]
    elif season == 'MAM':
        index = dayofyear[59:151]
    elif season == 'JFM':
        index = dayofyear[0:90]
    elif season == 'DJF':
        temp  = np.roll(dayofyear,31,axis=0)
        index = temp[0:90]
    elif season == 'ANNUAL':
        index = dayofyear

    if np.ndim(data) == 1:
        nyr     = int(data.shape[0]/365)
        data    = np.reshape(data,(nyr,365))
        data    = data[:,index]
        data    = np.reshape(data,(nyr*index.size))
    elif np.ndim(data) == 2:
        nyr     = int(data.shape[0]/365)
        ndim1   = data.shape[1]
        data    = np.reshape(data,(nyr,365,ndim1))
        data    = data[:,index,:]
        data    = np.reshape(data,(nyr*index.size,ndim1))
    elif np.ndim(data) == 3:
        nyr     = int(data.shape[0]/365)
        ndim1   = data.shape[1]
        ndim2   = data.shape[2]
        data    = np.reshape(data,(nyr,365,ndim1,ndim2))
        data    = data[:,index,:,:]
        data    = np.reshape(data,(nyr*index.size,ndim1,ndim2))

    return data


def smth9(array,p,q):
    # NCL equivalent 9-point smoothing function in python
                                                                                                                           
    import numpy as np

    dim  = array.shape
    temp = array

    for i in range(1,dim[0]-1):
        for j in range(1,dim[1]-1):
            temp[i,j] = array[i,j] + (p/4)*( array[i-1,j] + array[i,j-1] + array[i+1,j] + array[i,j+1] - 4*array[i,j] ) + (q/4)*( array[i-1,j+1] + array[i-1,j-1] + array[i+1,j+1] + array[i+1,j-1] - 4*array[i,j] )

    array = temp

    return array



def smooth(x,window,option):
    """                                                         
    Smoother for given window and numpy input array   
    Smoothing dimension always first one   
    """
    import numpy as np

    xmean = np.zeros(x.shape)

    if window > 0:
        if option == 'running':

            if np.ndim(x) == 1:
                for t in range(0,x.shape[0]):
                    if (t >= (window-1)/2) & (t <= x.size - (window-1)/2 - 1):
                        xmean[t] = np.mean(x[int(t-(window-1)/2):int(t+(window-1)/2+1)])
                    else:
                        xmean[t] = x[t]
            elif np.ndim(x) == 2:
                for t in range(0,x.shape[0]):
                    if (t >= (window-1)/2) & (t <= x.size - (window-1)/2 - 1):
                        xmean[t,:] = np.mean(x[int(t-(window-1)/2):int(t+(window-1)/2+1),:],axis=0)
                    else:
                        xmean[t,:] = x[t,:]
            elif np.ndim(x) == 3:
                for t in range(0,x.shape[0]):
                    if (t >= (window-1)/2) & (t <= x.size - (window-1)/2 - 1):
                        xmean[t,:,:] = np.mean(x[int(t-(window-1)/2):int(t+(window-1)/2+1),:,:],axis=0)
                    else:
                        xmean[t,:,:] = x[t,:,:]

    else:
        xmean = x

    return xmean



def cross_correlation(a,b,maxlag):
    """ 
    standard cross-correlation for specific lags
    """
    import numpy as np

    a  = (a - np.mean(a)) / (np.std(a) * len(a))
    b  = (b - np.mean(b)) / (np.std(b))
    cc = np.correlate(a, b, 'full')
    cc = cc[a.size-(maxlag+1):a.size+maxlag]

    return cc



def lag_regression(data1,data2,lag,sigthresh):
    """                                                                                                                                                                              
    regresses daily data1 onto data2 and outputs coefficient + significance                                                                                                          
    """
    import numpy as np
    from scipy   import stats

    if np.ndim(data2) == 2:

        ndim1 = data2.shape[1]
        coeff = np.zeros((lag.size,ndim1))
        sig   = np.zeros((lag.size,ndim1))

        for j in range(0,ndim1):
            for ilag in range(0,lag.size):
                if lag[ilag] == 0:
                    x = data2[:,j]
                    y = data1
                elif lag[ilag] > 0:
                    x = np.roll(data2[:,j],-lag[ilag])
                    x = x[0:-lag[ilag]]
                    y = data1[0:-lag[ilag]]
                elif lag[ilag] < 0:
                    y = np.roll(data1,lag[ilag])
                    x = data2[0:lag[ilag],j]
                    y = y[0:lag[ilag]]

                [coeff[ilag,j],dum1,dum2,sig[ilag,j],dum3] = stats.linregress(np.squeeze(y),x)

    elif np.ndim(data2) == 3:

        ndim1 = data2.shape[1]
        ndim2 = data2.shape[2]
        coeff = np.zeros((lag.size,ndim1,ndim2))
        sig   = np.zeros((lag.size,ndim1,ndim2))

        for j in range(0,ndim2):
            for k in range(0,ndim1):
                for ilag in range(0,lag.size):
                    if lag[ilag] == 0:
                        x = data2[:,k,j]
                        y = data1
                    elif lag[ilag] > 0:
                        x = np.roll(data2[:,k,j],-lag[ilag])
                        x = x[0:-lag[ilag]]
                        y = data1[0:-lag[ilag]]
                    elif lag[ilag] < 0:
                        y = np.roll(data1,lag[ilag])
                        x = data2[0:lag[ilag],k,j]
                        y = y[0:lag[ilag]]

                    [coeff[ilag,k,j],dum1,dum2,sig[ilag,k,j],dum3] = stats.linregress(np.squeeze(y),x)

    index       = np.where(sig <= sigthresh)
    index2      = np.where(sig > sigthresh)
    sig[index]  = 1
    sig[index2] = 0

    return coeff,sig


 

def get_anomaly(data):
    """
    removes the climatological seasonal cycle from reanalysis data.
    note: time dimension must be first dimension
    """
    import numpy as np 
    
    if np.ndim(data) == 1:
        nyr     = int(data.shape[0]/365)
        data    = np.reshape(data,(nyr,365))
        scycle  = np.mean(data,axis=0)  
        for t in range(0,nyr):      
            data[t,:] = data[t,:] - scycle[:] 
        data = np.reshape(data,(365*nyr,1))

    elif np.ndim(data) == 2:
        nyr     = int(data.shape[0]/365)
        ndim1   = data.shape[1]
        data    = np.reshape(data,(nyr,365,ndim1))
        scycle  = np.mean(data,axis=0)
        for t in range(0,nyr):
            data[t,:,:] = data[t,:,:] - scycle[:,:]
        data = np.reshape(data,(365*nyr,ndim1))
    elif np.ndim(data) == 3:
        nyr     = int(data.shape[0]/365)
        ndim1   = data.shape[1]
        ndim2   = data.shape[2]
        data    = np.reshape(data,(nyr,365,ndim1,ndim2))
        scycle  = np.mean(data,axis=0)
        for t in range(0,nyr):
            data[t,:,:] = data[t,:,:,:] - scycle[:,:,:]
        data = np.reshape(data,(365*nyr,ndim1,ndim2))

    return data


def ttest_mean_1samp(data,pcrit,mean,axis):
    """
    two tailed students t-test testng mean different from zero
    """
    import numpy as np
    from scipy   import stats

    [tstat,sig] = stats.ttest_1samp(data,mean,axis=axis)
    index1      = np.where(sig <= pcrit)
    index2      = np.where(sig > pcrit)
    sig[index1] = 1
    sig[index2] = 0

    return sig



def AxRoll(x,ax,invert=False):
    """
    Re-arrange array x so that axis 'ax' is first dimension.
    Undo this if invert=True 
    """
    import numpy as np

    if ax < 0:
        n = len(x.shape) + ax
    else:
        n = ax

    if invert is False:
        y = np.rollaxis(x,n,0)
    else:
        y = np.rollaxis(x,0,n+1)

    return y



def get_waves(input,wave=-1,ax=-1):
    """      
    extracts given wavenumber from given axis of input array
    """
    import numpy as np

    x             = AxRoll(input,ax)

    if wave < 0: # all waves 
        zmean = np.mean(x,0)
        for i in range(0,x.shape[0],1):
            x[i,:] = x[i,:] - zmean[:]
        output = AxRoll(x,ax,invert=True)
        
    else: # given wavenumber
        x    = np.fft.fft(x,axis=0)
        mask = np.zeros_like(x)
        if mask.ndim == 1:
            mask[wave]  = 1
            mask[-wave] = 1
        else:
            mask[wave,:]  = 1
            mask[-wave,:] = 1

        output = np.real(np.fft.ifft(x*mask,axis=0))
        output = AxRoll(output,ax,invert=True)

    return output


def find_nearest(array,value):
    """ 
    finds nearest value and index in array
    """
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx
