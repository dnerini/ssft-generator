#!/usr/bin/env python
"""
DESCRIPTION

    Simple implementation of the stochastic generator for radar rainfall fields based on the short-space Fourier transform as described in Nerini et al. (2017), "A non-stationary stochastic ensemble generator for radar rainfall
    fields based on the short-space Fourier transform", https://doi.org/10.5194/hess-21-2777-2017.

AUTHOR

    Daniele Nerini <daniele.nerini@gmail.com>

VERSION

    1.0
"""
from __future__ import division
from PIL import Image
import numpy as np

def ssft_generator(rainField, winsize=128, overlap=0.5, wintype = 'flat-hanning', war_thr=0.1): 

    '''
    Function to compute the locally correlated noise using SSFT.
    Please note that this simple implementation will fill with NaNs those regions
    that have not enough wet pixels within the window. 
    
    Parameters
    ----------
    rainField : numpyarray(float)
        Input 2d array with the rainfall field (or any kind of image)
    winsize : int
        Size-length of the window to compute the SSFT.
    overlap : float [0,1[ 
        The proportion of overlap to be applied between successive windows.
    win_type : string ['hanning', 'flat-hanning'] 
        Type of window used for localization.
    war_thr : float [0;1]
        Threshold for the minimum fraction of rain needed for computing the FFT.   
    '''
    
    # Rain/no-rain threshold
    norain = np.nanmin(rainField)

    # Define the shift of the window based on the overlap parameter
    if winsize == rainField.shape[0]:
        overlap = 0
    delta = int(winsize*(1 - overlap))
    delta = np.max((delta,1))
        
    # Set the seed and generate a field of white noise
    np.random.seed(42)
    randValues = np.random.randn(rainField.shape[0],rainField.shape[1])
        
    # Compute FFT of the noise field
    fnoise = np.fft.fft2(randValues)
         
    # Initialise variables
    idxi = np.zeros((2,1)); idxj = np.zeros((2,1))
    maskSum = np.zeros(randValues.shape)
    corrNoiseTotal = np.zeros(randValues.shape)
    
    # Loop rows  
    for i in range(0,rainField.shape[0],delta):
    
        # Loop columns
        for j in range(0,rainField.shape[1],delta):
            
            # Set window indices
            idxi[0] = i
            idxi[1] = int(np.min((i + winsize, rainField.shape[0])))
            idxj[0] = j
            idxj[1] = int(np.min((j + winsize, rainField.shape[1])))
 
            # Build window
            wind = _build_2D_window(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])), wintype)

            # At least 1/2 of the window within the frame
            if (wind.shape[0]*wind.shape[1])/winsize**2 > 1/2: 
                
                # Build mask based on window
                mask = np.zeros(rainField.shape) 
                mask[int(idxi[0]):int(idxi[1]),int(idxj[0]):int(idxj[1])] = wind
                
                # Apply the mask
                rmask = mask*rainField

                # Continue only if enough precip within window
                if  np.sum(rmask>norain)/wind.size > war_thr:

                    # Get fft of the windowed rainfall field
                    fftw = np.fft.fft2(rmask)
                    
                    # Normalize the spectrum
                    fftw.imag = ( fftw.imag - np.mean(fftw.imag) ) / np.std(fftw.imag)
                    fftw.real = ( fftw.real - np.mean(fftw.real) ) / np.std(fftw.real)
                    
                    # Keep only the amplitude spectrum (its absolute value)
                    fftw = np.abs(fftw)
                    
                    # Convolve the local spectra white the noise field (multiply the two spectra)
                    fcorrNoise = fnoise*fftw
                    
                    # Do the inverse FFT
                    corrNoise = np.fft.ifft2(fcorrNoise)
                    corrNoiseReal = np.array(corrNoise.real)
                    
                    # Merge the local fields
                    corrNoiseTotal += corrNoiseReal*mask
                    
                    # Update the sum of weights for later normalization
                    maskSum += mask

    # Normalize the sum
    idx = maskSum>0
    corrNoiseTotal[idx] = corrNoiseTotal[idx]/maskSum[idx]
    
    # Standardize the field
    corrNoiseTotal[idx] = ( corrNoiseTotal[idx] - np.mean(corrNoiseTotal[idx]) ) / np.std(corrNoiseTotal[idx])
    
    # Add NaNs
    corrNoiseTotal[~idx] = np.nan

    return corrNoiseTotal

def nested_generator(target, nr_frames = 1, max_level = 3, win_type = 'flat-hanning', war_thr = 0.1, overlap = 40, do_set_seed = True, do_plot = False):

    '''
    Function to compute the locally correlated noise using a nested approach.
    
    Parameters
    ----------
    target : numpyarray(float)
        Input 2d array with the rainfall field (or any kind of image)
    nr_frames : int
        Number of noise fields to produce.
    max_level : int 
        Localization parameter. 0: global noise, >0: increasing degree of localization.
    win_type : string ['hanning', 'flat-hanning'] 
        Type of window used for localization.
    war_thr : float [0;1]
        Threshold for the minimum fraction of rain needed for computing the FFT.
    overlap : int [px]
        Number of pixels that overlap between windows, helps in producing smoother fields.
    do_set_seed : bool
        Set the seed for the random number generator.
    do_plot : bool
        Plot the noise fields.
    '''
    
    # make sure non-rainy pixels are set to zero
    min_value = np.min(target)
    orig_target = target
    target -= min_value
    
    # store original field size
    orig_dim = target.shape
    orig_dim_x = orig_dim[1]
    orig_dim_y = orig_dim[0]
    
    
    # apply window to the image to limit spurious edge effects
    orig_window = _build_2D_window(orig_dim,win_type)
    target = target*orig_window
    
    # now buffer the field with zeros to get a squared domain       <-- need this at the moment for the nested approach, but I guess we could try to avoid it
    dim_x = np.max(orig_dim) 
    dim_y = dim_x
    dim = (dim_y,dim_x)
    ztmp = np.zeros(dim)
    if(orig_dim[1] > dim_x):
        idx_buffer = round((dim_x - orig_dim_x)/2)
        ztmp[:,idx_buffer:(idx_buffer + orig_dim_x)] = z
        z=ztmp 
    elif(orig_dim[0] > dim_y):
        idx_buffer = round((dim_y - orig_dim_y)/2)
        ztmp[idx_buffer:(idx_buffer + orig_dim_y),:] = z
        z=ztmp 
    # else do nothing
    
    ## Nested algorithm
    
    # prepare indices
    Idxi = np.array([[0,dim_y]])
    Idxj = np.array([[0,dim_x]])
    Idxipsd = np.array([[0,2**max_level]])
    Idxjpsd = np.array([[0,2**max_level]])
    
    # generate the FFT sample frequencies
    res_km = 1 
    freq = _get_fftfreq(dim_x, res_km)
    fx,fy = np.meshgrid(freq,freq)
    freq_grid = np.sqrt(fx**2 + fy**2)
    
    # get global fourier filter
    mfilter0 = _get_fourier_filter(target)
    # and allocate it to the final grid
    mfilter = np.zeros((2**max_level,2**max_level,mfilter0.shape[0],mfilter0.shape[1]))
    mfilter += mfilter0[np.newaxis,np.newaxis,:,:]
    
    # now loop levels and build composite spectra
    level=0 
    while level < max_level:

        for m in xrange(len(Idxi)):
        
            # the indices of rainfall field
            Idxinext,Idxjnext = _split_field(Idxi[m,:],Idxj[m,:],2)
            # the indices of the field of fourier filters
            Idxipsdnext,Idxjpsdnext = _split_field(Idxipsd[m,:],Idxjpsd[m,:],2)
            
            for n in xrange(len(Idxinext)):
                mask = _get_mask(dim[0],Idxinext[n,:],Idxjnext[n,:],win_type)
                war = np.sum((target*mask)>0)/(Idxinext[n,1]-Idxinext[n,0])**2 
                
                if war>war_thr:
                    # the new filter 
                    newfilter = _get_fourier_filter(target*mask)
                    
                    # compute logistic function to define weights as function of frequency
                    # k controls the shape of the weighting function
                    merge_weights = _logistic_function(1/freq_grid, k=0.05, x0 = (Idxinext[n,1] - Idxinext[n,0])/2)
                    newfilter = newfilter*(1 - merge_weights)
                    
                    # perform the weighted average of previous and new fourier filters
                    mfilter[Idxipsdnext[n,0]:Idxipsdnext[n,1],Idxjpsdnext[n,0]:Idxjpsdnext[n,1],:,:] *= merge_weights[np.newaxis,np.newaxis,:,:]
                    mfilter[Idxipsdnext[n,0]:Idxipsdnext[n,1],Idxjpsdnext[n,0]:Idxjpsdnext[n,1],:,:] += newfilter[np.newaxis,np.newaxis,:,:] 
                    
        # update indices
        level += 1
        Idxi, Idxj = _split_field((0,dim[0]),(0,dim[1]),2**level)
        Idxipsd, Idxjpsd = _split_field((0,2**max_level),(0,2**max_level),2**level)
        
    ## Power-filter images

	# produce normal noise array
    if do_set_seed: 
        np.random.seed(42)
	white_noise = np.random.randn(dim[0],dim[1],nr_frames)
    
    # build composite image of correlated noise
    corr_noise = np.zeros((dim_y,dim_x,nr_frames))
    sum_of_masks = np.zeros((dim_y,dim_x,nr_frames))
    idxi = np.zeros((2,1),dtype=int)
    idxj = np.zeros((2,1),dtype=int)
    winsize = np.round( dim[0] / 2**max_level )
    
    # loop frames
    for m in xrange(nr_frames):
    
        # get fourier spectrum of white noise field
        white_noise_ft = np.fft.fft2(white_noise[:,:,m])
    
        # loop rows
        for i in xrange(2**max_level):
            # loop columns
            for j in xrange(2**max_level):

                # apply fourier filtering with local filter
                this_filter = mfilter[i,j,:,:]
                this_corr_noise_ft = white_noise_ft * this_filter
                this_corr_noise = np.fft.ifft2(this_corr_noise_ft)
                this_corr_noise = np.array(this_corr_noise.real)
                
                # compute indices of local area
                idxi[0] = np.max( (np.round(i*winsize - overlap/2), 0) )
                idxi[1] = np.min( (np.round(idxi[0] + winsize  + overlap/2), dim[0]) )
                idxj[0] = np.max( (np.round(j*winsize - overlap/2), 0) )
                idxj[1] = np.min( (np.round(idxj[0] + winsize  + overlap/2), dim[1]) )
                
                # build mask and add local noise field to the composite image
                mask = _get_mask(dim[0],idxi,idxj,win_type)
                corr_noise[:,:,m] += this_corr_noise*mask
                sum_of_masks[:,:,m] += mask
                
    # normalize the sum
    idx = sum_of_masks > 0
    corr_noise[idx] = corr_noise[idx] / sum_of_masks[idx]
    
    # crop the image back to the original size
    difx = dim_x - orig_dim_x
    dify = dim_y - orig_dim_y
    output = corr_noise[int(dify/2):int(dim_y-dify/2),int(difx/2):int(dim_x-difx/2),:]
    
    # standardize the results to N(0,1)
    for m in xrange(nr_frames):
        output[:,:,m]  -= np.mean(output[:,:,m])
        output[:,:,m]  /= np.std(output[:,:,m])
    
    if do_plot:
        for m in xrange(nr_frames):
            plt.clf()
            plt.subplot(121)
            plt.imshow(target,interpolation='nearest')
            plt.subplot(122)
            plt.imshow(output[:,:,m],interpolation='nearest',vmin=-3.5,vmax=3.5)
            plt.pause(1)
   
    return output    
    
def read_data(fileName):

    '''
    Function to read the wheater radar fields.
    
    Parameters
    ----------
    fileName : str
        filename of the .gif weather radar image.
    '''
    
    # Open gif image
    rainImg = Image.open(fileName)
    nrCols = rainImg.size[0]
    nrRows = rainImg.size[1]
    rain8bit = np.array(rainImg,dtype=int)
    
    # Select 512x512 domain in the middle
    width=512;height=512
    borderSizeX = int( (rain8bit.shape[1] - height)/2 )
    borderSizeY = int( (rain8bit.shape[0] - width)/2 )
    mask = np.ones((rain8bit.shape))
    mask[0:borderSizeY,:] = 0
    mask[borderSizeY+height:,:] = 0
    mask[:,0:borderSizeX] = 0
    mask[:,borderSizeX+width:] = 0
    rain8bit = rain8bit[mask==1].reshape(height,width)
    
    # Generate lookup table
    noData = -999.0
    lut = _get_rainfall_lookuptable(noData)
    
    # Replace 8bit values with rain accumulations over 5 minutes
    rainrate = lut[rain8bit]
    
    # Convert to mm h-1
    rainrate[rainrate != noData] = rainrate[rainrate != noData]*(60/5)
    
    # Fills no-rain with zeros
    rainThreshold = 0.08 # mm h-1
    condition = (rainrate < rainThreshold) & (rainrate > 0.0)
    rainrate[condition] = 0.0
    
    # Convert rainrate to reflectivity
    dBZ = rainrate.copy()
    rainIdx = rainrate > 0
    zerosIdx = rainrate == 0
    # rainy pixels are converted to dBZ
    dBZ[rainIdx] = 10.0*np.log10(316*rainrate[rainIdx]**1.5)
    # we also subtract the rain threshold in order to reduce the discontinuity
    dBZ[rainIdx] = dBZ[rainIdx] - 10.0*np.log10(316*rainThreshold**1.5)
    # no-rainy pixel are set to zero
    dBZ[zerosIdx] = 0
    
    # Fills missing data with zeros 
    condition = dBZ == noData
    dBZ[condition] = 0
    
    return dBZ
 
def _build_2D_window(winsize, wintype='flat-hanning'):

    # Build 1-D window for rows and columns
    if wintype == 'hanning':
        w1dr = np.hanning(winsize[0])
        w1dc = np.hanning(winsize[1])  
    elif wintype == 'flat-hanning':
    
        T = winsize[0]/4
        W = winsize[0]/2
        B = np.linspace(-W,W,2*W)
        R = np.abs(B)-T
        R[R<0]=0.
        A = 0.5*(1.0 + np.cos(np.pi*R/T))
        A[np.abs(B)>(2*T)]=0.0
        w1dr = A
        
        T = winsize[1]/4
        W = winsize[1]/2
        B=np.linspace(-W,W,2*W)
        R = np.abs(B)-T
        R[R<0]=0.
        A = 0.5*(1.0 + np.cos(np.pi*R/T))
        A[np.abs(B)>(2*T)]=0.0
        w1dc = A   
        
    else:
        print("Unknown window type, returning a rectangular window.")
        w1dr = np.ones(winsize[0])
        w1dc = np.ones(winsize[1])
    
    # Expand to 2-D
    w2d = np.sqrt(np.outer(w1dr,w1dc))
    
    # Set nans to zero
    if np.sum(np.isnan(w2d))>0:
        w2d[np.isnan(w2d)]=np.min(w2d[w2d>0])

    return w2d
    
def _get_rainfall_lookuptable(noData):
    precipIdxFactor=71.5
    lut = np.zeros(256)
    for i in range(0,256):
        if (i < 2) or (i > 250 and i < 255):
            lut[i] = 0.0
        elif (i == 255):
            lut[i] = noData
        else:
            lut[i] = (10.**((i-(precipIdxFactor))/20.0)/316)**(1.0/1.5)
    
    return lut
    
def _get_fourier_filter(fieldin, do_norm = True):

    # FFT of the field
    fftw = np.fft.fft2(fieldin)
    
    # Normalize the real and imaginary parts
    if do_norm:
        fftw.imag = ( fftw.imag - np.mean(fftw.imag) ) / np.std(fftw.imag)
        fftw.real = ( fftw.real - np.mean(fftw.real) ) / np.std(fftw.real)
        
    # Extract the amplitude
    fftw = np.abs(fftw)

    return fftw  
    
def _split_field(idxi,idxj,Segments):

    sizei = (idxi[1] - idxi[0]) 
    sizej = (idxj[1] - idxj[0]) 
    
    winsizei = np.round( sizei / Segments )
    winsizej = np.round( sizej / Segments )
    
    Idxi = np.zeros((Segments**2,2))
    Idxj = np.zeros((Segments**2,2))
    
    count=-1
    for i in xrange(Segments):
        for j in xrange(Segments):
            count+=1
            Idxi[count,0] = idxi[0] + i*winsizei
            Idxi[count,1] = np.min( (Idxi[count,0] + winsizei, idxi[1]) )
            Idxj[count,0] = idxj[0] + j*winsizej
            Idxj[count,1] = min( (Idxj[count,0] + winsizej, idxj[1]) )

    Idxi = np.array(Idxi).astype(int); Idxj =  np.array(Idxj).astype(int)        
    return Idxi, Idxj
    
def _get_mask(Size,idxi,idxj,wintype):
    idxi = np.array(idxi).astype(int); idxj =  np.array(idxj).astype(int)
    winsize = (idxi[1] - idxi[0] , idxj[1] - idxj[0])
    wind = _build_2D_window(winsize,wintype)
    mask = np.zeros((Size,Size)) 
    mask[idxi.item(0):idxi.item(1),idxj.item(0):idxj.item(1)] = wind
    return mask

def _logistic_function(x, L = 1,k = 1,x0 = 0):
    return L/(1 + np.exp(-k*(x - x0)))
    
def _get_fftfreq(n, d=1.0):
    if n % 2 == 0:
        f = np.concatenate([np.arange(0,n/2), np.arange(-n/2,0)])/(d*n)
    else:
        f = np.concatenate([np.arange(0,(n-1)/2+1), np.arange(-(n-1)/2,0)])/(d*n)
    return f