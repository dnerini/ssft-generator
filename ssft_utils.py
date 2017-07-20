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

def ssft_generator(rainField, winsize=128, overlap=0.5): 

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
            idxi[1] = np.min((i + winsize, rainField.shape[0]))
            idxj[0] = j
            idxj[1] = np.min((j + winsize, rainField.shape[1])) 
 
            # Build window
            wind = _build_2D_Hanning_window(((idxi[1]-idxi[0]),(idxj[1]-idxj[0])))

            # At least 1/2 of the window within the frame
            if (wind.shape[0]*wind.shape[1])/winsize**2 > 1/2: 
                
                # Build mask based on window
                mask = np.zeros(rainField.shape) 
                mask[idxi.item(0):idxi.item(1),idxj.item(0):idxj.item(1)] = wind
                
                # Apply the mask
                rmask = mask*rainField

                # Continue only if enough precip within window
                if  np.sum(rmask>norain)/wind.size > 0.05:

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
 
def _build_2D_Hanning_window(winsize):

    # Build 1-D window for rows
    w1dr = np.hanning(winsize[0])
    
    # Build 1-D window for columns
    w1dc = np.hanning(winsize[1])  
    
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
    
