#!/usr/bin/env python
"""
DESCRIPTION

    Demonstrate the use of the stochastic generator for radar rainfall fields based on the short-space Fourier transform as described in Nerini et al. (2017), "A non-stationary stochastic ensemble generator for radar rainfall
    fields based on the short-space Fourier transform", doi:10.5194/hess-21-1-2017. 

AUTHOR

    Daniele Nerini <daniele.nerini@gmail.com>

VERSION

    1.0
"""

import matplotlib.pyplot as plt
from ssft_utils import ssft_generator, read_data


#+++++++++++ Select radar image

# case 1: 201503300520
# case 2: 201505151600
# case 3: 201506080230
# case 4: 201506151145

caseStudy = '201505151600'

#+++++++++++ Read the radar image

fileName = 'fig/radarField' + caseStudy + 'UTC.gif'
rainField_dBZ = read_data(fileName)

#+++++++++++ Generate spatially correlated noise based on radar image

# Apply global generator
winsize1 = rainField_dBZ.shape[0]
overlap1 = 0
corrNoiseGlobal = ssft_generator(rainField_dBZ, winsize=winsize1, overlap=overlap1)

# Apply local generator
winsize2 = 128
overlap2 = 0.5
corrNoiseLocal = ssft_generator(rainField_dBZ, winsize=winsize2, overlap=overlap2)

#+++++++++++ Plot the results

f,ax = plt.subplots(1,3)

# Reference field
ax[0].imshow(rainField_dBZ,cmap='Greys',interpolation='nearest',vmin=0,vmax=60)
ax[0].set_xticks([]);ax[0].set_yticks([]);ax[0].set_xticklabels([]);ax[0].set_yticklabels([])
ax[0].set_title('Rain analysis')

# Global noise
ax[1].imshow(corrNoiseGlobal,interpolation='nearest',vmin=-3.5,vmax=3.5)
ax[1].set_xticks([]);ax[1].set_yticks([]);ax[1].set_xticklabels([]);ax[1].set_yticklabels([])
ax[1].set_title('Global noise')
ax[1].text(0.02,0.98,'winsize = ' + str(winsize1) + '\noverlap = ' + str(overlap1),fontsize=11,transform=ax[1].transAxes,ha='left',va='top') 

# Local noise
ax[2].imshow(corrNoiseLocal,interpolation='nearest',vmin=-3.5,vmax=3.5)
ax[2].set_xticks([]);ax[2].set_yticks([]);ax[2].set_xticklabels([]);ax[2].set_yticklabels([])
ax[2].set_title('Local noise')
ax[2].text(0.02,0.98,'winsize = ' + str(winsize2) + '\noverlap = ' + str(overlap2),fontsize=11,transform=ax[2].transAxes,ha='left',va='top') 

plt.show()
