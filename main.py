'''
Main script to call to work with the optimization of the auroral boundaries
in the IMAFE FUV images. Specific helper functions is loaded helpers.py

'''

import numpy as np
import numpy as np
import glob
import helpers
from pysymmetry import fuvpy
import matplotlib.pyplot as plt
from pysymmetry.visualization import polarsubplot
from pathlib2 import Path
import xarray as xr


######################
### INITIALIZATION ###

#Global variables
datapath = '/Users/jone/BCSS-DAG Dropbox/Jone Reistad/Sara_PhD_data/Paper1/'
event = '2000-11-29' #'2000-11-24' #2000-11-29
dayglow_method = ('std',)
flatfield_issue = True
image_filtering = False

#Parameters determining binning, histograms, and boundary detection
latres = 0.5
mltres = 1
highlat = 85
stepsize = 5 # counts, for building histograms
thres_hist = 11 #threshold for setting boundary based on histigrams
############################
############################



############################
# Load images in event
filename = datapath+event + '/idl_saved_files/' + event + '.nc'
my_file = Path(filename)
if my_file.exists(): #Check if dayglow processed files exists
    wic = xr.open_dataset(filename)
else:
    wic = helpers.load_data(event, datapath, dayglow_method = dayglow_method, \
            flatfield_issue=flatfield_issue, image_filtering=False)
    wic.attrs = '' #need to delete attributes to be able to store. This is silly.
    wic.to_netcdf(filename)

############################
#Make an initial guess on the size of the oval, based on some course histograms
init_guess = helpers.bin_data(wic, LATRES=1, \
        MLTRES=1, LOWLAT_MIDNIGHT=40, \
        LOWLAT_NOON=40, HIGHLAT=85, INIT=True)
eqb_init_hist, pb_init_hist = helpers.make_histogram(init_guess, stepsize=5) #make histograms
eqb, pb = helpers.set_boundary(init_guess, eqb_init_hist, pb_init_hist, THRESHOLD=thres_hist) # get boundary
lowlat_noon = pb[:,2] - 6
lowlat_midnight = np.mean(eqb[:,[0,4]],axis=1) - 5


############################
# Bin the data according to the parameters determined above
binned_dict = helpers.bin_data(wic, LATRES=latres, \
        MLTRES=mltres, LOWLAT_MIDNIGHT=lowlat_midnight, \
        LOWLAT_NOON=lowlat_noon, HIGHLAT=highlat)

############################
# Calculate the histograms
eqb_hist, pb_hist = helpers.make_histogram(binned_dict, stepsize=stepsize)

############################
# Apply the boundary algorithm
eqb, pb = helpers.set_boundary(binned_dict, eqb_hist, pb_hist, THRESHOLD=thres_hist)



##########################################
# Plot all images
helpers.plot_images_event(wic, binned_dict, eqb_hist, pb_hist, eqb, pb, datapath+event)
