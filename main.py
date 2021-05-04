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
event = '2000-11-24' #2000-11-29
dayglow_method = ('std',)
flatfield_issue = True
image_filtering = False

#Parameters determining binning, histograms, and boundary detection
latres = 0.5
mltres = 1
lowlat_midlight = 56
lowlat_noon = 63
highlat = 85
stepsize = 5 # counts, for building histograms
thres_hist = 11 #threshold for setting boundary based on histigrams

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

#TODO: Make function to determine the low latitude cut off when making the binned MLT slices. This really helps the algorithm for the equatorward boundaries, as the unperfect dayglow model becomes less crucial.


# Bin the data according to the parameters set above
binned_dict = helpers.bin_data(wic, LATRES=latres, \
        MLTRES=mltres, LOWLAT_MIDNIGHT=lowlat_midlight, \
        LOWLAT_NOON=lowlat_noon, HIGHLAT=highlat)

# Then calculate the histograms
eqb_hist, pb_hist = helpers.make_histogram(binned_dict, stepsize=stepsize)

# Apply the boundary algorithm
eqb, pb = helpers.set_boundary(binned_dict, eqb_hist, pb_hist, THRESHOLD=thres_hist)



##########################################
#Plotting for inspection along the way
image = 50 #which image to plot

#Histogram and intensity profile figure
fig = plt.figure(figsize=(20,14))
fig, axs = plt.subplots(6,4, figsize=(15,10), facecolor='w')
fig.subplots_adjust(left = 0.09, right = 0.99, top = 0.93, bottom = 0.04, hspace = .05, wspace=.01)
axs = axs.ravel()
mlat0 = binned_dict['mlat_centre']
mlat = binned_dict['mlat_centre'][1:-1]
mlts = binned_dict['mlt_centre']
counts = binned_dict['binned_counts']
for p in range(len(mlts)):
    ax = axs[p]
    twin1 = ax.twinx()
    ax.plot(mlat0, counts[p,:,image], label='count', c='blue')
    ax.vlines(eqb[p,image], 0, np.max(counts[p,:,image]), color='black', linestyle='dashed')
    ax.vlines(pb[p,image], 0, np.max(counts[p,:,image]), color='black')
    ax.tick_params(axis="y",direction="in",pad=-20)
    twin1.plot(mlat, eqb_hist[p,:,image], label='eqb_hist',c='orange')
    twin1.plot(mlat, pb_hist[p,:,image], label='pb_hist',c='green')
    twin1.tick_params(axis="y",direction="in",pad=-20)
    ax.set_title('mlt: '+str(mlts[p]))#+'-'+str(6*p+6))
    if p == 0:
        ax.legend()
        #twin1.legend()
plt.show()

#Plpot the image investigated
fig = plt.figure()
ax = fig.add_subplot(111)
pax = polarsubplot.Polarsubplot(ax, color = 'gray', linestyle = '-', minlat = 50, linewidth=1)
img = wic.isel(date=image)
pax.showFUVimage(img,inImg='cimage', crange=[-500,1500])
pax.plot(helpers.__ll_vs_mlt(binned_dict['mlt_centre'], \
        LOWLAT_MIDNIGHT=lowlat_midlight, LOWLAT_NOON=lowlat_noon),binned_dict['mlt_centre'], c='orange')
pax.plot(eqb[:,image], binned_dict['mlt_centre'], c='black', linestyle='dashed')
pax.scatter(eqb[:,image], binned_dict['mlt_centre'], c='black')
pax.plot(pb[:,image], binned_dict['mlt_centre'], c='black')
pax.scatter(pb[:,image], binned_dict['mlt_centre'], c='black')
plt.show()
