'''
This file contain various helper functions needed in the auroral boundary
determination process

'''
from pysymmetry import fuvpy
import glob
import numpy as np
import pandas as pd
from pathlib2 import Path
import os
import matplotlib.pyplot as plt
from pysymmetry.visualization import polarsubplot
import xarray as xr


ll_vs_mlt = lambda mlt: LOWLAT_MIDNIGHT + (LOWLAT_NOON - LOWLAT_MIDNIGHT) * (1 - np.cos(mlt * 2 * np.pi / 24)) / 2 # function that defines the lower latitude boundary

def __ll_vs_mlt(mlt, LOWLAT_MIDNIGHT=45, LOWLAT_NOON=55):
    '''
    return the low latitude limit at the specified MLT to be usd when constructing
    histograms.
    '''

    llb = LOWLAT_MIDNIGHT + (LOWLAT_NOON - LOWLAT_MIDNIGHT) * (1 - np.cos(mlt * 2 * np.pi / 24)) / 2 # function that defines the lower latitude boundary
    return llb

    #mlt = np.array(mlt)
    #llb = np.full(mlt.shape, LOWLAT_MIDNIGHT)
    #llb[(mlt > 8) & (mlt < 16)] = LOWLAT_NOON
    #return llb


def fit_circle(data):
    '''
    data: xarray dataset with the images to produce circle fits to. The fit is done
            on each image individually
    '''
    n_images = data.sizes['date'] #number of images in event
    for i in range(n_images):
        break



def load_data(event, datapath, dayglow_method = ('std',), flatfield_issue=True, \
        image_filtering=False):
    '''
    Load data into xarray object using fuvpy.

    event: string identifying the event
    datapath: path pointing to folder containting events
    dayglow model: the tyoe of dayglow model to used
    flatfield_issue: Some events have shown that improvements can be made by applying the flatfield
        correction differently than what has been done in FUVIEW. Setting this keyword to True
        will use this alternative flatfield implementation
    image_filtering: placeholder keyword to support filtering of the images before analysis
    '''

    wicfiles = glob.glob(datapath + event + '/idl_files/*')
    wic = fuvpy.readFUVimage(wicfiles) #Load into xarray object using fuvpy
    if flatfield_issue:
        wic = fuvpy.reflatWIC(wic) #Fixing issue with flatfield from FUVIEW
    wic = fuvpy.makeFUVdayglowModel(wic, model=dayglow_method) #Subtract dayglow
    wic['cimage']=wic['cimage'].where(wic['bad'],np.nan) #Set the flagged bad pixels to nan


    return wic


def bin_data(wic, LATRES=1, MLTRES=1, LOWLAT_MIDNIGHT=45, LOWLAT_NOON=55, \
        HIGHLAT=85, INIT=False, dzalim=70):
    '''
    Copy of the code in 'new_bd_range_improved.py' lines ~91-232.
    LATRES: width in degrees of the latitude binning
    MLTRES: width in hrs of the MLT binning
    LOWLAT_MIDNIGHT: where to stop searching at midnight. An array corresponding to
                each image unless INIT=True. Then a scalar
    LOWLAT_NOON: Where to stop searching at noon. An array corresponding to
                each image unless INIT=True. Then a scalar
    HIGHLAT: maximum latitude for the latitude binning. Scalar.
    INIT: set to True if the purpose of the binning is to determine
            LOWLAT_NOON/MIDNIGHT for further analysis
    dzalim: degrees, limit on satellite zenith angle to use. Set pixels with
            dza>dzalim to nan

    '''

    # initialize arrays that will contain the histograms for all images - must be same shape as grid
    n_images = wic.sizes['date'] #number of images in event
    #avg_counts = np.zeros([bin_mlt.shape[0],bin_lat.shape[0],n_images])*np.nan # all nans
    avg_counts = []
    mlat_centre = []
    mlat_xx = []
    mlt_xx = []

    #Gridding and binning business
    if INIT:
        mltbins = np.array([0,3,9,15,21,24])
        bin_mlt = np.array([1.5,6,12,18,22.5]) #bin centre
        latbins = np.r_[LOWLAT_MIDNIGHT:HIGHLAT+LATRES:LATRES]
        LOWLAT_NOON = np.ones(n_images) * LOWLAT_NOON
        LOWLAT_MIDNIGHT = np.ones(n_images) * LOWLAT_MIDNIGHT
    else:
        mltbins = np.r_[0 :24+MLTRES:MLTRES]
        bin_mlt = mltbins[:-1] + MLTRES/2 #bin centre

    for i in range(n_images):
        if np.isnan(LOWLAT_MIDNIGHT[i]) | np.isnan(LOWLAT_NOON[i]):
            latbins = np.r_[50:HIGHLAT+LATRES:LATRES]
            bin_lat = latbins[:-1] + LATRES/2  #bin centre
            mlat_centre.append(bin_lat)
            mltxx, mlatxx = np.meshgrid(bin_mlt, bin_lat, indexing = 'ij')
            mlt_xx.append(mltxx)
            mlat_xx.append(mlatxx)
            avg_counts.append(np.zeros([len(bin_mlt),len(bin_lat)]))
        else:
            #Gridding and binning business specific for each image
            latbins = np.r_[LOWLAT_MIDNIGHT[i]:HIGHLAT+LATRES:LATRES]
            bin_lat = latbins[:-1] + LATRES/2  #bin centre
            mlat_centre.append(bin_lat)
            mltxx, mlatxx = np.meshgrid(bin_mlt, bin_lat, indexing = 'ij')
            mlt_xx.append(mltxx)
            mlat_xx.append(mlatxx)

            #Extract data from single image
            w = wic.isel(date=i) #image i
            mlat = w['mlat'].values.copy().flatten()
            mlt = w['mlt'].values.copy().flatten()
            w['cimage'] = xr.where(w['dza']>dzalim,np.nan,w['cimage'])
            counts = w['cimage'].values.copy().flatten()

            # Grid data and do some proecssing on the gridded data
            # use pandas to bin in 2D (this is alternative to the histogram stuff that Anders does)
            avg_count = pd.Series(counts).groupby([pd.cut(mlt, mltbins), pd.cut(mlat, latbins)]).median()
            # turn multiindex dataframe into numpy array - the rows will be the latitude profiles
            avg_count = avg_count.unstack().values
            # subtract minimum of each row - use nanmin to ignore nans:
            avg_count = avg_count - np.nanmin(avg_count, axis = 1).reshape((-1, 1))
            # zero zounts that are below the MLT dependent low latitude boundary
            avg_count[mlatxx < __ll_vs_mlt(mltxx, LOWLAT_MIDNIGHT=LOWLAT_MIDNIGHT[i], \
                    LOWLAT_NOON=LOWLAT_NOON[i])] = 0
            # subtract minimum of each row again - use nanmin to ignore nans:
            avg_count = avg_count - np.nanmin(avg_count, axis = 1).reshape((-1, 1))
            #avg_counts[:,:,i] = avg_count
            avg_counts.append(avg_count)

    binned_dict = {'mlt_centre':bin_mlt, 'mlat_centre':mlat_centre, 'mltxx':mlt_xx, \
            'mlatxx':mlat_xx, 'binned_counts':avg_counts, 'LATRES':LATRES, \
            'MLTRES':MLTRES, 'LOWLAT_MIDNIGHT':LOWLAT_MIDNIGHT, 'LOWLAT_NOON':LOWLAT_NOON, \
            'HIGHLAT':HIGHLAT}
    return binned_dict

def make_histogram(binned_dict, stepsize=5):
    '''
    Make histogram of the binned data returned by bin_data()
    stepsize: counts, threshold for step in histograms
    '''
    # initialize arrays that will contain the histograms for all images - must be same shape as grid
    n_images = len(binned_dict['binned_counts']) #number of images in event
    eblat_hist = []
    pblat_hist = []
    #eblat_hist = np.zeros([binned_dict['mlt_centre'].shape[0],binned_dict['mlat_centre'].shape[0]-2, \
    #        n_images])*np.nan # all nans
    #pblat_hist = np.zeros([binned_dict['mlt_centre'].shape[0],binned_dict['mlat_centre'].shape[0]-2, \
    #        n_images])*np.nan # all nans

    for i in range(n_images):
        # initialize an array that will contain the histograms - must be same shape as grid
        eblat_hist0 = np.zeros(binned_dict['binned_counts'][i].shape) # all zeros
        pblat_hist0 = np.zeros(binned_dict['binned_counts'][i].shape) # all zeros
        avg_count = binned_dict['binned_counts'][i]

        # loop through the threshold steps:
        for thr in np.r_[stepsize: np.nanmax(avg_count) + 1: stepsize]:
            # equatorward boundary:
            eblat_hist0[np.arange(avg_count.shape[0]), np.argmax(avg_count > thr, axis = 1)] += 1 # increment the histogram array by one at first occurrence above threshold - in each row
            # for the poleward boundary, we need to use argmax on avg_count with the
            # counts in the reversed order, since it returns the first occurrence where the
            # condition is fulfilled:
            iii = np.argmax(avg_count[:, ::-1] > thr, axis = 1)
            # and then we need to subtract this number from the highest row index to get the location in the format that we want:
            iii = avg_count.shape[-1] - iii - 1
            pblat_hist0[np.arange(avg_count.shape[0]), iii] += 1

        #Final processing of the histograms
        # remove element from histogram at the edge
        eblat_hist0[binned_dict['mlatxx'][i] < (__ll_vs_mlt(binned_dict['mltxx'][i], \
            LOWLAT_MIDNIGHT=binned_dict['LOWLAT_MIDNIGHT'][i], \
            LOWLAT_NOON=binned_dict['LOWLAT_NOON'][i]) + binned_dict['LATRES'])] = 0
        # remove first and last (in lat) element of histogram
        pblat_hist0 = pblat_hist0[:, 1:-1]
        eblat_hist0 = eblat_hist0[:, 1:-1]

        #Stack together histograms of each image into 3D array
        eblat_hist.append(eblat_hist0)
        pblat_hist.append(pblat_hist0)
        #eblat_hist[:,:,i] = eblat_hist0
        #pblat_hist[:,:,i] = pblat_hist0

    return [eblat_hist, pblat_hist]



def set_boundary(binned_dict, eqb_hist, pb_hist, THRESHOLD=10):
    '''
    Boundary detection algorithm based on the histigrams
    binned_dict: What is returned from bin_data() functions
    eqb_hist: list of equatorward histograms for each image. Each histogram array
            has dimensions [n_mlt,n_mlat]
    pb_hist: list of poleward histograms for each image. Each histogram array
            has dimensions [n_mlt,n_mlat]
    THRESHOLD: trigger threshold for number of counts in histigrams
    '''
    n_images = len(binned_dict['binned_counts']) #number of images in event
    n_mlts = eqb_hist[0].shape[0]

    #Make arrays to hold final final boundaries
    eqb = np.zeros([n_images,n_mlts]) * np.nan
    pb = np.zeros([n_images, n_mlts]) * np.nan

    for i in range(n_images):
        mlats = binned_dict['mlat_centre'][i][1:-1] #corresponds to the histograms
        eqb_index = np.argmax(eqb_hist[i] > THRESHOLD, axis=1) #index of first occirrence avove threshold
        #This way of inferring the indexes fails when the first element along axis=1 is greater than
        # THRESHOLD. This will be flagged as nan below.
        eqb_i = mlats[eqb_index] -1 #equatorward boundary values. -1 is experimental
        nans = eqb_index == 0
        eqb_i[nans] = np.nan
        eqb[i,:] = eqb_i

        # Similar as when constructing the histograms, for the poleward boundary,
        # we need to use argmax on pb_hist with the
        # counts in the reversed order, since it returns the first occurrence where the
        # condition is fulfilled:
        iii = np.argmax(pb_hist[i][:, ::-1] > THRESHOLD, axis = 1)
        # and then we need to subtract this number from the highest row index to get the location in the format that we want:
        pb_index = pb_hist[i].shape[1] - iii - 1
        pb_i = mlats[pb_index]
        nans = pb_index == pb_hist[i].shape[1] - 1
        pb_i[nans] = np.nan
        pb[i,:] = pb_i

    return [eqb, pb]

def plot_images_event(wic, binned_dict, eqb_hist, pb_hist, eqb, pb, path, dzalim=70):
    '''
    Plot all images in event
    '''

    #Create directory
    my_file = Path(path + '/jone_plots/')
    if not my_file.exists():
        os.mkdir(path + '/jone_plots/')
    my_file = Path(path + '/jone_plots/' + path[-10:] + '/')
    if not my_file.exists():
        os.mkdir(path + '/jone_plots/' + path[-10:] + '/')

    #Plot every image
    n_images = wic.sizes['date'] #number of images in event
    for image in range(n_images):
        img = wic.isel(date=image)
        img['cimage'] = xr.where(img['dza']>dzalim,np.nan,img['cimage'])
        title = img['id'].values.tolist() + ': ' + img['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist() + '-' + str(image)
        title = title.replace(':','.')
        #Histogram and intensity profile figure
        fig, axs = plt.subplots(6,4, figsize=(15,10), facecolor='w')
        fig.subplots_adjust(left = 0.09, right = 0.99, top = 0.93, bottom = 0.04, \
                hspace = .05, wspace=.01)
        axs = axs.ravel()
        mlat_counts = binned_dict['mlat_centre'][image]
        mlat_hist = binned_dict['mlat_centre'][image][1:-1] #drop first and last element
        mlts = binned_dict['mlt_centre']
        counts = binned_dict['binned_counts'][image]
        for p in range(len(mlts)):
            ax = axs[p]
            twin1 = ax.twinx()
            ax.plot(mlat_counts, counts[p,:], label='count', c='blue')
            ax.vlines(eqb[image,p], 0, np.max(counts[p,:]), color='black', linestyle='dashed')
            ax.vlines(pb[image,p], 0, np.max(counts[p,:]), color='black')
            ax.tick_params(axis="y",direction="in",pad=-25)
            twin1.plot(mlat_hist, eqb_hist[image][p,:], label='eqb_hist',c='orange')
            twin1.plot(mlat_hist, pb_hist[image][p,:], label='pb_hist',c='green')
            twin1.tick_params(axis="y",direction="in",pad=-22)
            ax.text(0.3,0.9, 'mlt: '+str(mlts[p]), color='black', size=10, transform=ax.transAxes)
            if p == 0:
                ax.text(0.7,0.9, 'counts', color='blue', size=10, transform=ax.transAxes)
                ax.text(0.7,0.8, 'Eq-hist', color='orange', size=10, transform=ax.transAxes)
                ax.text(0.7,0.7, 'P-hist', color='green', size=10, transform=ax.transAxes)
                ax.text(1.5,1.15,title, size=16, transform=ax.transAxes)
                #twin1.legend()
        #plt.show()
        fig.savefig(path + '/jone_plots/' + path[-10:] + '/histogram_' + title + '_.png', bbox_inches='tight', dpi = 250)

        #Plpot the image investigated
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pax = polarsubplot.Polarsubplot(ax, color = 'gray', linestyle = '-', minlat = 50, linewidth=1)
        a = pax.showFUVimage(img,inImg='cimage', crange=[-500,1500])
        ax.set_title(img['id'].values.tolist() + ': ' + img['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values.tolist())
        pax.plot(__ll_vs_mlt(binned_dict['mlt_centre'], \
                LOWLAT_MIDNIGHT=binned_dict['LOWLAT_MIDNIGHT'][image], LOWLAT_NOON=binned_dict['LOWLAT_NOON'][image]),binned_dict['mlt_centre'], c='orange')
        pax.plot(eqb[image,:], binned_dict['mlt_centre'], c='black', linestyle='dashed')
        pax.scatter(eqb[image,:], binned_dict['mlt_centre'], c='black')
        pax.plot(pb[image,:], binned_dict['mlt_centre'], c='black')
        pax.scatter(pb[image,:], binned_dict['mlt_centre'], c='black')
        cax = fig.add_axes([0.06, 0.05, 0.9, 0.06])
        plt.colorbar(pax.ax.collections[0],orientation='horizontal',cax=cax, extend='both')
        #plt.show()
        fig.savefig(path + '/jone_plots/' + path[-10:] + '/image_' + title + '_.png', bbox_inches='tight', dpi = 250)
