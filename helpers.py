'''
This file contain various helper functions needed in the auroral boundary
determination process

'''
from pysymmetry import fuvpy
import glob
import numpy as np
import pandas as pd

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
        HIGHLAT=85):
    '''
    Copy of the code in 'new_bd_range_improved.py' lines ~91-232.
    LATRES: width in degrees of the latitude binning
    MLTRES: width in hrs of the MLT binning
    LOWLAT_MIDNIGHT: where to stop searching at midnight
    LOWLAT_NOON: Where to stop searching at noon
    HIGHLAT: maximum latitude for the latitude binning

    '''

    #Gridding and binning business
    latbins = np.r_[LOWLAT_MIDNIGHT:HIGHLAT+LATRES:LATRES]
    mltbins = np.r_[0 :24+MLTRES:MLTRES]
    bin_mlt = mltbins[:-1] + MLTRES/2 #bin centre
    bin_lat = latbins[:-1] + LATRES/2  #bin centre
    mltxx, mlatxx = np.meshgrid(bin_mlt, bin_lat, indexing = 'ij')

    # initialize arrays that will contain the histograms for all images - must be same shape as grid
    n_images = wic.sizes['date'] #number of images in event
    avg_counts = np.zeros([bin_mlt.shape[0],bin_lat.shape[0],n_images])*np.nan # all nans

    for i in range(n_images):
        #Extract data from single image
        w = wic.isel(date=i) #image i
        mlat = w['mlat'].values.copy().flatten()
        mlt = w['mlt'].values.copy().flatten()
        counts = w['cimage'].values.copy().flatten()

        # Grid data and do some proecssing on the gridded data
        # use pandas to bin in 2D (this is alternative to the histogram stuff that Anders does)
        avg_count = pd.Series(counts).groupby([pd.cut(mlt, mltbins), pd.cut(mlat, latbins)]).median()
        # turn multiindex dataframe into numpy array - the rows will be the latitude profiles
        avg_count = avg_count.unstack().values
        # subtract minimum of each row - use nanmin to ignore nans:
        avg_count = avg_count - np.nanmin(avg_count, axis = 1).reshape((-1, 1))
        # zero zounts that are below the MLT dependent low latitude boundary
        avg_count[mlatxx < __ll_vs_mlt(mltxx, LOWLAT_MIDNIGHT=LOWLAT_MIDNIGHT, \
                LOWLAT_NOON=LOWLAT_NOON)] = 0
        # subtract minimum of each row again - use nanmin to ignore nans:
        avg_count = avg_count - np.nanmin(avg_count, axis = 1).reshape((-1, 1))
        avg_counts[:,:,i] = avg_count

    binned_dict = {'mlt_centre':bin_mlt, 'mlat_centre':bin_lat, 'mltxx':mltxx, \
            'mlatxx':mlatxx, 'binned_counts':avg_counts, 'LATRES':LATRES, \
            'MLTRES':MLTRES, 'LOWLAT_MIDNIGHT':LOWLAT_MIDNIGHT, 'LOWLAT_NOON':LOWLAT_NOON, \
            'HIGHLAT':HIGHLAT}
    return binned_dict

def make_histogram(binned_dict, stepsize=5):
    '''
    Make histogram of the binned data returned by bin_data()
    stepsize: counts, threshold for step in histograms
    '''
    # initialize arrays that will contain the histograms for all images - must be same shape as grid
    n_images = binned_dict['binned_counts'].shape[2] #number of images in event
    eblat_hist = np.zeros([binned_dict['mlt_centre'].shape[0],binned_dict['mlat_centre'].shape[0]-2, \
            n_images])*np.nan # all nans
    pblat_hist = np.zeros([binned_dict['mlt_centre'].shape[0],binned_dict['mlat_centre'].shape[0]-2, \
            n_images])*np.nan # all nans

    for i in range(n_images):
        # initialize an array that will contain the histograms - must be same shape as grid
        eblat_hist0 = np.zeros(binned_dict['binned_counts'][:,:,i].shape) # all zeros
        pblat_hist0 = np.zeros(binned_dict['binned_counts'][:,:,i].shape) # all zeros
        avg_count = binned_dict['binned_counts'][:,:,i]

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
        eblat_hist0[binned_dict['mlatxx'] < (__ll_vs_mlt(binned_dict['mltxx'], \
            LOWLAT_MIDNIGHT=binned_dict['LOWLAT_MIDNIGHT'], LOWLAT_NOON=binned_dict['LOWLAT_NOON']) + binned_dict['LATRES'])] = 0
        # remove first and last (in lat) element of histogram
        pblat_hist0 = pblat_hist0[:, 1:-1]
        eblat_hist0 = eblat_hist0[:, 1:-1]

        #Stack together histograms of each image into 3D array
        eblat_hist[:,:,i] = eblat_hist0
        pblat_hist[:,:,i] = pblat_hist0
    return [eblat_hist, pblat_hist]



def set_boundary(binned_dict, eqb_hist, pb_hist, THRESHOLD=10):
    '''
    Boundary detection algorithm based on the histigrams
    '''

    mlats = binned_dict['mlat_centre'][1:-1] #corresponds to the histograms
    w = np.where(eqb_hist>THRESHOLD)
    eqb_index = np.argmax(eqb_hist > THRESHOLD, axis=1) #index of first occirrence avove threshold
    #This way of inferring the indexes fails when the first element along axis=1 is greater than
    # THRESHOLD. This will be flagged as nan below.
    nans = eqb_index == 0
    eqb = mlats[eqb_index] #equatorward boundary values
    eqb[nans] = np.nan

    # Similar as when constructing the histograms, for the poleward boundary,
    # we need to use argmax on pb_hist with the
    # counts in the reversed order, since it returns the first occurrence where the
    # condition is fulfilled:
    iii = np.argmax(pb_hist[:, ::-1,:] > THRESHOLD, axis = 1)
    # and then we need to subtract this number from the highest row index to get the location in the format that we want:
    pb_index = pb_hist.shape[1] - iii - 1
    nans = pb_index == pb_hist.shape[1] - 1
    pb = mlats[pb_index]
    pb[nans] = np.nan
    return [eqb, pb]
