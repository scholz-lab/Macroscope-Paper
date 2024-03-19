#!/usr/bin/python3

import os
import sys
import numpy as np
import matplotlib.pylab as plt
import pims
import pandas as pd
from skimage import io

from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from skimage.morphology import disk, remove_small_holes, remove_small_objects, binary_closing, binary_opening, binary_erosion
from skimage.filters import gaussian, threshold_li
from skimage.measure import  regionprops, label

import pharaglow as pg
import pharaglow.features as pgf
import pharaglow.run as pgr

@pims.pipeline
def thresholdDros(img):
    """Use Yen threshold to obtain mask of pharynx.

    Args:
        im (numpy.array or pims.Frame): image

    Returns:
        np.array: binary image with only the largest object
    """
    mask = img>threshold_li(img)
    #mask = remove_small_holes(mask, 400)
    # erode a bit for tight turns
    mask = binary_erosion(mask.reshape(-1, params['length']), disk(10))
    mask = binary_opening(mask,  disk(10))
    labeled = label(mask)
    # keep only the largest item
    area = 0
    for region in regionprops(labeled):
        if area <= region.area:
            mask = labeled==region.label
            area = region.area
    return mask

### multiplexed analysis
def runPharaglowOnImageDros(image_red, image_green, framenumber, params, **kwargs):
    """ Run pharaglow-specific image analysis of a pharynx on a single image.

    Args:
        image (numpy.array or pims.Frame): input image
        framenumber (int): frame number to indicate in the resulting
        dataframe which image is being analyzed.
        arams (dict): parameter dictionary containing image analysis parameters.

    Returns:
        pandas.DataFrame: collection of data created by pharaglow for this image.
    """
    colnames = ['Mask', 'SkeletonX', 'SkeletonY','ParX', 'ParY', 'Xstart',\
         'Xend', 'Centerline', 'dCl', 'Widths', \
        'Contour', 'Straightened_red', 'Straightened_green', 'Kymo_red', 'Kymo_green']
    
    # skeletonize - custom to remove local gradient
    tmp_red = gaussian(image_red, 2) - gaussian(image_red, 150)
    
    mask = thresholdDros(tmp_red)

    # make the mask smooth to avoid skeleton branching
    mask_smooth = binary_opening(mask)
    skel = pgf.skeletonPharynx(mask)
    # getting centerline and widths along midline
    if np.sum(mask) == 0 or np.sum(skel)<6:
        print(f'Frame {framenumber} has too few points.')
        # make the columns that are images the right shape
        results = [mask.ravel()]+ [np.nan]*10+[np.zeros((params['widthStraight'],params['nPts']))]*2+[np.zeros(params['nPts'])]*2
        # make the columns that are images the right shape
       
    else:
        #centerline fit
        order = pgf.sortSkeleton(skel)
        ptsX, ptsY = np.where(skel)
        skelX, skelY = ptsX[order], ptsY[order]
        # getting centerline and widths along midline
        scale = params.pop('scale', 1)
        # note : for very bent shapes the widths are wrong
        parX, parY, xstart, xend, cl, dCl, widths, contour = \
        pgr.runPharaglowCL(mask,skelX, skelY, params['length'], scale = scale)
        # redo widths for very bent shapes
        
        # image transformation operations
        # straightened image
        straightened_red = pgf.straightenPharynx(image_red, xstart,xend,\
                                            parX, parY, params['widthStraight'],\
                                            params['nPts'])

        straightened_green = pgf.straightenPharynx(image_green, xstart,xend,\
                                            parX, parY, params['widthStraight'],\
                                            params['nPts'])
        # run kymographs
        kymo_red = np.mean(straightened_red, axis=1)
        kymo_green= np.mean(straightened_green, axis=1)
        results = [mask, skelX, skelY, parX, parY, xstart, xend,\
             cl, dCl, widths, contour,  straightened_red, straightened_green, kymo_red, kymo_green]

    data = {}
    for col, res in zip(colnames,results):
        data[col] = res
    df = pd.DataFrame([data], dtype = 'object')
    df['frame'] = framenumber
    print('Done', framenumber)
    return df,



def dros_orientation(df, ref_frame = 0):
    """ Get all images into the same orientation by comparing to a sample image.

    Args:
        df (pandas.DataFrame): a pharaglow dataframe after running .run.runPharaglowOnImage()

    Returns:
        pandas.DataFrame: dataFrame with flipped columns where neccessary
    """
    
    df['Similarity'] = False
    # use the distance between centerlines to gauge if flipped
    
    key = 'Centerline'
    sample = np.array(df[key][0])
    for ri, row in df.iterrows():
        current = np.array(row[key])
        sim = np.sum((current-sample)**2)<\
            np.sum((current-sample[::-1])**2)
        df.loc[ri, 'Similarity'] = sim
        if sim:
            sample = current
        else:
            sample = current[::-1]
    
    # now flip the orientation where the sample is upside down
    for key in ['SkeletonX', 'SkeletonY', 'Centerline', 'dCl', 'Widths',\
           'Straightened_red',  'Straightened_green', 'Kymo_red', 'Kymo_green']:
        if key in df.columns:
            df[key] = df.apply(lambda row: row[key] if row['Similarity'] \
                else row[key][::-1], axis=1)
    return df

def parallel_analysis(args, param, parallelWorker, framenumbers = None,  nWorkers = 5, output= None, depth = 'uint8', **kwargs):
    """Use multiprocessing to speed up image analysis. This is inspired by the trackpy.batch function.
    Args:
        args (tuple): contains iterables eg. (frames, masks) or just frames that will be iterated through.
        param (dict): image analysis parameters
        parallelWorker (func): a function defining what should be done with args
        framenumbers (list, optional): a list of frame numbers corresponding to the frames in args. Defaults to None.
        nWorkers (int, optional): Processes to use, if 1 will run without multiprocessing. Defaults to 5.
        output ([type], optional): {None, trackpy.PandasHDFStore, SomeCustomClass} a storage class e.g. trackpy.PandasHDFStore. Defaults to None.
        depth (str, optional): bit depth of frames. Defaults to 'uint8'.
    Returns:
        output or (pandas.DataFrame, numpy.array)
    """
    if framenumbers is None:
        framenumbers = np.arange(len(args[0]))

    # Prepare wrapped function for mapping to `frames`
    detection_func = partial(parallelWorker, params = param)
    #if nWorkers ==1:
    #    func = map
    #    pool = None
    #else:
        # prepare imap pool
    pool = ProcessPoolExecutor(max_workers=nWorkers, mp_context=mp.get_context("fork"))
    func = pool.map

    objects = []
    images = []
    try:
        for i, res in enumerate(func(detection_func, zip(*args, framenumbers))):
            if i%100 ==0:
                print(f'Analyzing image {i} of {len(args[0])}', flush=True)
            if len(res[0]) > 0:
                # Store if features were found
                if output is None:
                    objects.append(res[0])
                    if len(res)>1:
                        images += res[1]
                else:
                    # here we keep images within the dataframe
                    if len(res)>1:
                        res[0]['images'] = res[1]
                    output.put(res[0])
    finally:
        if pool:
            # Ensure correct termination of Pool
            pool.shutdown()

    if output is None:
        if len(objects) > 0:
            objects = pd.concat(objects).reset_index(drop=True)
            if len(images)>0:
                images = np.array([pad_images(im, shape, param['length'], reshape = True, depth=depth) for im,shape in zip(images, objects['shapeX'])])
                images = np.array(images).astype(depth)
            return objects, images
        else:  # return empty DataFrame
            warnings.warn("No objects found in any frame.")
            return pd.DataFrame([]), images
    else:
        return output
    
    
def parallelWorker(args, **kwargs):
    """helper wrapper to run object detection with multiprocessing.
    Args:
        args (div.): arguments for detect
    Returns:
        pandas.DataFrame: dataframe with information for each image
        list: list of corresponding images.
    """
    return runPharaglowOnImageDros(*args, **kwargs)
    

    
if __name__ == "__main__":
    # set up filenames and output locations
    if len(sys.argv) > 1:
        movie = sys.argv[1]
    else:
        movie = 'Drosophila_larvae_vinaigre_011'
    
    # input data
    fname = f'/gpfs/soma_fs/nif/nif9201.bak/Euphrasie/Tracking experiment/Drosophila tracks/Unsuccessful/{movie}/'
    frames_red = pims.open(fname + '*main.tiff')
    frames_green = pims.open(fname + '*minor.tiff')
    print(f'Analyzing {movie} in {fname} with {len(frames_red)} frames.')
    # output is here:
    outPath = '/gpfs/soma_fs/home/scholz/scratch/drosophila/'
    outfile = f'{movie}_results.json'

    # parameters and parallelization
    nWorkers = 100
    params = {"length":frames_red[0].shape[1], #second axis of the image to send unraveled images
    "widthStraight":75,# half width of the larvae in px
    "nPts":300,# points along centerline to sample. to get pixel-resolution, use same number as larve in px
    "linewidth":2
    }
    
    # analysis churns
    tmp,_ = parallel_analysis((frames_red,frames_green,), params,\
                  parallelWorker= parallelWorker, framenumbers = np.arange(len(frames_red)), nWorkers = nWorkers, output= None)

    tmp = dros_orientation(tmp, ref_frame = 100)
    # take out the kymographs into a better format -- remove nans and replace with zeros
    kymo_red = np.array([np.array(x) for x in tmp['Kymo_red'].values]).T
    kymo_green = np.array([np.array(x) for x in tmp['Kymo_green'].values]).T
    
    tmp = tmp.drop(['Mask', 'SkeletonX', 'SkeletonY', 'ParX', 'ParY', 
                   'Xstart', 'Xend', 'dCl', 'Widths',  'Gradient', 
                     'KymoGrad', 'Similarity', 'Xtmp', 'Kymo_red', 'Kymo_green'], axis = 1, errors = 'ignore')
    print(f'Saving columns: {tmp.columns}')
    # save data
    tmp.to_json(os.path.join(outPath, outfile), orient='split')

    # save kymos as tiff for exact storage
    io.imsave(os.path.join(outPath,f'{movie}_kymo_green.tif'), kymo_green)
    io.imsave(os.path.join(outPath,f'{movie}_kymo_red.tif'), kymo_red)


    # save plots
    f, ax = plt.subplots(1,3,figsize=(30,3))
    fps = 15 # TODO CHECK
    ax[0].imshow(kymo_red, aspect = 'auto', extent = (0, kymo_red.shape[1]/fps, 0, 100))
    ax[1].imshow(kymo_green, aspect = 'auto',extent = (0, kymo_red.shape[1]/fps, 0, 100)) 
    ax[2].imshow(kymo_green-kymo_red, aspect = 'auto',extent = (0, kymo_red.shape[1]/fps, 0, 100))
    plt.setp(ax, xlabel='time (s)', ylabel = '%body length', xlim = (0,60));
    plt.tight_layout()
    plt.savefig(os.path.join(outPath, f'Kymographs_{movie}.svg'))