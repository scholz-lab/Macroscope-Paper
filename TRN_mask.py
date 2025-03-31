## Calcium data
# A modified tracker for low-resolution calcium tracking. The general idea is loosely based on pharaglow. Basic paralellization is available.

# imports
import os
import numpy as np
import pandas as pd
import pims
import yaml
import submitit

from skimage.filters import threshold_yen, gaussian
from skimage.morphology import closing, square, square


# functions
@pims.pipeline
def cropim(im, crop_w, crop_h):
    """ Crop an image anchored at center.
    Args:
        im (numpy.array or pims.Frame): image to be cropped
        crop_w (int): width to crop image to
        crop_h (int): height to crop image to
    Returns:
        numpy.array: cropped image
    """
    h,w = im.shape
    return im[h//2-crop_h:h//2+crop_h, w//2-crop_w:w//2+crop_w]

@pims.pipeline
def bgsubtract_gauss(image, bg, filtersize):
    """ Preprocess an image using background subtractionn and smoothing with a gaussian kernel.
    Args:
        image (numpy.array or pims.Frame): larger image
        bg (numpy.array or pims.Frame): larger image
        filtersize (int): size of the gaussian filter applied to the image
    Returns:
        numpy.array: image of same size as input 
    """
    return gaussian(image-bg, filtersize)

@pims.pipeline
def measure_im(im, mask):
    """Measure image intensity at mask: maximum, mean, 5 perecentile, 95 percentile, sum.
    Args:
        im (numpy.array or pims.Frame): image
        mask (numpy.array): mask for all images
    Returns:
        sig (numpy.array): measured values (max, mean, mean, 5 perecentile, 95 percentile, sum)
    """
    i = im.frame_no
    try:
        sig = np.array([np.max(im[mask[i]]), np.mean(im[mask[i]]), *np.percentile(im[mask[i]], [5,95]), np.sum(mask[i])])
    except ValueError:
        sig = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    return sig

@pims.pipeline
def mask_image(image, tfactor = 0.9, closing_size = 3):
    """ Create a binary mask of the input image using yen thresholding.
    Args:
        image (numpy.array or pims.Frame): original image
    Returns:
        numpy.array: binary image of same size as input 
    """
    #thresh = np.percentile(image, [99.999])[0]#threshold_yen(image)
    #thresh = threshold_li(image, initial_guess = thresh)
    thresh = threshold_yen(image)*tfactor
    bw = closing(image >= thresh, square(closing_size))
    return bw

@pims.pipeline
def constant_size(mask, half_mask_size=2):
    """ Create a new binary mask from a mask and a mask size.
    Args:
        mask (numpy.array or pims.Frame): binary mask
        half_mask_size (int): half of the desired mask size
    Returns:
        numpy.array: binary mask of same size as input 
    """
    cms = np.round(np.mean(np.where(mask), axis=1)).astype(int)
    new_mask = np.zeros_like(mask)
    new_mask[cms[0]-half_mask_size : cms[0]+half_mask_size+1,
             cms[1]-half_mask_size : cms[1]+half_mask_size+1] = 1
    return new_mask
    
    
def load_mov(inpath, mov):
    """ Load movie with pims from inpath and videofile-name.
    Args:
        inpath (str): path to movie
        mov (str): videofile-name
    Returns:
        numpy.array: binary mask of same size as input 
    """
    if os.path.isdir(os.path.join(inpath, mov)):
        main = pims.open(os.path.join(inpath, mov, "*main*.tiff"))
        minor = pims.open(os.path.join(inpath, mov, "*minor*.tiff"))
    elif os.path.isdir(os.path.join(inpath, mov+'c')):
        main = pims.open(os.path.join(inpath, mov+'c', "*main*.tiff"))
        minor = pims.open(os.path.join(inpath, mov+'c', "*minor*.tiff"))
    else:
        pass
    return main, minor

def multidf_multikeys(data):
    """ Create a multiindex dataframe from nested dictionary
    Args:
        data (dict): nested dictionary
    Returns:
        df (pandas.DataFrame): DataFrame containing data from input dictionary
    """
    df = pd.DataFrame([])
    multi_index = []
    for key, dict_ in data.items():
        for k, d in dict_.items():
            df = pd.concat([df, d], axis=1)
            multi_index.append([(key,k,c) for c in d.columns])
    multi_index = [tup for lst in multi_index for tup in lst]
    df.columns = pd.MultiIndex.from_tuples(multi_index)
    return df

def process_videos_in_parallel(mov, info, inpath, config):        
    # get window for analysis 
    #window = tuple(map(int, info.loc[mov]['valid range'].split('-')))
    
    # load movies
    main, minor = load_mov(inpath, mov)

    # crop both channels
    crop = config['settings']['crop']
    main_ = cropim(main, crop_w=crop[0], crop_h=crop[1])
    minor_ = cropim(minor, crop_w=crop[0], crop_h=crop[1])

    # background subtraction for main
    bg = np.percentile(main_[::100], 50, axis=0)
    main_bgzero = bgsubtract_gauss(main_, bg, .3)

    # get mask from background subtracted main image
    mask = mask_image(main_bgzero, tfactor = 8)
    #mask = constant_size(mask) # use square mask instead of cell

    # measure both channels at mask
    #ref = measure_im(main_[window[0]:window[1]], mask)
    #sig =  measure_im(minor_[window[0]:window[1]], mask)
    ref = measure_im(main_, mask)
    sig =  measure_im(minor_, mask)

    # create dataframe
    ref_arr = np.array(ref)
    sig_arr= np.array(sig)
    ref_arr = pd.DataFrame(ref_arr, columns=['max', 'mean', '5th percentile', '95th percentile', 'mask size'])
    sig_arr = pd.DataFrame(sig_arr, columns=['max', 'mean', '5th percentile', '95th percentile', 'mask size'])
    deltaf_mov = sig_arr/ref_arr

    return mov, {'ref': ref_arr, 'sig': sig_arr, 'sig/ref': deltaf_mov}


# Settings
config_path = "TRN_config.yml"
config = yaml.safe_load(open(config_path, "r"))
datestr = config['filesettings']['date']
exp = config['filesettings']['experiment']
inpath = os.path.join(config['filesettings']['videopath'], exp)
outpath = os.path.join(config['filesettings']['outpath'], exp, datestr)
if not os.path.exists(outpath):
    os.makedirs(outpath)

info = pd.read_csv(config['filesettings']['fileinfo'], index_col='plate')
info = info[~(info['exclude']==True)]
list_movies = list(info.index)


# initialise data dictionary


# Define batch size
executor = submitit.SlurmExecutor(folder='logs-calcium')

# Setting the parameters for the slurm job
executor.update_parameters(
    partition="CPU,GPU",
    time="01:00:00",  
    mem="350G",
    cpus_per_task=48,
    comment="calcium-job",
    job_name="calcium-job",
    array_parallelism=100,
)

jobs = []
with executor.batch():
    # Loop over movies specified in info
    for i, mov in enumerate(list_movies):
        print(f"{i+1}/{len(list_movies)}: {mov}")
        job = executor.submit(process_videos_in_parallel, mov, info, inpath, config)
        jobs.append(job)

# Waiting for jobs to complete
# append
arr_types = ['ref','sig','sig/ref']
data = {j.result()[0]:j.result()[1] for j in jobs}

# create multiindex dataframe
signal_multi = multidf_multikeys(data)

# dump json
first_mov = np.sort(list_movies)[0].replace(exp,'')
last_mov = np.sort(list_movies)[-1].replace(exp,'')
out_fn = f'{exp}_{first_mov}-{last_mov}_mask_{datestr}.json'
signal_multi.to_json(os.path.join(outpath, out_fn))
print(f'json file {out_fn} saved at {outpath}')