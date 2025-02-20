
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import re
from itertools import chain


### functions

def safe_eval(string):
    """
    Safe way to evaluate a string without throwing errors.
    If string can be evaluated as float, returns float, else a string
    Args:
        string (str): string to evaluate
    Returns:
        v (str or float)
    """
    try:
        v = float(string)
    except:
        v = str(string)
    return v

def get_stage_dict(file):
    """
    Extract a dictionary from stage coordinates file from GlowTracker
    Args:
        file (str): path to coordinate file from Glowtracker
    Returns:
        result_dict (dict): nested dictionary with keys and values from metadata from coordinate file, excluding actual recording
    """
    result_dict = {}
    current_key = None
    # open file
    with open(file, 'r') as file:
        # iterate over lines
        for line in file:
            line = line.strip()
            # assign current_key if line startswith "#"
            if line.startswith('#'):
                current_key = line[1:].strip()  # Remove the '#' and strip any leading/trailing whitespace
                result_dict[current_key] = {}
            # with current_key set, created nested dict from following line
            elif current_key is not None:
                if re.match("^[a-zA-Z]+.*", line):
                    values = re.split(" |,", line) # split line
                    values = [safe_eval(v) for v in values] # evaluate
                    # set first value to nested dict_key
                    if len(values) >= 3:
                        result_dict[current_key][values[0]] = values[1:]
                    else:
                        result_dict[current_key][values[0]] = values[1]
                else:
                    break
    return result_dict

def get_param(file):
    """
    Read in the parameters from the stage file. Change format and keys.
    Args:
        file (str): path to coordinate file from Glowtracker
    Returns:
        param (dict): Updated parameter dictionary.
    """
    result_dict = get_stage_dict(file)
    
    param = {
        'scale':result_dict['Camera']['pixelsize'], 
        'fps':result_dict['Camera']['framerate'], 
        'imagenormaldir':result_dict['Camera']['imagenormaldir'],
        'use_matrix':True if 'imageToStage' in result_dict['Camera'].keys() else False,
        'width':result_dict['Tracking']['roi_x'], 
        'height':result_dict['Tracking']['roi_y'],
    }
    
    return param

def get_matrix(coordsfile, imagenormaldir='Z'):
    """
    Read in the image-to-stage rotation matrix. With hardcoded reflection matrix! TODO should not be hardcoded!
    Args:
        coordsfile (str): path to coordinate file from Glowtracker
    Returns:
        m (np.array): rotaion matrix to rotate recorded images by.
    """
    # rotate centerlines using matrix
    result_dict = get_stage_dict(coordsfile)
    m = np.array(result_dict['Camera']['imageToStage']).reshape(2,2)
    if imagenormaldir == '-Z':
        flip_y = np.array([[1,0],[0,-1]]) # reflection matrix
        m = m @ flip_y # order matters. first rotation then reflect
    
    return m

def rotate_3rdaxis(calibrationMatrix, arr, param): #im for image and ratio_stage_image_coordinates to 
    """
    Takes a stage calibration matrix and centerline coordinates in px.
    Transforms it into image coordinates (um), by applying a rotation to the third axis
    Args:
        calibrationMatrix (np.array): rotaion matrix, with scale and reflection if needed.
        arr (np.array): array based on recording images in pixel coordinates. Array with shape (frames, points, xy).
    Returns:
        rot (np.array): updated array.
    """
    cent = arr - np.stack([param['width']//2,param['height']//2])
    rot = np.einsum('ij,lkj->ikl', calibrationMatrix, cent)
    return  rot

def reformat_coordinatefile(stage_fn, dlc_df, cl_org):
    """
    Find a txt file in the input path and recalculate coordinates in useful units. 
    Also add signals from image processing
    Args:
        stage_fn (str): path to stage_coordinate file 
        dlc_df (pd.DataFrame): DataFrame object from poseestimation with bodypart coordinates in pixel, multiindex [header, bodyparts, [x,y,likelihood]]
        cl_org (pd.DataFrame): DataFrame object with coordinates of centerline in pixel, multiindex [,[x,y]]
    Returns:
        worm (pd.DataFrame): DataFrame object with rotated coordinates for each bodypart
        tracks (pd.DataFrame): DataFrame object with stage coordinates in mm (X,Y), in um (Xstage,Ystage), actual center of mass (Xcms, Ycms)
        cl_rot (pd.DataFrame): DataFrame object with rotated centerline coordinates
        matrix (np.array): rotation matrix used
    """
    idx = pd.IndexSlice
    param = get_param(stage_fn) # get param

    # get the header for tracks from line 27
    with open(stage_fn, 'r') as f:
        lines = f.readlines()
        header = lines[27].strip().strip('#').split() # Remove the '#' character and any leading/trailing whitespace
    # Read the rest of the data into a DataFrame
    tracks = pd.read_csv(stage_fn, delimiter=' ', skiprows=28, usecols=range(len(header)), header=None)
    tracks.columns = header # Assign the processed header to the DataFrame

    # smooth X and Y with savgol-filter
    tracks.loc[:,idx['X','Y']] = savgol_filter(tracks.loc[:,idx['X','Y']], 20, 2, axis=0)

    # translate stage from mm to um
    tracks.loc[:,'Xstage'] = tracks.loc[:,'X']*1000 # in um
    tracks.loc[:,'Ystage'] = tracks.loc[:,'Y']*1000 # in um
    tracks = tracks.iloc[dlc_df.index]

    # transform pixel data to um data and get rotation matrix
    if param['use_matrix']:
        matrix = get_matrix(stage_fn, param['imagenormaldir']) # scale included in rotation matrix
    else:
        matrix = np.identity(2)*param['scale'] # use a default matrix that doesn't rotate
    
    ### dlc_df
    # make 3d array for dlc_df
    dlc_arr = np.stack([dlc_df.loc[:,idx[:,:,'x']].values, dlc_df.loc[:,idx[:,:,'y']].values])
    dlc_arr = np.moveaxis(dlc_arr,0,-1)
    # transform coordinates with rotation matrix
    xworm, yworm = rotate_3rdaxis(matrix, dlc_arr, param)
    # create worm dataframe in shape of dlc_df
    worm = pd.DataFrame([]).reindex_like(dlc_df).drop('likelihood', axis=1, level=2)
    worm = worm.reset_index(names=['frame'], level=0)
    worm.loc[:,'time'] = worm.loc[:,'frame']/param['fps']
    # add stage offset
    worm.loc[:,idx[:,:,'x']] = pd.DataFrame(xworm).add(tracks.loc[:,'Xstage']).T.values
    worm.loc[:,idx[:,:,'y']] = pd.DataFrame(yworm).add(tracks.loc[:,'Ystage']).T.values
    # cacluate cms for worm from all bps
    tracks.loc[:,'Xcms'] = worm.loc[:,idx[:,:,'x']].mean(axis=1)
    tracks.loc[:,'Ycms'] = worm.loc[:,idx[:,:,'y']].mean(axis=1)
    
    ### centerline
    # make 3d array for centerline
    cl_arr = np.stack([cl_org.loc[:,idx[:,'x']].values, cl_org.loc[:,idx[:,'y']].values])
    cl_arr = np.moveaxis(cl_arr,0,-1)
    # transform coordinates with rotation matrix
    cl_x, cl_y = rotate_3rdaxis(matrix, cl_arr, param)
    # create worm dataframe in shape of cl_org
    cl_rot = pd.DataFrame([]).reindex_like(cl_org).drop('likelihood', axis=1, level=1)
    # add stage offset
    cl_rot.loc[:,idx[:,'x']] = pd.DataFrame(cl_x).add(tracks.loc[:,'Xstage']).T.values
    cl_rot.loc[:,idx[:,'y']] = pd.DataFrame(cl_y).add(tracks.loc[:,'Ystage']).T.values
    
    return worm, tracks, cl_rot, matrix

def get_bouts(data):
    """
    Create a dictionary onsets and duration of bouts, represented as binary 
    Args:
        data (pd.DataFrame): DataFrame object with binary data
    Returns:
        on_dur (dict): dictionary with key: (onset, duration),... for each column in the DataFrame
    """
    # Find the start and end indices of binary bouts
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    on_dur = {}
    for col in data.columns:
        d = data[col]
        starts = np.where((d == 1) & (d.shift(1, fill_value=0) == 0))[0] #indices if current element 1 and previous 0
        ends = np.where((d == 1) & (d.shift(-1, fill_value=0) == 0))[0] #indices if current element 1 and next 0
        dur = ends - starts #bout duration
        on_dur[col] = list(zip(starts, dur)) # save start and duration in dict
    
    return on_dur


def ethogram_fromOnOff(on=None, off=None, onoff_tuples=None, arr_len=None, offAsDuration=False):
    """
    Creates a binary ethogram from the input
    Args:
        on (list): optional, list with onsets, default None, must be set if onoff_tuples is None
        off (list): optional,  list with offsets, default None, must be set if onoff_tuples is None
        onoff_tuples (list): optional, list of tuples with (on, off), default None, must be set if on/off is None
        arr_len (int): optional, desired length of output array
        offAsDuration (bool): if True offset is handled as Duration
    Returns:
        arr (np.array): binary array
    """
    if onoff_tuples is None:
        try:
            onoff_tuples = list(zip(on.astype(int), off.astype(int))) # create list with tuples from on and off
        except:
            raise ValueError("on and off must be set if onoff_tuple is None")
        
    if arr_len is None: #set arr_len if not provided
        arr_len = (np.max(np.array(onoff_tuples)[:,0]) + np.max(np.array(onoff_tuples)[:,1])) if offAsDuration else np.max(np.array(onoff_tuples)[:,1])
        arr_len = int(arr_len)

    arr = np.zeros(arr_len) #init array with zeros
    # create array with indices from onoff_tuples
    if offAsDuration:
        peaks_index = np.array(list(chain.from_iterable(range(start, start + end + 1) for start, end in onoff_tuples))) 
    else:
        peaks_index = np.array(list(chain.from_iterable(range(start, end + 1) for start, end in onoff_tuples)))
    # set bout indices to 1
    arr[peaks_index.astype(int)] = 1
    return arr

def calculate_mean_values(df_on, df_dur, df_values):
    """
    Calculate the mean of bouts from timeseries dataframe by providing onset and duration of bouts
    Args:
        df_on (pd.DataFrame): DataFrame containing onsets with shape (N,M), N: number of bouts, M: objects showing bouts
        df_dur (pd.DataFrame): DataFrame like df_on only for bout duration
        df_values (pd.DataFrame): DataFrame with values to take mean over, with shape (n,M), n: indexer/frames
    Returns:
        df_mean (pd.DataFrame): DataFrame containing means per bout with shape (N,M)
    """
    # init dataframe with index like df_on, i.e. one row per bout
    df_mean = pd.DataFrame([]).reindex_like(df_on)
    # create array from df_values.index with axis 1 like df_on
    df_index = np.repeat(df_values.index.values[:,np.newaxis], df_on.shape[1], axis=1)
    # create array from df_values with axis 1 like df_on
    df_values = np.repeat(df_values.values[:,np.newaxis], df_on.shape[1], axis=1)
    # iterate over bouts
    for (i, row_on), (j, row_dur) in zip(df_on.iterrows(), df_dur.iterrows()):
        mask = (df_index >= row_on.values) & (df_index <= (row_on+row_dur).values) # create mask for bout
        # iterate over axis 1
        for k in range(mask.shape[1]):
            l = df_on.columns[k] # current column
            m = mask[:,k] # adjust mask to column            
            if df_values[m,k].shape[0] < 1:
                continue
            mean_value = np.mean(df_values[m,k]) # take mean with mask, specific to columns
            df_mean.loc[i, l] = mean_value # assign mean

    return df_mean

def calculate_mode_values(df_on, df_dur, df_values):
    """
    Calculate the mode of bouts from timeseries dataframe by providing onset and duration of bouts
    Args:
        df_on (pd.DataFrame): DataFrame containing onsets with shape (N,M), N: number of bouts, M: objects showing bouts
        df_dur (pd.DataFrame): DataFrame like df_on only for bout duration
        df_values (pd.DataFrame): DataFrame with values to take mean over, with shape (n,M), n: indexer/frames
    Returns:
        df_mean (pd.DataFrame): DataFrame containing means per bout with shape (N,M)
    """
    # init dataframe with index like df_on, i.e. one row per bout
    df_mean = pd.DataFrame([]).reindex_like(df_on).astype(df_values.dtypes)
    # create array from df_values.index with axis 1 like df_on
    df_index = np.repeat(df_values.index.values[:,np.newaxis], df_on.shape[1], axis=1)
    # create array from df_values with axis 1 like df_on
    df_values = np.repeat(df_values.values[:,np.newaxis], df_on.shape[1], axis=1)
    # iterate over bouts
    for (i, row_on), (j, row_dur) in zip(df_on.iterrows(), df_dur.iterrows()):
        mask = (df_index >= row_on.values) & (df_index <= (row_on+row_dur).values) # create mask for bout
        # iterate over axis 1
        for k in range(mask.shape[1]):
            #if df_index[row_on,k] == np.nan:
            l = df_on.columns[k] # current column
            m = mask[:,k] # adjust mask to column
            if df_values[m,k].shape[0] < 1:
                continue
            mean_value = pd.DataFrame(df_values[m,k]).mode() # take mean with mask, specific to columns
            df_mean.loc[i, l] = mean_value.iloc[0,0] # assign mean
    return df_mean

def mode_chunks(data, max_range):
    """
    Performs mode on bout chunks that are shorter than max_range.
    Args:
        data (array or DataFrame): Data to perform mode on
        max_range (int): maximum range/duration of bouts to perform mode on
    Returns:
        new_data (pd.DataFrame): mode data
    """
    # ensure DataFrame and object
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data = data.astype(object)

    new_data = pd.DataFrame([]).reindex_like(data) # init DataFrame
    # iterate over columns
    for col in data.columns:
        d = data[col]
        starts = np.where(d.shift(1) != d)[0] # starts when previous different than current
        ends = np.where(d.shift(-1) != d)[0] # ends when next different than current
        durs = ends - starts # bout duration
        values = d[starts]  # bout label
        index = np.arange(len(starts)) # index over bouts
        df = pd.DataFrame([starts,ends,durs,values,index], index=['starts','ends','durs','values','index']).T

        bool_short = durs <= max_range # bouts that are shorter than max_range
        grouper = np.cumsum( np.append([1],np.diff(index[bool_short])) != 1 ) # creates a grouper variable based on chunk membership, seperated by bouts bigger than max_range

        chunks = df[bool_short].groupby(grouper) # groups chunks
        new_df = pd.DataFrame([])
        for n, chunk in chunks:
            chunk = chunk.sort_values('durs')
            mode_chunk = chunk.groupby('values').sum() # overall duration of bout type
            winner = np.argmax(mode_chunk['durs']) # index of longest overall bout type
            # set results in new_df
            new_df = pd.concat([new_df,
                                pd.DataFrame([chunk['starts'].min(),chunk['ends'].max(),chunk['durs'].sum(),mode_chunk.index[winner]],
                                            index=['starts','ends','durs','values']).T],
                                axis=0)
        # reinsert bouts over max_range, sortvalues according to starts
        new_df = pd.concat([new_df, df[~bool_short].drop('index', axis=1)]).sort_values('starts').reset_index(drop=True)
        # reindex based on starts and fill in gaps
        new_df = new_df.set_index('starts').reindex(range(len(data)), method='ffill')['values'].values
        new_data[col] = new_df
    return new_data

"""
def ethogram_minrange(data_org, min_range=10):
    data = data_org.copy()
    # Find the start and end indices of binary bouts
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data = data.astype(object)
    for col in data.columns:
        d = data[col]
        starts = np.where((d == 1) & (d.shift(1, fill_value=0) == 0))[0]
        ends = np.where((d == 1) & (d.shift(-1, fill_value=0) == 0))[0]
        return starts, ends
        dur = ends - starts
        onsets, offsets = starts[dur < min_range], ends[dur < min_range]
        index = np.concatenate([list(range(st, et+1)) for st, et in zip(onsets, offsets)])
        data.loc[index, col] = np.nan
    return data

# Define a custom function to find the most common string in a window
def most_common_string(window):
    counter = Counter(window)
    most_common = counter.most_common(1)[0][0]
    return most_common

# Create a custom rolling window function
def rolling_most_common(df, window_size):
    result = []
    for i in range(len(df) - window_size + 1):
        window = df[i:i + window_size]
        result.append(most_common_string(window))
    return [None] * (window_size - 1) + result

def ethogram_fromOnOff(left_ips, right_ips, arr_len=None):
    '''
    peaks: peaks as found by scipy_find_peaks
    left_ips: parameter of peaks returned when width is set
    right_ips: parmater of peaks returned when width is set
    '''

    if arr_len is None:
        arr_len = np.max(right_ips).astype(int)
    arr = np.zeros(arr_len)
    peaks_tuple = list(zip(left_ips.astype(int), right_ips.astype(int)))
    peaks_index = np.array(list(chain.from_iterable(range(start, end + 1) for start, end in peaks_tuple)))
    arr[peaks_index] = 1
    return arr
"""