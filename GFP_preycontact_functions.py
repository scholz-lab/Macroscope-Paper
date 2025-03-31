import numpy as np
import pandas as pd

import pims
from skimage.measure import regionprops
from scipy.stats.mstats import ttest_rel

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
from matplotlib.lines import Line2D

@pims.pipeline
def im_props(im_crop, mask):
    """
    Calculate the maximum, mean intensity, cms and skew of the image.
    Args:
    Returns:
        max intensity, mean intensity, 5 percentile of intensity, 95 percentile intensity, mask size
    """
    # image properties
    props = regionprops(mask, intensity_image = im_crop)[0]
    return props.intensity_max, props.intensity_mean, *np.percentile(props.image_intensity, [5,95]), np.sum(mask)
    
def run_processing(image, fr, mask, yhat):
    props = im_props(image, mask)
    return [fr, *props, yhat]

@pims.pipeline
def merge_falsecolor(images, rgb_values, brightness_factors, avoid_oversat=True):
    all_images = []
    for img, rgb, brighter in zip(images, rgb_values, brightness_factors):
        img = increase_brightness(img, brighter, avoid_oversat=avoid_oversat)
        img = grayscale_to_color(img, rgb)
        all_images.append(img)
        
    # Merge the images by summing them up
    merged = np.zeros_like(all_images[0])
    for img in all_images:
        merged += img
    
    return np.clip(merged, 0, 255).astype(np.uint8)

@pims.pipeline
def increase_brightness(image, factor, avoid_oversat=True):
    # Ensure the image is in the correct range after brightness adjustment
    img = np.array(image).astype(float)
    if avoid_oversat and np.max(img)*factor >= 255:
        factor = 255/np.max(img)
    img = img * factor
    return np.clip(img,0,255).astype(np.uint8)

@pims.pipeline
def grayscale_to_color(image, target_color):
    img_array = np.array(image)
    img_array = img_array/255
    color_image = (img_array[..., np.newaxis] * target_color)

    return color_image
    

def get_stats(df, range_a, range_b, level_animals=0, level_traces=-1):
    mean_a = df.loc[range_a].mean()
    mean_b = df.loc[range_b].mean()
    
    t, p = ttest_rel(mean_a,mean_b, alternative='greater')
    
    cohens_d = (mean_b.mean() - mean_a.mean())/mean_b.std()
    N_animals = len(df.columns.get_level_values(level_animals).unique())
    N_traces = len(df.columns.get_level_values(level_traces))

    return (t, p, cohens_d, N_animals, N_traces)

def CLtrajectory_plotter(CLine, XY, y, cluster_color, cluster_label, fn='', figsize=(10,10)):
    fig, ax = plt.subplots(figsize=(10,10))
    legend_elements = [Line2D([0], [0],color=cluster_color[i], label=cluster_label [i]) for i in cluster_label]
    adjustCL = (CLine-np.nanmean(CLine))+np.repeat(XY.reshape(XY.shape[0],1,XY.shape[1]), CLine.shape[1], axis=1)-np.nanmean(XY, axis=0)# fits better than subtracting 50
    #adjustXY = XY-np.nanmean(XY, axis=0)
    for l in np.unique(y).astype(int):
    #for l in [2,3,5,8]:#[1,2,6,7]#[2,3,5,8]
        #if l != 6:
        il = np.where(y == l)[0]
        ax.plot(*adjustCL[il].T, c=cluster_color[l], alpha = 0.1,)#cluster_color[l]
            #plt.scatter(XY[:,0][il],XY[:,1][il], marker=".", lw=2, c=bar_c[l], alpha=0.1)
    ax.set_title(fn)
    ax.axis('equal')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1,1))
    return fig

class EthogramPlotter():
    def __init__(self, df, cluster_color, cluster_label, fps, plot_fps=None, xtick_spread = 30, multi_level=0):
        self.df = df
        self.fps = fps
        self.plot_fps = fps if plot_fps is None else plot_fps
        
        self.cluster_color = cluster_color
        self.cluster_label = cluster_label
        self.xtick_spread = xtick_spread
        
        self.init_color()

    def init_color(self):
        colors = [c for c in self.cluster_color.values()]
        self.cmap_cluster = mpl.colors.ListedColormap(colors, name='cluster', N=None)
        
    def stacked(self, data=None, ax=None, cbar = False, figsize=(4,5)):
        if ax is None:
            f, ax = plt.subplots(1, figsize=figsize)
        if data is None:
            data = self.df.copy()
        data = data[::int(self.fps/self.plot_fps)].T
        data = data.fillna(-1)
        #
        timeinsec = np.arange(data.shape[1]/self.plot_fps)
        # set limits .5 outside true range
        mat = ax.imshow(data, cmap=self.cmap_cluster, vmin=-1, 
                        vmax=5, aspect='auto', 
                        #extent = (min(timeinsec), max(timeinsec),0,data.shape[0]), 
                        origin='lower', interpolation='nearest')
        #print(range(len(timeinsec))[::self.xtick_spread*self.fps])
        ax.set_xticks(np.arange(0, data.shape[1]+1, self.xtick_spread*self.fps))
        ax.set_xticklabels(np.arange(min(timeinsec), np.max(timeinsec)+self.xtick_spread, self.xtick_spread).astype(int))
        return mat

    def multi_stack(self, adaptive_figsize=(4,5), xlim=(None,None), ylim=(None,None), multi_level=0):
        self.multi_level = multi_level
        self.conditions = self.df.columns.get_level_values(multi_level).unique()
        f, ax = plt.subplots(1,len(self.conditions), figsize=tuple(np.multiply(adaptive_figsize,(len(self.conditions),1))))
        if len(self.conditions) <= 1:
            ax = [ax]
        for i,cond in enumerate(self.conditions):
            self.stacked(self.df[cond], ax[i])
            ax[i].set_title(cond)
        ax[0].set_ylabel('Tracks')
        ax[0].set_yticks(range(self.df[cond].shape[1]))
        ax[0].set_yticklabels(self.df[cond].columns)
        #plt.setp(ax,xlim=xlim, ylim=ylim, xlabel= 'Time (s)')
        return f

    def single(self, y_column, metrics=[], smooth=30, adaptive_figsize=(20,2)):
        plot_col = self.df[y_column].copy().dropna().T
        plot_col = plot_col[::int(self.fps/self.plot_fps)]
        #onoff = proc.onoff_dict(plot_col, labels=np.unique(plot_col))
        
        timeinsec = np.arange(min(plot_col.columns)/self.plot_fps, max(plot_col)/self.plot_fps)
        fig, axs = plt.subplots(1+len(metrics), 1, 
                                figsize=tuple(np.multiply(adaptive_figsize,(1, 1+len(metrics)))),
                                constrained_layout=True, sharex=True)
        if not isinstance(axs,np.ndarray):
            axs = [axs]
    
        axs[0].imshow(plot_col, cmap=self.cmap_cluster, vmin=-1, 
                        vmax=5, aspect='auto', 
                        origin='lower')
        for i,met in enumerate(metrics):
            met = met.reset_index(drop=True)
            axs[i+1].plot(met.rolling(smooth, min_periods=1).mean(),c='k')
        for ax in axs:
            ax.set_xticks(np.arange(0, plot_col.shape[1], self.xtick_spread*self.fps))
            ax.set_xticklabels(np.arange(min(timeinsec), max(timeinsec), self.xtick_spread).astype(int))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(5*self.plot_fps))
        axs[-1].set_xlabel('sec')
        
        plt.legend(handles=[Patch(facecolor=self.cluster_color[i]) for i in np.unique(plot_col).astype(int)],
            labels=[self.cluster_label[k] for k in np.unique(plot_col)],
            ncol=3, loc='upper left',
            bbox_to_anchor=(0, -0.5))
        fig.suptitle(f'Ethogram of {y_column}',fontsize=16)
        return fig
    