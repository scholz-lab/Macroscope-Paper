import numpy as np
import pandas as pd
import re
from scipy.stats.mstats import ttest_rel
import matplotlib.pylab as plt
from matplotlib.pyplot import cm


def info_get_prevstimulus(info, movie):
    mov = movie.replace('c','')
    mov_info = info.loc[mov]
    add_stim = 0
    same_plate = mov_info['same plate as']
    if isinstance(same_plate,str) and same_plate < mov:
        add_stim += info.loc[same_plate]['stim N']
        add_stim += info_get_prevstimulus(info, same_plate)
    return add_stim
    
class CalciumPlotter():
    """
    Args:
        df_multi
        stim_dur: in seconds
        ISI: in seconds
        repeats: 
        show_stim_edge:
        fps:
        stim_before: Not implemented
        sig_type: str
        trace1: main trace used and used for mean_signal, except when normalize==True, then trace2/mean(trace2) / trace3/mean(trace3)
        trace2: 
        trace3: 
        window: 
    """
    def __init__(self, df_multi, stim_dur = 1, ISI = 30, repeats = 5, show_stim_edge = 10, fps =30, stim_before=0, sig_type='mean', 
                 trace1='sig/ref', trace2='sig', trace3='ref', window=None):
        self.stim_dur = stim_dur
        self.ISI = ISI
        self.repeats = repeats
        self.show_stim_edge = show_stim_edge
        self.fps = fps
        self.stimulus_onoff = np.cumsum(np.tile([ISI,stim_dur],repeats)).reshape(repeats,2)*self.fps
        self.t_stimulus = self.stimulus_onoff[:,0]
        self.stim_before = stim_before
        self.df_multi = df_multi
        self.window = window
        self.sig_type = sig_type
        self.trace1 = trace1
        self.trace2 = trace2
        self.trace3 = trace3
        self.set_trace()

    def set_trace(self):
        self.trace1 = self.trace1 if self.sig_type is None else (self.trace1, self.sig_type)
        self.trace2 = self.trace2 if self.sig_type is None else (self.trace2, self.sig_type)
        self.trace3 = self.trace3 if self.sig_type is None else (self.trace3, self.sig_type)

    def crop_df(self, df, mov):
        if self.window is not None:
            mov_window = self.window[mov]
            df = df.loc[range(*mov_window)]
        return df

    def signal_trace(self, column_level_0, xtick_dist=10, figsize=(20,2), rolling_mean = 15, ):
        fig, ax = plt.subplots(1, figsize=(20,2))
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines.right.set_position(("axes", 1.05))
        ax.set_zorder(5)
        ax.patch.set_alpha(0)

        df = self.crop_df(self.df_multi[column_level_0], column_level_0)
        ax.plot(pd.DataFrame(df[self.trace1]).rolling(rolling_mean).mean(), c='k')
        ax2.plot(pd.DataFrame(df[self.trace2]).rolling(rolling_mean).mean(),c='green')
        ax3.plot(pd.DataFrame(df[self.trace3]).rolling(rolling_mean).mean(),c='red')

        ax.spines['right'].set_visible(False)
        ax.set_ylabel(f'{self.trace1 if isinstance(self.trace1, str) else ' '.join(self.trace1)}')
        ax2.spines['right'].set_color('green')
        ax2.set_ylabel(f'{self.trace2 if isinstance(self.trace2, str) else ' '.join(self.trace2)}')
        ax3.spines['right'].set_color('red')
        ax3.set_ylabel(f'{self.trace3 if isinstance(self.trace3, str) else ' '.join(self.trace3)}')
        
        ylim = ax.get_ylim()
        for p in self.stimulus_onoff:
            ax.broken_barh([(p[0],p[1]-p[0])], ylim, facecolor='#27213C', label='stimulus', alpha=.5)
            
        ax.set_xticks(np.arange(0,len(self.df_multi),xtick_dist*self.fps))
        ax.set_xticklabels(np.arange(0,len(self.df_multi),xtick_dist*self.fps)//self.fps)
        return fig


    def create_alignment(self, normalize=False): 
        self.df_AlignStim = pd.DataFrame([])
        
        for i, (name, df) in enumerate(self.df_multi.T.groupby(level=0)):
            df = df.T
            df = self.crop_df(df, name)
            for st in self.t_stimulus:
                start, end = st, st+self.stim_dur
                pre, post = start-self.show_stim_edge*self.fps, end + self.show_stim_edge*self.fps
                d = df[pre:post].reset_index(drop=True)#.rolling(rolling_mean, min_periods=1).mean()

                #d.columns = pd.MultiIndex.from_product([[name],[st],d.columns.get_level_values(-1)])
                d = insert_level_multiindex(d, st, level=1)
                d = d.reset_index(drop=True)
                
                if not int(self.show_stim_edge*self.fps * 0.5) in d.index or not int(self.show_stim_edge*self.fps * 1.5) in d.index :
                    continue
                
                self.df_AlignStim = pd.concat([self.df_AlignStim, d], axis=1)

        if self.df_AlignStim.empty:
            print('Not enough data for alignment')
            return self.df_AlignStim

        # index
        ensure_tup = self.trace1 if isinstance(self.trace1, tuple) else [self.trace1]
        trace_level = [np.where(self.df_AlignStim.columns.to_frame()==col)[1][0] for col in ensure_tup]
        
        # normalize
        if normalize:
            trace2_mean = np.nanmean(self.df_AlignStim.loc[:300,(slice(None),)*trace_level[0]+self.trace2], axis=0)
            trace2_norm = (self.df_AlignStim.loc[:,(slice(None),)*trace_level[0]+self.trace2]-trace2_mean) / trace2_mean + 1
            
            trace3_mean = np.nanmean(self.df_AlignStim.loc[:300,(slice(None),)*trace_level[0]+self.trace3], axis=0)
            trace3_norm = (self.df_AlignStim.loc[:,(slice(None),)*trace_level[0]+self.trace3]-trace3_mean) / trace3_mean + 1
            
            R = trace2_norm.droplevel(trace_level, axis=1) / trace3_norm.droplevel(trace_level, axis=1)+1
            
            self.norm = R - np.nanmean(R.loc[:300]) / np.nanmean(R.loc[:300])
            
            self.trace1 = 'dR/R0'
            self.set_trace()
            
            multiCol = pd.concat([pd.DataFrame([*self.norm.columns]), pd.DataFrame(np.repeat([self.trace1], self.norm.shape[1], axis=0), columns=trace_level)], axis=1)
            self.norm.columns = pd.MultiIndex.from_frame(multiCol)
            self.df_AlignStim = pd.concat([self.df_AlignStim, self.norm], axis=1)
                
        return self.df_AlignStim
        
    def mean_signal(self, rolling_mean = 10, aligned_df = None, ylim=None, normalize=False, sem_error=True, xtick_dist=5, axs=None, meanlabel=None):
        if aligned_df is None:
            aligned_df = self.create_alignment(normalize=normalize)
                
        #mean_signal = aligned_df.T.groupby(level=list(range(1, aligned_df.columns.nlevels))).mean().T[self.trace1]
        #std_signal = aligned_df.T.groupby(level=list(range(1, aligned_df.columns.nlevels))).std().T[self.trace1]

        if self.df_AlignStim.empty:
            return
        
        # index
        ensure_tup = self.trace1 if isinstance(self.trace1, tuple) else [self.trace1]
        trace_level = [np.where(aligned_df.columns.to_frame()==col)[1][0] for col in ensure_tup]

        # smooth
        aligned_df = aligned_df.rolling(rolling_mean, min_periods=1).mean()

        # colors
        m_color = cm.rainbow(np.linspace(0, 1, len(self.df_multi.columns.get_level_values(0).unique())))
        level0color = dict(zip(aligned_df.columns.get_level_values(0).unique(), m_color))

        # get mean and std
        mean_signal = aligned_df.T.groupby(level=trace_level).mean().T[self.trace1]
        error_signal = aligned_df.T.groupby(level=trace_level).std().T[self.trace1]
        if sem_error:
            error_signal = error_signal / np.sqrt(len(aligned_df.columns.get_level_values(0).unique()))

        if axs is None:
            fig, axs = plt.subplots(1,2, figsize=(10,4))

        line_list = {}
        for mov_stim in aligned_df.droplevel(trace_level, axis=1):
            d = aligned_df[mov_stim]
            line, = axs[1].plot(d[self.trace1], c=level0color[mov_stim[0]])
            
            if mov_stim[0] not in line_list.keys():
                line_list[mov_stim[0]] = line
            
        axs[0].plot(mean_signal, linewidth= 1, label=meanlabel)
        axs[0].fill_between(range(len(aligned_df)), mean_signal+error_signal, mean_signal-error_signal, alpha=.5, zorder=2)
        if ylim is None:
            ylim = axs[0].get_ylim()
        print(ylim)
        axs[0].bar(self.show_stim_edge*self.fps, ylim[1]-ylim[0], bottom=ylim[0], width=self.stim_dur*self.fps, fc='darkgrey', align='edge', alpha=.5, zorder=0)
        axs[0].set_ylabel(f'mean {self.trace1 if isinstance(self.trace1, str) else ' '.join(self.trace1)}')
        
        axs[1].plot(mean_signal, linewidth= 1, c='k')
        ax1_ylim = axs[1].get_ylim()
        axs[1].bar(self.show_stim_edge*self.fps, ax1_ylim[1], bottom=ax1_ylim[0], width=self.stim_dur*self.fps, fc='darkgrey', align='edge', alpha=.5, zorder=0)
        axs[1].set_ylim(ax1_ylim)
        
        if ylim is not None:
            axs[0].set_ylim(ylim)
            
        if not meanlabel is None:
            axs[0].legend()
        axs[1].legend(line_list.values(), line_list.keys(), loc='upper left', bbox_to_anchor=(1,1))
        for ax in axs:
            ax.set_xticks(np.arange(0,len(aligned_df),xtick_dist*self.fps))
            ax.set_xticklabels(np.arange(0,len(aligned_df),xtick_dist*self.fps)//self.fps)
            ax.set_xlabel('Time (s)')
        
        try:
            fig
        except NameError:
            pass
        else:
            return fig
        
def insert_level_multiindex(df, value, level=0):
    index_df = df.columns.to_frame(index=False)
    index_df.columns = index_df.columns+1
    index_df.insert(level, 'new', value)
    index_df.columns = range(0, index_df.shape[1])
    
    # Set new index
    df.columns = pd.MultiIndex.from_frame(index_df)
    return df

def stats_df(df, range_a, range_b, alignment, filter, df_out=None):
    df = df.copy()
    df = df.dropna(axis=1)
    
    mean_a = df.loc[range_a].mean()
    mean_b = df.loc[range_b].mean()
    
    t, p = ttest_rel(mean_a,mean_b)
    cohens_d = (mean_b.mean() - mean_a.mean())/mean_b.std()
    N_animals = len(df.columns.get_level_values(0).unique())
    N_traces = df.shape[1]
    
    #common_cols = list(reduce(set.intersection, map(set, df.columns)))
    repeated = [set(m) for m in df.columns]
    common_cols = list(repeated[0].intersection(*repeated))
    
    _ = pd.DataFrame([[common_cols, alignment, filter, 
                       f'{range_a[0]}-{range_a[-1]+1}', 
                       f'{range_b[0]}-{range_b[-1]+1}', 
                       t, p, cohens_d,
                       N_animals, N_traces]], 
                     columns = ['datatype', 'aligned', 'filter', 'frames a', 'frames b', 't-statistic', 'p-value', 'cohens d', 'N_animals', 'N_traces'])
    if df_out is not None:
        return pd.concat([df_out, _], axis=0).reset_index(drop=True)
    else:
        return _

def str_to_tuple(s):
    return tuple(map(int, re.split('-|, |,', s)))

def angle(row):
    v1 = [row.X, row.Y]
    v2 = [row.X1, row.Y1]
    return np.degrees(np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))

def calculate_reversals(data, animal_size, animal_ratio, angle_threshold, scale, fps=30):
    """Adaptation of the Hardaker's method to detect reversal event. 
    A single worm's centroid trajectory is re-sampled with a distance interval equivalent to 1/10 
    of the worm's length (100um) and then reversals are calculated from turning angles.
    Inputs:
        animal_size: animal size in um.
        angle_threshold (degree): what defines a turn 
    Output: None, but adds a column 'reversals' to data.
    """
    data = data.copy()

    cummul_distance = np.cumsum(data['velocity_smooth']/fps)
    sample_distance = np.arange(np.min(cummul_distance), np.max(cummul_distance), animal_size/animal_ratio)

    sample_indices = pd.DataFrame(abs(cummul_distance.values - sample_distance[:, np.newaxis])).T
    sample_indices = sample_indices.idxmin(axis=0).values

    # create a downsampled trajectory from these indices
    traj_Resampled = data.loc[sample_indices, ['X', 'Y']].diff()
    # we ignore the index here for the shifted data
    traj_Resampled[['X1', 'Y1']] = traj_Resampled.shift(1).fillna(0)

    # calculate angle
    traj_Resampled['angle'] = 0
    traj_Resampled['angle']= traj_Resampled.apply(lambda row: angle(row), axis=1)
    
    rev = traj_Resampled.index[traj_Resampled.angle>=angle_threshold]
    data.loc[:,'angle'] = traj_Resampled['angle']
    data.loc[:,'reversals'] = 0
    data.loc[rev,'reversals'] = 1
    return data, sample_indices