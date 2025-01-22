#%%

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress, sem
from scipy.signal import savgol_filter
import os
from random import randrange
from matplotlib.animation import FuncAnimation

%matplotlib tk


#path = 'C:\\Users\\landgrafn\\NFCyber\\WalterWhities\\data\\'
path = 'C:\\Users\\landgrafn\\Desktop\\feet\\'
common_name = '.csv'

x_pixel, y_pixel = 570, 570
arena_length = 400          # in mm
fps = 100


all_bodyparts = ['snout', 'mouth', 'paw_VL','paw_VR', 'middle', 'paw_HL', 'paw_HR','anus', 'tail_start', 'tail_middle', 'tail_end']
center_bodyparts = ['middle']
centerline_bodyparts = ['tail_start', 'anus', 'middle']
paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
colors = ['b', 'orange', 'r', 'c', 'm', 'y']
#colors = ['r', 'c', 'm', 'y']


def get_files(path):
    # get files

    files = [path+file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    print(f'{len(files)} files found\n')

    return files
file_list = get_files(path)


#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# FUNCTIONS

# basic functions
def euclidian_dist(point1x, point1y, point2x, point2y):
    # calculates the euclidian distance between 2 bps in mm

    try:
        distance_mm = np.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)
    except:
        print('ERROR: euclidian_dist was not working')
    
    return distance_mm
def point2line_dist(slope, intercept, point_x, point_y):
    # returns the shortest distance between a line (given slope and intercept) and a xy point

    A, B, C = slope, -1, intercept
    distance = (abs(A*point_x + B*point_y + C) / np.sqrt(A**2 + B**2))

    return distance
def timestamps_onsets(series, min_duration_s, onset_cond=float):
    # you have a series with conditions (e.g. True/False or float/nan ->change for onset_cond!) in each row
    # return the start & stop frame of the [cond_onset, cond_offset]
    min_frame_nb = min_duration_s * fps

    timestamp_list = []
    start_idx, prev_idx = None, None
    
    # get on- and offset frames between switches between True & False
    if onset_cond == True:
        for idx, val in series.items():
            if (val == onset_cond) and (start_idx == None):
                start_idx = idx
            
            elif (val != onset_cond) and (start_idx != None):
                if idx - start_idx >= min_frame_nb:
                    timestamp_list.append((start_idx, prev_idx))
                start_idx = None
            prev_idx = idx

        # for last value
        if start_idx is not None:
            if series.index[-1] - start_idx >= min_frame_nb:
                timestamp_list.append([start_idx, series.index[-1]])
    
    # get on- and offset frames between switches between np.nan & float numbers
    elif onset_cond == float:
        for idx, val in series.items():
            if not np.isnan(val) and (start_idx == None):
                start_idx = idx
            
            elif np.isnan(val) and (start_idx is not None):
                if idx - start_idx >= min_frame_nb:
                    timestamp_list.append((start_idx, prev_idx))
                start_idx = None
            prev_idx = idx

        # for last value
        if start_idx is not None:
            if series.index[-1] - start_idx >= min_frame_nb:
                timestamp_list.append([start_idx, series.index[-1]])
    
    return timestamp_list

# standard functions
def cleaning_raw_df(df, nan_threshold=0.3):
    # CLEAN UP raw data frame, nan if likelihood < x (nan_threshold), ints&floats insead of strings
    # combines the string of the two rows to create new header

    new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     # df.iloc[0] takes all values of first row
    df.columns = new_columns
    df = df.drop(labels=[0, 1, 2], axis="index")

    # adapt index
    df.set_index('bodyparts_coords', inplace=True)
    df.index.names = ['frames']

    # turn all the prior string values into floats and index into int
    df = df.astype(float)
    df.index = df.index.astype(int)
    
    # flip the video along the x axis
    for column in df.columns:
        if '_y' in column:
            df[column] = y_pixel - df[column] 

    # whereever _likelihood <= x, change _x & _y to nan
    for bodypart in all_bodyparts:
        filter = df[f'{bodypart}_likelihood'] <= nan_threshold
        df.loc[filter, f'{bodypart}_x'] = np.nan
        df.loc[filter, f'{bodypart}_y'] = np.nan
        df = df.drop(labels=f'{bodypart}_likelihood', axis="columns")
    
    # transform every coordinate into mm
    mm_per_px = arena_length / np.mean([x_pixel, y_pixel])
    df = df * mm_per_px
    
    return df
def animal_moving(df, min_speed=50, min_moving_duration_s=0.5):
    # define segments where the animal is moving acc to the parameters min_speed (in mm/s) and miin_moving_duration_s (in sec)

    # for defining movement, we shift a very stable column (e.g. anus) for certain frames into the future and calculate movement acc to the resulting euclidian difference
    duration_shifted_s = 0.1
    frames_shifted = int(duration_shifted_s * fps)
    df['anus_x_shifted'] = df['anus_x'].shift(periods=frames_shifted)
    df['anus_y_shifted'] = df['anus_y'].shift(periods=frames_shifted)
    df['dist_anus_shifted'] = euclidian_dist(df['anus_x'].values, df['anus_y'].values, df['anus_x_shifted'].values, df['anus_y_shifted'].values)


    # min_speed is in mm/s, min_moving_duration is in s  !!!depends on duration_shifted_s, which should be 0.1s!!!
    nec_dist_mm = duration_shifted_s * min_speed
    animalmoving = df['dist_anus_shifted'] >= nec_dist_mm
    df['animalmoving'] = animalmoving

    # get True on- and offset
    moving_segments = timestamps_onsets(animalmoving, min_moving_duration_s, onset_cond=True)

    return df, moving_segments
def paw_speed(df):
    # calculate the speed of the paws' movement by shifting the column frames into the future and calculating the resulting euclidian distance

    # frames to shift paws into the future
    frames_shifted_paw_speed = 1

    for paw in ['paw_HL', 'paw_HR', 'paw_VL', 'paw_VR']:
        df[f'{paw}_x_shifted'] = df[f'{paw}_x'].shift(periods=frames_shifted_paw_speed)
        df[f'{paw}_y_shifted'] = df[f'{paw}_y'].shift(periods=frames_shifted_paw_speed)
        df[f'dist_{paw}_shifted'] = euclidian_dist(df[f'{paw}_x'].values, df[f'{paw}_y'].values, df[f'{paw}_x_shifted'].values, df[f'{paw}_y_shifted'].values)

        # distance (mm) per second
        df[f'dist_{paw}_shifted'] = df[f'dist_{paw}_shifted'] * fps/frames_shifted_paw_speed
        df[f'{paw}_speed'] = df[f'dist_{paw}_shifted'].rolling(window=4).mean()

    return df


# stride functions
def paw_speed_minmax(df, nec_speed_on=80, nec_speed_off=80):
    # goes through all paws and saves the toe-off and foot-strike as one step
    # toe-off is considered as exceeding the nec_speed_on (in mm/s), foot-strike is considered as being slower than the nec_speed_off (in mm/s)
    steps_dict = {}

    for paw in paws:
        in_swing = False
        step_start = None
        steps_this_paw = []

        for idx, val in df[f'{paw}_speed'].items():
            if not in_swing and val >= nec_speed_on:
                step_start = idx
                in_swing = True
            
            elif in_swing and val <= nec_speed_off:
                    
                    # include a "step must be longer than x frames"
                    if (idx - step_start) <= 10:
                        in_swing = False
                    else:
                        step_end = idx
                        steps_this_paw.append((step_start, step_end))
                        in_swing = False

        steps_dict[paw] = steps_this_paw
                
    return steps_dict

def get_strides(df, steps_dict):
    '''
    A stride is defined by the paw_HL only. 

    A stride starts with one frame after the paw_HL foot-strike, where the paw_HL reduces its speed under the prior defined limit.
    During a stride happens the paw_HL toe-off, where the paw_HL exceeds its speed over a prior defined limit.
    A stride ends with the frame of the paw_HL foot-strike.
    '''

    # everything is defined by paw_HL
    steps_HL = steps_dict.get('paw_HL', [])
    strides = []
    stride_began = False

    # START: 1 frame after foot-strike, END: frame of foot-strike
    for start, end in steps_HL:
        if not stride_began:
            stride_start = end + 1
            stride_began = True
        elif stride_began:
            stride_end = end
            strides.append((stride_start, stride_end))
            stride_start = end + 1
    


    # !!! THE FOLLOWING SEGMENTS ARE LIMITATIONS, THAT TRY TO DELETE STRIDES THAT ARE FOR DIFFERENT REASONS NOT PERFECT
    # !!! WATCH OUT THAT YOU HAVE ENOUGH STRIDES IN THE END AND MAYBE RISK TO TAKE NOT PERFECT STRIDES 
    # !!! USE/UNUSE THESE LIMITATIONS BY CHANGING THE 'strides = blablabla'

    # discard strides that are not inside the animal_moving_segments
    strides_in_move = []
    for start, end in strides:
        if df.loc[start:end, 'animalmoving'].all() == True:
            strides_in_move.append((start, end))
    strides = strides_in_move

    # discard strides where there is no toe-off and foot-strike of paw_HR
    steps_HR = steps_dict.get('paw_HR', [])
    strides_in_move_and_HR = []
    for start_HL, end_HL in strides_in_move:
        for start_HR, end_HR in steps_HR:
            if start_HL < start_HR and end_HL > end_HR:
                strides_in_move_and_HR.append((start_HL, end_HL))
    strides = strides_in_move_and_HR
    
    # discard strides when they are the first or last of a track (also optional, only use when you have enough data)
    good_strides = []
    for mov_start, mov_end in moving_segments:
        # collect strides within current moving segment
        segment_strides = [(stride_start, stride_end) 
                            for stride_start, stride_end in strides
                            if mov_start < stride_start and mov_end > stride_end]
        # delete first and last stride of the movement segment
        if len(segment_strides) >= 3:
            segment_strides = segment_strides[1:-1]
            good_strides.extend(segment_strides)
    strides = good_strides


    # discard strides if nose/mouth/middle/anus/tail_start coordinates during stride are nan
    relevant_bps = ['snout']#, 'middle', 'anus', 'tail_start', 'paw_HR', 'paw_HL', 'paw_VL', 'paw_VR', 'tail_middle', 'tail_end']
    very_good_strides = []
    for stride_start, stride_end in strides:
        all_clear = True
        for bp in relevant_bps:
            if df.loc[stride_start:stride_end, f'{bp}_x'].isna().any():
                all_clear = False
        if all_clear:
            very_good_strides.append((stride_start, stride_end))
    strides = very_good_strides

    print(f'number of accepted strides: {len(strides)}')
    return strides

def stride_normalization(df, num_bin=20):
    # you'll have lists with different lengths, therefore trim everything into bins that have the same number of datapoints (num_bin)
    # num_bin is the number of bins, should be less than the available data points
    # normalizes all strides into the same amount of data points

    norm_strides_dict = {}
    for paw in paws:
        norm_stride = []
        for start, end in stride_segments:
            speeds = df.loc[start:end+1, f'{paw}_speed']

            n = len(speeds)
            bin_edges = np.linspace(0, n, num_bin+1, dtype=int)

            bin_means = []
            for i in range(num_bin):
                bin_start = bin_edges[i]
                bin_end = bin_edges[i+1]
                bin_data = speeds[bin_start:bin_end]
                bin_mean = np.mean(bin_data) if len(bin_data) > 0 else np.nan
                bin_means.append(bin_mean)
            
            norm_stride.append(bin_means)
        norm_strides_dict[paw] = norm_stride
        

    # create a list of the means of paw speeds
    mean_strides_dict = {}
    for paw, strides in norm_strides_dict.items():
        
        mean_stride = []
        for i in range(num_bin):
            values = [stride[i] for stride in strides if not np.isnan(stride[i])]
            mean_value = np.mean(values) if values else np.nan
            mean_stride.append(mean_value)

        mean_strides_dict[paw] = mean_stride


    return norm_strides_dict, mean_strides_dict

def calc_stride_vectors(df, stride_segments):
    # get the middle point at the start of the stride and the end of the stride and calc distance and vector
    stride_distances = []
    stride_vectors = []

    for start, end in stride_segments:
        x1 = np.mean(df.loc[start-1 : start+1, 'middle_x'])
        y1 = np.mean(df.loc[start-1 : start+1, 'middle_y'])
        x2 = np.mean(df.loc[end-1 : end+1, 'middle_x'])
        y2 = np.mean(df.loc[end-1 : end+1, 'middle_y'])

        stride_distances.append(euclidian_dist(x1, y1, x2, y2))

        slope, intercept, r_value, p_value, std_err = linregress([x1, x2], [y1, y2])
        stride_vectors.append((slope, intercept))

    return stride_distances, stride_vectors

def lateral_displacement(df, stride_segments, stride_vectors, bodypart, num_bins=20):
    # for every frame in the stride, get the bp coordinates and claculate its distance from the intrinsic stride vector

    # get distance from bp to intrinsic stride_vector
    lateral_displacements = []
    for i, stride in enumerate(stride_segments):
        slope, intercept = stride_vectors[i]
        lateral = []
        for frame in range(stride[0], stride[1]+1):
            lateral.append(point2line_dist(slope, intercept, df.loc[frame, f'{bodypart}_x'], df.loc[frame, f'{bodypart}_y']))
        lateral_displacements.append(lateral)
    
    
    # you'll have lists with different lengths, therefore trim everything into num_bins
    lateral_displacements_norm = []
    for stride in lateral_displacements:
        n = len(stride)
        bin_edges = np.linspace(0, n, num_bins+1, dtype=int)

        bin_means = []
        for i in range(num_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            bin_data = stride[bin_start:bin_end]
            bin_mean = np.mean(bin_data) if len(bin_data) > 0 else np.nan
            bin_means.append(bin_mean)
        lateral_displacements_norm.append(bin_means)


    # create a list of the means of lateral bp displacements
    mean_lateral_displacements = []
    for i in range(num_bins):
        values = [stride[i] for stride in lateral_displacements_norm if not np.isnan(stride[i])]
        mean_lateral_displacements.append(np.mean(values))


    return lateral_displacements_norm, mean_lateral_displacements




# plotting
def plot_bodyparts(timewindow, df, spec_bodyparts, bodyparts, title):

    paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
    
    for start, end in timewindow:
        i_color = randrange(len(colors))

        # only include rows that are in the time window
        rows2plot = df.loc[start : end]

        if 'center' in spec_bodyparts:
            # plot animal center via removing the nan values
            x = rows2plot['center_x'][~np.isnan(rows2plot['center_x'])]
            y = rows2plot['center_y'][~np.isnan(rows2plot['center_y'])]
            #plt.scatter(x, y, label=f'{start}_center_{end}', s=0.2, c=colors[i_color])
            plt.plot(x, y, label=f'{start}-{end}', c=colors[i_color])
        
        if 'static_feet' in spec_bodyparts:            
            for paw in paws:
                plt.scatter(rows2plot[f'static_{paw}_x'], rows2plot[f'static_{paw}_y'], label=f'static_{paw}', c=colors[i_color], marker='x', s=20)
                
                if 'raw_feet' in spec_bodyparts:
                        plt.scatter(rows2plot[f'{paw}_x'], rows2plot[f'{paw}_y'], label=paw, c=colors[i_color], marker='.', s=10)

        
        for part in bodyparts:
            plt.scatter(rows2plot[f'{part}_x'], rows2plot[f'{part}_y'], label=part, c=colors[i_color], marker='+', s=10)
            

        i_color += 1
    
    plt.xlim(0, arena_length)
    plt.ylim(0, arena_length)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    #plt.legend()
    plt.title(title)
    plt.show()
def plot_ind_pawspeeds(df, steps_dict, paws, lim):
        
    # plots paw speeds and their steps as horizontal lines
    for i, paw in enumerate(paws):

        x = range(lim[0], lim[1]+1)
        y = df.loc[lim[0]:lim[1], f'{paw}_speed']

        plt.plot(x, y, c=colors[i], label=paw)
        plt.fill_between(x, y, color=colors[i], alpha=0.2)

        lines = steps_dict.get(paw, [])
        for start, end in lines:
            plt.hlines(-50 + i*15, start, end, colors[i], linewidth=2)

    plt.xlim(lim[0], lim[1])
    plt.xlabel('Time (frames)')
    plt.ylabel('Paw Speed (mm/s)')
    plt.legend()
    plt.title('Steps')
    plt.show()
def plot_average_stride(output_name, norm_strides_dict, mean_strides_dict, paws):
    
    # plot the desired paws for the average stride
    for i, paw in enumerate(paws):
        x = np.linspace(0, 100, len(mean_strides_dict[paw]))

        # calculate standard deviation for the error region of the mean plot
        stride_matrix = np.array(norm_strides_dict[paw])
        std_dev = np.nanstd(stride_matrix, axis=0)
        sem_dev = (np.nanstd(stride_matrix, axis=0) / np.sqrt(len(stride_matrix)))
    
        mean = np.array(mean_strides_dict[paw])

        plt.plot(x, mean, c=colors[i], linewidth=2, label=f'{paw} n={len(norm_strides_dict[paw])}')
        plt.fill_between(x, mean-sem_dev, mean+sem_dev, color=colors[i], alpha=0.3)

        # individual traces
        if False == True:
            for stride in norm_strides_dict[paw]:
                plt.plot(x, stride, c=colors[i], alpha=0.05)
        
    plt.xlabel('Percent Stride')
    plt.ylabel('Paw Speed [mm/s]')
    #plt.ylim(0,800)
    plt.title(f'Stride Architecture')
    plt.legend()
    plt.savefig(f'{output_name}_Stride.png')
    plt.show()
def plot_displacement(output_name, lateral_displacements_norm, mean_lateral_displacements, bodypart):

    
    x = np.linspace(0, 100, len(mean_lateral_displacements))

    # calculate standard deviation for the error region of the mean plot
    stride_matrix = np.array(lateral_displacements_norm)
    std_dev = np.nanstd(stride_matrix, axis=0)
    sem_dev = (np.nanstd(stride_matrix, axis=0) / np.sqrt(len(stride_matrix)))

    mean = np.array(mean_lateral_displacements)

    plt.plot(x, mean, c='k', linewidth=2, label=f'{bodypart} n={len(lateral_displacements_norm)}')
    plt.fill_between(x, mean-sem_dev, mean+sem_dev, color='k', alpha=0.3)

    # individual traces
    if False == True:
        for move in lateral_displacements_norm:
            plt.plot(x, move, c='k', alpha=0.505)



    plt.xlim(0, max(x))
    plt.xlabel('Percent Stride')
    plt.ylabel('Displacement [mm]')
    plt.title(f'Lateral {bodypart} Displacement {output_name}')
    #plt.legend()
    plt.savefig(f'{output_name}_Lateral {bodypart} Displacement.png')
    plt.show()




for file in file_list:
    print(f'{file}')
    output_name = file.replace(path, '')
    output_name = output_name.replace('.csv', '')
    
    # standard stuff
    main_df = pd.read_csv(file, header=None, low_memory=False)
    main_df = cleaning_raw_df(main_df)
    main_df, moving_segments = animal_moving(main_df)
    main_df = paw_speed(main_df)


    # stride stuff
    steps_dict = paw_speed_minmax(main_df)
    stride_segments = get_strides(main_df, steps_dict)
    norm_strides_dict, mean_strides_dict = stride_normalization(main_df)
    stride_distances, stride_vectors = calc_stride_vectors(main_df, stride_segments)

    print(stride_segments)




    # lateral stuff
    for bp in ['snout']:#, 'tail_start', 'tail_middle', 'tail_end', 'anus']:
        lateral_displacements_norm, mean_lateral_displacements = lateral_displacement(main_df, stride_segments, stride_vectors, bp)
        plot_displacement(output_name, lateral_displacements_norm, mean_lateral_displacements, bp)





    # plot_ind_pawspeeds(main_df, steps_dict, ['paw_HL', 'paw_HR'], [stride_segments[0][0], stride_segments[0][1]])
    # plot_average_stride(output_name, norm_strides_dict, mean_strides_dict, ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR'])



#%%


anim_bodyparts = ['snout', 'paw_VL','paw_VR', 'paw_HR', 'paw_HL']

def animation(df, bodypart_list):

    colors = ['gray', 'b', 'c', 'orange', 'r']
    fig, ax = plt.subplots()
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    

    def update(frame):
        ax.clear()
        ax.set_xlim(240, 330)
        ax.set_ylim(100, 250)

        for i, bp in enumerate(bodypart_list):
            x = df[f'{bp}_x'].iloc[frame]
            y = df[f'{bp}_y'].iloc[frame]
            ax.scatter(x, y, color=colors[i], s=70)
            ax.text(x+2, y, bp, fontsize=8, color=colors[i])
        return ax

    anim = FuncAnimation(fig, update, frames=np.arange(7936, 7961, 1), interval=100)
    anim.save('continuousSineWave.mp4',  
          writer = 'ffmpeg', fps = 10) 
    return anim

my_animation = animation(main_df, anim_bodyparts)

