#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# Lets keep df as the main DataFrame, where the index is the frame number. 
# If we calculate sth new (e.g. animal center, static_paw), there is the function fuse_dfs() to add new columns to the df. 
# Whenever we work on/manipulate the df, lets do this with a df.copy(), so the df is untouched





# IMPORT & DEFINE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
     
df = pd.read_csv('data\\2m_292_crop.csv', header=None)

x_pixel, y_pixel = 570, 570
fps = 100
all_bodyparts = ['snout', 'mouth', 'paw_VL','paw_VR', 'middle', 'paw_HL', 'paw_HR','anus', 'tail_start', 'tail_middle', 'tail_end']
colors = ['b', 'g', 'r', 'c', 'm', 'y']



#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# FUNCTIONS

# FUSE 2 dfs
def fuse_dfs(df1, df2):
    
    # reindex df2 to the index of df1
    df2 = df2.reindex(df1.index)

    # add df2 to df1
    df1 = df1.join(df2)

    return df1

# you have a series with conditions (e.g. True/False or float/nan) in each row (depending on a previously established filter condition), return the start& stop frame of the [True onset, True offset]
def timestamps_onsets(series, min_frame_nb, onset_condition):
    timestamp_list = []
    start_idx = None
    prev_idx = None

    for idx, val in series.items():
        if (val==onset_condition) and (start_idx is None):
            start_idx = idx
            print(val)
        
        elif (not val==onset_condition) and (start_idx is not None):
            if idx - start_idx >= min_frame_nb:
                timestamp_list.append([start_idx, prev_idx])
            start_idx = None
        prev_idx = idx

    # for last value
    if start_idx is not None:
        if series.index[-1] - start_idx >= min_frame_nb:
            timestamp_list.append([start_idx, series.index[-1]])
    
    return timestamp_list



# CLEAN UP raw data frame, nan if likelihood < x, ints&floats insead of strings
def cleaning_raw_df(df):
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
        filter = df[f'{bodypart}_likelihood'] <= 0.9
        df.loc[filter, f'{bodypart}_x'] = np.nan
        df.loc[filter, f'{bodypart}_y'] = np.nan
        df = df.drop(labels=f'{bodypart}_likelihood', axis="columns")
    
    return df

# get the center of the animal via mean(stable bodyparts) and create new dataframe, also skip frames
def animal_center(df, frames2skip):

    # only includes every k-th row to reduce noise
    df_crop = df.iloc[::frames2skip]

    # choose bodyparts that are always tracked
    stable_bodyparts = ['middle', 'paw_HL', 'paw_HR', 'anus', 'tail_start']
    stable_bodyparts_x = [f'{bodypart}_x' for bodypart in stable_bodyparts]
    stable_bodyparts_y = [f'{bodypart}_y' for bodypart in stable_bodyparts]

    # per row, take the mean of the stable bodyparts to have a stable animal position time series
    row_means_x = df_crop[stable_bodyparts_x].mean(axis="columns")
    row_means_y = df_crop[stable_bodyparts_y].mean(axis="columns")

    # create a new data frame with animal position
    df_center = pd.DataFrame(data={'pos_x': row_means_x, 'pos_y':row_means_y})

    # add a gaussian filter over the time series to reduce outlayers, don't know how necessary this is
    df_center = df_center.rolling(window=10, win_type='gaussian', min_periods=1, center=True).mean(std=1)
    
    # add df_center to df
    df = fuse_dfs(df, df_center)

    return df

# get list of start- and stop-frames, where paws are static and df with only the mean paw position, rest is nan
def static_feet(df, min_paw_movement, min_static_frame_nb):

    # define paws
    paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
    paws_coor = [f'{paw}_x' for paw in paws] + [f'{paw}_y' for paw in paws]

    # calculate the difference between the k frame and the k-x frame
    df_feet = df[paws_coor].copy()
    paws_diff = abs(df_feet.diff(periods=1, axis='rows'))   # periods=x


    for paw in paws:
        x, y = f'{paw}_x', f'{paw}_y'
        
        # check in which frames the vector magnitude is below x
        vector_lengths = np.sqrt(paws_diff[x]**2 + paws_diff[y]**2)
        paw_is_static = vector_lengths <= min_paw_movement
        static_paw_timestamps = timestamps_onsets(paw_is_static, min_static_frame_nb, True)
    
        # creating a mask to change all values outside of timestamp-rows to nan, so that only static paw values exist
        mask = np.ones(len(df_feet), dtype=bool)
        for start, end in static_paw_timestamps:
            mask[start:end] = False
        df_feet.loc[mask, [x, y]] = np.nan

        # go through timestamps and change all values in this timestamp to the mean values of the timestamp
        for start, end in static_paw_timestamps:
            for coordinate in [x, y]:
                step_mean = df_feet.loc[start:end, coordinate].mean()
                df_feet.loc[start:end, coordinate] = step_mean
    
    # add prefix to column names and add df_feet to df
    df_feet = df_feet.add_prefix('static_')
    df = fuse_dfs(df, df_feet.copy())

    return df




# get list of start- and stop-frames and new dataframe, where the animal is moving straight according to df_center
def catwalk(df, min_vector_lengh, max_angle_diff, min_catwalk_length):

    # creates df_center from df
    x = df['pos_x'][~np.isnan(df['pos_x'])]
    y = df['pos_y'][~np.isnan(df['pos_y'])]
    df_center = pd.DataFrame(data=[x, y]).transpose()

    # calculate the difference between the k frame and the k-x frame
    changes = df_center.diff(periods=1, axis='rows')

    # check at which frames the vector magnitude is above x
    vector_lengths = np.sqrt(changes['pos_x']**2 + changes['pos_y']**2)
    does_move = vector_lengths >= min_vector_lengh

    # checks at which frames, the angle difference is below x
    angles = np.arctan2(changes['pos_x'], changes['pos_y'])     # angle between the vector (0,0) to (1,0) and the vector (0,0) to (x2,y2)
    angle_diff = abs(angles.diff(periods=1))
    does_straight = angle_diff <= max_angle_diff

    # combine both True/False series
    catwalk_filter = does_move & does_straight
    
    # get True on- and offset
    catwalk_timestamps = timestamps_onsets(catwalk_filter, min_catwalk_length, True)

    return catwalk_timestamps



# (list of [start, stop],   df,     special_bodyparts=['pos', 'feet', 'raw_feet'],      bodypart=[e.g. 'snout', 'mouth'],   title of plot)
def plot_bodyparts(timewindow, df, spec_bodyparts, bodyparts, title):

    for start, end in timewindow:

        # only include rows that are in the time window
        rows2plot = df.loc[start : end]

        i_color = 0
        if 'pos' in spec_bodyparts:
            # plot animal center via removing the nan values
            x = rows2plot['pos_x'][~np.isnan(rows2plot['pos_x'])]
            y = rows2plot['pos_y'][~np.isnan(rows2plot['pos_y'])]
            plt.scatter(x, y, label='center', c='grey')
        
        if 'feet' in spec_bodyparts:
            # plot static feet
            paws = ['paw_VL', 'paw_VR', 'paw_HL', 'paw_HR']
            for paw in paws:
                plt.scatter(rows2plot[f'static_{paw}_x'], rows2plot[f'static_{paw}_y'], label=f'static_{paw}', c=colors[i_color], marker='x', s=60)
                
                # plot raw feet
                if 'raw_feet' in spec_bodyparts:
                    plt.scatter(rows2plot[f'{paw}_x'], rows2plot[f'{paw}_y'], label=paw, c=colors[i_color], marker='.', s=10)
                
                i_color += 1
        
        for part in bodyparts:
            plt.scatter(rows2plot[f'{part}_x'], rows2plot[f'{part}_y'], label=part, c=colors[i_color], marker='+', s=20)
            i_color += 1


    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    #plt.legend()
    plt.title(title)
    plt.show()






#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# DO STUFF

# CLEAN the df into our format, add CENTER OF ANIMAL to df, add STATIC PAWS to df
main_df = cleaning_raw_df(df)
main_df = animal_center(main_df, 10)
main_df = static_feet(main_df, 5, 5)


# get timestamps of CATWALK
catwalk_timestamps = catwalk(main_df, 10, 0.15, 100)



# PLOT
# (list of [start, stop],   df,     special_bodyparts=['pos', 'feet', 'raw_feet'],      bodypart=[e.g. 'snout', 'mouth'],   title of plot)
plot_bodyparts([catwalk_timestamps[3]], main_df, ['pos', 'feet'], [], 'straight, fast and long Catwalks (animal center)')


#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# StaticFeet_2_Midline_distance


# first we check the distances on a straight line, e.g.
timewindow = catwalk_timestamps[3]

# now, lets focus on one static_paw first: 
# Whenever there is a static_paw frame_window, take the coords of the animal_center that are in the static_paw frames
# If there are less than 3 frames with animal_center coordinates in the static_paw frame_window, take the 3 animal_center coords closest to frame_window
# create a best-fitting linear regression onto the >=3 animal_center coords
# calculate the closest distance/perpendicular of static_paw to regression line and save in list

onsets = timestamps_onsets(main_df['static_paw_VL_x'], 1, ~np.isnan())
print(onsets)

a = 3

if not np.isnan(a):
    print('see')


# %%
