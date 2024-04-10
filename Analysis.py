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

# you have a series with conditions (e.g. True/False or float/nan) in each row
# return the start & stop frame of the [cond_onset, cond_offset]
def timestamps_onsets(series, min_frame_nb, onset_cond=float):
    timestamp_list = []
    start_idx = None
    prev_idx = None
    
    # get on- and offset frames between switches between True & False
    if onset_cond == True:
        for idx, val in series.items():
            if (val == onset_cond) and (start_idx == None):
                start_idx = idx
            
            elif (val != onset_cond) and (start_idx != None):
                if idx - start_idx >= min_frame_nb:
                    timestamp_list.append([start_idx, prev_idx])
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
        static_paw_timestamps = timestamps_onsets(paw_is_static, min_static_frame_nb, onset_cond=True)
    
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
    catwalk_timestamps = timestamps_onsets(catwalk_filter, min_catwalk_length, onset_cond=True)

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
    plt.legend()
    plt.title(title)
    plt.show()



# POINT TO MIDLINE DISTACE
# take a list of indices (>= 1 index values) and add indices where there is a float in higher&lower row in certain column
def add_float_of_higherlower_row(df, index_list, column):
    column_number = df.columns.get_loc(column)
    
    # check the next float value towards lower rows
    lower_row = None
    low_row = df.index.get_loc(index_list[0])
    row_number = low_row - 1
    while row_number >= 0:
        value = df.iloc[row_number, column_number]
        if not np.isnan(value):
            lower_row = row_number
            break
        row_number -= 1

    # check the next float value towards higher rows
    higher_row = None
    high_row = df.index.get_loc(index_list[-1])
    row_number = high_row + 1
    while row_number < len(df):
        value = df.iloc[row_number, column_number]
        if not np.isnan(value):
            higher_row = row_number
            break
        row_number += 1

    if lower_row != None:
        index_list.append(df.index[lower_row])
    if higher_row != None:
        index_list.append(df.index[higher_row])
    
    index_list.sort()

    return index_list

# takes point coords and regression line slope& intercept and returns perpendicular distance
def point2line_dist(point, slope, intercept):
    x, y = point
    distance = abs(slope*x - y + intercept) / np.sqrt(slope**2 + 1)
    return distance

# takes a frame window and returns a regression line of the animal midline during window
def internal_midline_in_window(df, frame_window, midline_column):
    midcol_x, midcol_y = f'{midline_column}_x', f'{midline_column}_y'
    start, end = frame_window

    # get all 'pos' float values (not nan) in the frame_window and add index to list
    pos_in_static = df.loc[start : end, midcol_x].copy().dropna()
    pos_in_static_idx = pos_in_static.index.tolist()

    # if there are no pos_coords in the frames_window of static_feet, add the mean static feet frame
    if len(pos_in_static_idx) == 0:
        pos_in_static_idx.append(int(np.mean(frame_window)))

    # if only 2 or less pos_coords, add one above and below until we have >= 3
    while len(pos_in_static_idx) < 3:
        pos_in_static_idx = add_float_of_higherlower_row(df, pos_in_static_idx, midcol_x)

    # get xy coords of relevant midline frames
    pos_coords = df.loc[pos_in_static_idx, [midcol_x, midcol_y]].to_numpy()
    
    pos_coords = np.array([sublist for sublist in pos_coords if not any(np.isnan(x) for x in sublist)])

    print(pos_coords)
    print('\n')

    # fit linear regression to coords
    slope, intercept = np.polyfit(pos_coords[:, 0], pos_coords[:, 1], 1)

    return pos_coords, slope, intercept

# calculate distance from a point to straight midline
# static point (e.g. static paw): window is start&end frame of static frames, point=window[0]
# moving point (e.g. moving nose): 'window' is only 1 frame so window must be [frame, frame])
def point2midline_dist(df, window, midline_column, point_column):
    
    # linear regression line
    pos_coords, mid_slope, mid_intercept = internal_midline_in_window(df, window, midline_column)
    point_coords = df.loc[window[0], [f'{point_column}_x', f'{point_column}_y']].to_numpy()

    # point-midline distance
    distance = point2line_dist(point_coords, mid_slope, mid_intercept)

    #plot_point2midline(pos_coords, mid_slope, mid_intercept, point_coords, distance)
    return distance


def plot_point2midline(pos_coords, mid_slope, mid_intercept, point_coords, distance):
    
    pos_coords_x = [pos[0] for pos in pos_coords]
    pos_coords_y = [pos[1] for pos in pos_coords]
    plt.scatter(pos_coords_x, pos_coords_y, label='center')

    regline_x = np.arange(0, x_pixel, 1)
    regline_y = (mid_slope * regline_x) + mid_intercept
    plt.plot(regline_x, regline_y, label='midline')

    point_x, point_y = point_coords
    plt.scatter(point_x, point_y, label='point')

    plt.xlim(30, 470)
    plt.ylim(45, 155)
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

# now, lets focus on one static_paw first: 
# Whenever there is a static_paw frame_window, take the coords of the animal_center that are in the static_paw frames
# X If there are less than 3 frames with animal_center coordinates in the static_paw frame_window, take the 3 animal_center coords closest to frame_window
# X create a best-fitting linear regression onto the >=3 animal_center coords
# X calculate the closest distance/perpendicular of static_paw to regression line and save in list


all_dists = []
paws = ['static_paw_VR', 'static_paw_VL', 'static_paw_HR', 'static_paw_HL']
for paw in paws:
    print('DDD')
    dists = []
    for walk_start, walk_end in catwalk_timestamps:
        
        # get a frame_window of all static_paws
        staticpaw_in_window = main_df.loc[walk_start : walk_end, f'{paw}_x']
        static_window = timestamps_onsets(staticpaw_in_window, 1, onset_cond=float)

        for static_paw in static_window:
            distance = point2midline_dist(main_df, static_paw, 'pos', paw)
            dists.append(distance)
    all_dists.append(dists)


print(all_dists)

front_paws = all_dists[0] + all_dists[1]
hind_paws = all_dists[2] + all_dists[3]
x_data = ['front', 'hind']
y_data = [np.mean(front_paws), np.mean(hind_paws)]

front_paws_std = np.std(front_paws)
hind_paws_std = np.std(hind_paws)
error = [front_paws_std, hind_paws_std]

plt.bar(x_data, y_data, yerr=error)



plt.show()

# %%


