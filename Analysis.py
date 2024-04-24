#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# Lets keep main_df as the main DataFrame, where the index is the frame number. 
# If we calculate sth new (e.g. animal center, static_paw), there is the function fuse_dfs() to add new columns to the df. 
# Whenever we work on/manipulate the df, lets do this with a df.copy(), so the df is untouched





# IMPORT & DEFINE

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
     
df = pd.read_csv('data\\2m_292_crop.csv', header=None)

x_pixel, y_pixel = 570, 570
arena_length = 400 #mm
px_per_mm = np.mean([x_pixel, y_pixel]) / arena_length
fps = 100
all_bodyparts = ['snout', 'mouth', 'paw_VL','paw_VR', 'middle', 'paw_HL', 'paw_HR','anus', 'tail_start', 'tail_middle', 'tail_end']
colors = ['b', 'g', 'r', 'c', 'm', 'y']



#%% HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# FUNCTIONS


def fuse_dfs(df1, df2):
    # FUSE 2 dfs
    # reindex df2 to the index of df1 (important if you have nan gaps)
    df2 = df2.reindex(df1.index)
    df1 = df1.join(df2)

    return df1


def timestamps_onsets(series, min_frame_nb, onset_cond=float):
    # you have a series with conditions (e.g. True/False or float/nan) in each row
    # return the start & stop frame of the [cond_onset, cond_offset]
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
    
    # add a gaussian filter over the df to reduce outlayers, don't know how necessary this is
    df = df.rolling(window=10, win_type='gaussian', min_periods=1, center=True, axis=1).mean(std=1)

    return df



def point2centerline_dist(df, point, centerline_bps):

    # get all 'pos' float values (not nan) in the frame_window ad add index to list
    pos_in_static = df.loc[static_window[0] : static_window[1], f'{midline}_x'].copy().dropna()
    pos_in_static_idx = pos_in_static.index.tolist()


    # if there are no pos_coords in the frames_window of static_feet, add a middle frame
    if len(pos_in_static_idx) == 0:
        pos_in_static_idx.append(int(np.mean(static_window)))

    # if only 2 or less pos_coords, add one above and below until we have >= 3
    while len(pos_in_static_idx) < 3:
        pos_in_static_idx = add_higherlower_float_row(df, pos_in_static_idx, f'{midline}_x')

    print(f'relevant_pos_frames: {pos_in_static_idx}')


# get the center of the animal via mean(stable bodyparts) and create new dataframe, also skip frames
def add_columns_to_df(df, px_per_mm):

    new_df = pd.DataFrame()

    # create time series for center that is represented by the mean of stable bodyparts
    # choose bodyparts that are always tracked
    stable_bodyparts = ['middle', 'anus', 'tail_start']
    stable_bodyparts_x = [f'{bodypart}_x' for bodypart in stable_bodyparts]
    stable_bodyparts_y = [f'{bodypart}_y' for bodypart in stable_bodyparts]

    # per row, take the mean of the stable bodyparts to have a stable animal position time series
    new_df['pos_x'] = df[stable_bodyparts_x].mean(axis="columns")
    new_df['pos_y'] = df[stable_bodyparts_y].mean(axis="columns")

    # add distances beetween bps    
    new_df['dist_H_paws'] = euclidian_dist(df['paw_HL_x'].values, df['paw_HL_y'].values, df['paw_HR_x'].values, df['paw_HR_y'].values, px_per_mm)
    new_df['dist_V_paws'] = euclidian_dist(df['paw_VL_x'].values, df['paw_VL_y'].values, df['paw_VR_x'].values, df['paw_VR_y'].values, px_per_mm)
    new_df['dist_L_paws'] = euclidian_dist(df['paw_HL_x'].values, df['paw_HL_y'].values, df['paw_VL_x'].values, df['paw_VL_y'].values, px_per_mm)
    new_df['dist_R_paws'] = euclidian_dist(df['paw_HR_x'].values, df['paw_HR_y'].values, df['paw_VR_x'].values, df['paw_VR_y'].values, px_per_mm)
    
    # takes bps that would represent an intrinsic centerline, calculates linear regression line for each row and adds columns for slope and intercept
    # takes nan if less than 2 bps without nan
    def linear_fit(row, x_headers, y_headers):
        # fits a linear regression line to the column values provided and returns the coefficients
        x = row.loc[x_headers]
        y = row.loc[y_headers]

        if np.sum(~np.isnan(x)) < 2:
            return (np.nan, np.nan)
        
        coeffs = np.polyfit(x, y, 1)
        return coeffs
    
    centerline_x_headers = ['snout_x', 'mouth_x', 'middle_x', 'anus_x', 'tail_start_x']
    centerline_y_headers = ['snout_y', 'mouth_y', 'middle_y', 'anus_y', 'tail_start_y']
    new_df[['centerline_slopes', 'centerline_intercepts']] = df.apply(linear_fit, args=(centerline_x_headers, centerline_y_headers), axis=1, result_type='expand')
    
    
    
    
    new_df['dist_HL2center'] = euclidian_dist(df['paw_HL_x'].values, df['paw_HL_y'].values, new_df[#NO; USE CENTERLINE].values, new_df['pos_y'].values, px_per_mm)
    new_df['dist_HR2center'] = euclidian_dist(df['paw_HR_x'].values, df['paw_HR_y'].values, new_df[#NO; USE CENTERLINE].values, new_df['pos_y'].values, px_per_mm)
    new_df['dist_VL2center'] = euclidian_dist(df['paw_VL_x'].values, df['paw_VL_y'].values, new_df[#NO; USE CENTERLINE].values, new_df['pos_y'].values, px_per_mm)
    new_df['dist_VR2center'] = euclidian_dist(df['paw_VR_x'].values, df['paw_VR_y'].values, new_df[#NO; USE CENTERLINE].values, new_df['pos_y'].values, px_per_mm)



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

# now, lets focus on one static_paw first: 
    # X Whenever there is a static_paw frame_window, take the coords of the animal_center that are in the static_paw frames
    # X If there are less than 3 frames with animal_center coordinates in the static_paw frame_window, take the 3 animal_center coords closest to frame_window
    # O create a best-fitting linear regression onto the >=3 animal_center coords
    # O calculate the closest distance/perpendicular of static_paw to regression line and save in list


# returns the index of the next float number while skipping NaNs (in either direction)
def find_next_float_number(df, column, start_row, dir_higher=True):
    column_number = df.columns.get_loc(column)
    
    # check the next float value towards higher rows
    if dir_higher == True:
        row_number = start_row + 1
        while row_number < len(df):
            value = df.iloc[row_number, column_number]
            if not np.isnan(value):
                return row_number
            row_number += 1
        return None
    
    # check the next float value towards lower rows
    else:
        row_number = start_row - 1
        while row_number >= 0:
            value = df.iloc[row_number, column_number]
            if not np.isnan(value):
                return row_number
            row_number -= 1
        return None

# get index_list with >= 1 index values and add indices with column_floats above & below
def add_higherlower_float_row(df, index_list, column):

    low_row = df.index.get_loc(index_list[0])
    lower_row = find_next_float_number(df, column, low_row, dir_higher=False)

    high_row = df.index.get_loc(index_list[-1])
    higher_row = find_next_float_number(df, column, high_row, dir_higher=True)
    
    index_list.append(lower_row)
    index_list.append(higher_row)

    return index_list


# get distance from static paw to midline
def staticpaw2midline_dist(df, static_window, midline):

    # get all 'pos' float values (not nan) in the frame_window ad add index to list
    pos_in_static = df.loc[static_window[0] : static_window[1], f'{midline}_x'].copy().dropna()
    pos_in_static_idx = pos_in_static.index.tolist()


    # if there are no pos_coords in the frames_window of static_feet, add a middle frame
    if len(pos_in_static_idx) == 0:
        pos_in_static_idx.append(int(np.mean(static_window)))

    # if only 2 or less pos_coords, add one above and below until we have >= 3
    while len(pos_in_static_idx) < 3:
        pos_in_static_idx = add_higherlower_float_row(df, pos_in_static_idx, f'{midline}_x')

    print(f'relevant_pos_frames: {pos_in_static_idx}')



walk_window = catwalk_timestamps[0]
print(f'walk_window: {walk_window}')

# get a frame_window of all static_paws
staticpaw_in_move = main_df.loc[walk_window[0] : walk_window[1], 'static_paw_VL_x']
static_window = timestamps_onsets(staticpaw_in_move, 1, onset_cond=float)[0]
print(f'static_paw_window: {static_window}')

staticpaw2midline_dist(main_df, static_window, 'pos')




# %%


def euclidian_dist(bp1xVals, bp1yVals, bp2xVals, bp2yVals, px_per_mm):
    # calculates the euclidian distance between 2 bps in mm
    series = (np.sqrt((bp1xVals - bp2xVals) ** 2 + (bp1yVals - bp2yVals) ** 2)) / px_per_mm

    return series



def add_distances_to_df(df,px_per_mm):

    



    