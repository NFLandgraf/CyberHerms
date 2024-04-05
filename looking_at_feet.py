# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:50:20 2024

The idea of this script is to analyse and visualize data from a single video


To Do
-convert pixel to microns - i guess each videos needs a calibration?

-define frames when mouse is moving (power spectrum of paw distance?)

-define frames when mouse is moving in straight line (use angle of body? relative to ?)

-define frames when mouse is turning left or turning right

-define the other common parameters

-plot the average stance for a given frame range?

@author: Paul LC Feyen
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


path2csv = r'C:\Users\Paul\Documents\DATA\PV feet\2m_301_adjusted_adjustedDLC_resnet50_WalterWhitiesMar26shuffle1_230000.csv'


f_Start = 7750
f_End = 8000

# f_Start = 7750
# f_End = 12000


#%% ### =========  Let's define some functions ==========================

def plot_bodyparts(df, bodyparts, frame_range):
    """
    Plots the positions (x, y) of one or multiple bodyparts across a desired range of frames.
    
    Parameters:
    df (DataFrame): The DataFrame containing the tracking data.
    bodyparts (list of str): The list of bodyparts to plot.
    frame_range (tuple): The range of frames to plot (start, end).
    """
    # Generate a list of colors for plotting
    colors = plt.cm.jet(np.linspace(0, 1, len(bodyparts)))

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Extract the column headers for body parts and coordinates
    column_headers = df.iloc[1]
    coords_row = df.iloc[2]

    # Filter columns to get only the ones we are interested in
    for i, bodypart in enumerate(bodyparts):
        bp_columns = [col for col in df.columns if bodypart in column_headers[col]]
        x_columns = [col for col in bp_columns if 'x' in coords_row[col]]
        y_columns = [col for col in bp_columns if 'y' in coords_row[col]]

        if not x_columns or not y_columns:
            raise ValueError(f"Data for {bodypart} not found in DataFrame.")

        # Get the data for the specified range
        x_data = df.loc[frame_range[0]:frame_range[1], x_columns].mean(axis=1)
        y_data = df.loc[frame_range[0]:frame_range[1], y_columns].mean(axis=1)

        # Plot the data
        plt.scatter(x_data, y_data, label=bodypart, color=colors[i])

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Body Parts Position Across Frames')
    plt.legend()
    # ax = plt.gca()
    # ax.set_aspect('equal', 'box')
    plt.show()


def calculate_distance_between_bodyparts(df, bodypart1, bodypart2):
    """
    Calculates the Euclidean distance between two bodyparts across all frames.
    
    Parameters:
    df (DataFrame): The DataFrame containing the tracking data.
    bodypart1 (str): The first bodypart.
    bodypart2 (str): The second bodypart.
    
    Returns:
    distances (Series): A pandas Series containing the distance between the two bodyparts for each frame.
    """
    # Extract the column headers for body parts and coordinates
    column_headers = df.iloc[1]
    coords_row = df.iloc[2]

    # Find the columns corresponding to the x and y coordinates of the first bodypart
    bp1_x_cols = [col for col in df.columns if bodypart1 in column_headers[col] and 'x' in coords_row[col]]
    bp1_y_cols = [col for col in df.columns if bodypart1 in column_headers[col] and 'y' in coords_row[col]]

    # Find the columns corresponding to the x and y coordinates of the second bodypart
    bp2_x_cols = [col for col in df.columns if bodypart2 in column_headers[col] and 'x' in coords_row[col]]
    bp2_y_cols = [col for col in df.columns if bodypart2 in column_headers[col] and 'y' in coords_row[col]]

    if not bp1_x_cols or not bp1_y_cols or not bp2_x_cols or not bp2_y_cols:
        raise ValueError(f"Data for one or both bodyparts not found in DataFrame.")

    # Get the data for all frames
    bp1_x_data = df.loc[3:, bp1_x_cols].astype(float).mean(axis=1)
    bp1_y_data = df.loc[3:, bp1_y_cols].astype(float).mean(axis=1)
    bp2_x_data = df.loc[3:, bp2_x_cols].astype(float).mean(axis=1)
    bp2_y_data = df.loc[3:, bp2_y_cols].astype(float).mean(axis=1)

    # Calculate the Euclidean distance between the two bodyparts across all frames
    distances = np.sqrt((bp2_x_data - bp1_x_data) ** 2 + (bp2_y_data - bp1_y_data) ** 2)

    return distances





#%% #### =========  Let's do some stuff with the data ==========================
dataframe = pd.read_csv(path2csv, header=None)
print(dataframe)

#lets calculate the distance between the front left and back left paw
bodypart1 = 'paw_VL'  # Replace with your actual body part name
bodypart2 = 'paw_HL'  # Replace with your actual body part name
distances_VL_HL = calculate_distance_between_bodyparts(dataframe, bodypart1, bodypart2)

bodypart1 = 'paw_VR'  
bodypart2 = 'paw_HR'  
distances_VR_HR = calculate_distance_between_bodyparts(dataframe, bodypart1, bodypart2)

bodypart1 = 'snout'  
bodypart2 = 'anus'  
distances_S_A = calculate_distance_between_bodyparts(dataframe, bodypart1, bodypart2)



#%% lets make some plots of things



#lets plot the paws for a certain set of frames
bodyparts_to_plot = ['paw_HR','paw_HL','paw_VR', 'paw_VL']  # Add the actual body part names you want to plot
frames_range = (f_Start, f_End)  # Set your desired frame range
plot_bodyparts(dataframe, bodyparts_to_plot, frames_range)

#lets plot the paws for a certain set of frames
bodyparts_to_plot = ['anus','snout', 'mouth']  # Add the actual body part names you want to plot
frames_range = (f_Start, f_End)  # Set your desired frame range
plot_bodyparts(dataframe, bodyparts_to_plot, frames_range)


# distance between paw-pairs versus time
fig, axs = plt.subplots(2, 1)

axs[0].plot(distances_VL_HL[f_Start:f_End], color = 'cyan')
axs[0].plot(distances_VR_HR[f_Start:f_End], color = 'magenta')
axs[0].plot(distances_S_A[f_Start:f_End], color = 'grey')

axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Distance (pixels)')
axs[0].set_title('Distance Between 2 Body Parts')
axs[0].legend(['FL-BL', 'FR-BR', 'S-A'])



# distance pairs of paw-pairs
axs[1].scatter(distances_VL_HL[f_Start:f_End],distances_VR_HR[f_Start:f_End], color = 'black', alpha = 0.01)
axs[1].set_xlabel('Distance FL-BL (pixels)')
axs[1].set_ylabel('Distance FR-BR (pixels)')
axs[1].set_aspect('equal', 'box')

# spectrogram paw distance
fig, axs = plt.subplots(2, 1)

axs[0].plot(distances_VL_HL[f_Start:f_End], color = 'cyan')
axs[0].autoscale(enable=None, axis="x", tight=True)
Pxx, freqs, bins, im = axs[1].specgram(distances_VR_HR[f_Start:f_End], NFFT=125, noverlap = 62)
