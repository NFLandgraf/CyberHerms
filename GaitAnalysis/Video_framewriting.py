#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import cv2
from matplotlib import image 
import os
import re
from PIL import Image
from PIL import ImageDraw



input_file = '3m_292_editDLC_resnet50_WalterWhitiesMar26shuffle1_750000_filtered_labeled.mp4'


raw_folder = 'raw'



def write_all_frames_from_video(input_file, output_folder):
    # takes video and creates png for every frame in video
    os.makedirs(output_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(input_file)
    nframes = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, image = vidcap.read()

    print('\nwrite_all_frames_from_video')
    count = 0
    pbar = tqdm(total=int(nframes))
    while success:
        cv2.imwrite(output_folder + '\\frame%d.png' % count, image)
        success,image = vidcap.read()
        count += 1
        pbar.update(1)

write_all_frames_from_video(input_file, raw_folder)

#%%

edit_folder = 'edit'

def draw_and_write(input_folder, output_folder):
    # takes pngs and draws polygon onto it, if mouse in that position
    os.makedirs(output_folder, exist_ok=True)

    images = [file for file in os.listdir(input_folder) if file.endswith('.png')]

    # sort the list of strings based on the int in the string
    sorted_intimages = sorted([int(re.sub("[^0-9]", "", element)) for element in images])
    sorted_images = [f'frame{element}.png' for element in sorted_intimages]

    print('\ndraw_and_write')
    for i in tqdm(range(len(sorted_images))):
        
        image_path = os.path.join(input_folder, sorted_images[i])
        data = cv2.imread(image_path)

        image = cv2.putText(data, str(i), (50, 50) , cv2.FONT_HERSHEY_SIMPLEX ,  1, (255, 0, 0) , 2, cv2.LINE_AA) 
        
        output_path = os.path.join(output_folder, f"frame{i}.png")
        cv2.imwrite(output_path, image)


draw_and_write(raw_folder, edit_folder)


#%%

input_folder = 'edit'

def video_from_frames(input_folder):
    # created video from all frames with drawn polygons
    output = os.path.join(f'video_edit.mp4')

    images = [file for file in os.listdir(input_folder) if file.endswith('.png')]

    # sort the list of strings based on the int in the string
    sorted_intimages = sorted([int(re.sub("[^0-9]", "", element)) for element in images])
    sorted_images = [f'frame{element}.png' for element in sorted_intimages]

    # setting the video properties according to first image 
    frame0 = cv2.imread(os.path.join(input_folder, sorted_images[0]))
    height, width, layers = frame0.shape 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, 20, (width, height)) 

    # Appending the images to the video
    print('\nvideo_from_frames')
    for i in tqdm(range(len(sorted_images))):
        img_path = os.path.join(input_folder, sorted_images[i])
        img = cv2.imread(img_path)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


video_from_frames(input_folder)
