import os 
import numpy
import cv2
from PIL import Image

palette_path = r'palette_davis.png'
palette = Image.open(palette_path).getpalette()

mask_dir = r'BiDec_result'

for video_name in os.listdir(mask_dir):
    video_dir = os.path.join(mask_dir, video_name)
    for frame_name in os.listdir(video_dir):
        frame = os.path.join(video_dir, frame_name) # every frame path

        image = Image.open(frame).convert('P')
        image.putpalette(palette)
        image.save(frame)                           # overwrite the mask file