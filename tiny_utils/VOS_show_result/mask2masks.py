import cv2
import os
from PIL import Image
import numpy as np

img_name = '/mnt/share/et21-guopx/codes/BiDecVOS/DATA/DAVIS/2017/trainval/Annotations/480p/bike-packing/00000.png'

palette_path = 'palette_davis.png'
palette = Image.open(palette_path).getpalette()

mask = Image.open(img_name).convert('P')
mask = np.array(mask)

for i in range(1, mask.max()+1):
    mask_i = np.array(mask, copy=True)
    mask_i[mask!=i] = 0
    mask_i = Image.fromarray(mask_i)
    mask_i.putpalette(palette)
    mask_i.save(str(i)+'.png')
    