import os
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
 
frame_dir = "/Users/pxguo/Documents/LocalData/my_videos/JPEGImages/water_short/"
save_dir = '/Users/pxguo/Documents/LocalData/my_videos/JPEGImages/water_short_canny/'
frame_list = os.listdir(frame_dir)
for frame in frame_list:
    frame_path = os.path.join(frame_dir, frame)
    save_path = os.path.join(save_dir, frame)

    src = cv.imread(frame_path)
    img = src.copy()
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst_img = cv.Canny(gray_img, 50, 100)

    # lines = cv.HoughLinesP(dst_img, 1, np.pi / 360, 20, minLineLength=10, maxLineGap=10)
    # for line in lines: 
    #     for x1, y1, x2, y2 in line:
    #         cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv.imwrite(save_path, dst_img)
    # cv.waitKey(0)

