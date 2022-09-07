import os
import cv2
import numpy as np
import math
from sklearn import linear_model
import matplotlib.pyplot as plt

def LineRegression(data):
    # data: np.array, [[x,y], [x,y] ... ]
    x = data[:,0:1]
    y = data[:,1:]

    LR = linear_model.LinearRegression()
    LR.fit(x, y)

    return LR

def LR_pred(x, LR):
    y = x*LR.coef_ + LR.intercept_
    return y


frame_dir = "/Users/pxguo/Documents/LocalData/my_videos/JPEGImages/water_short/"
save_dir = "/Users/pxguo/Documents/LocalData/my_videos/JPEGImages/water_short_result/"
frame_list = os.listdir(frame_dir)
for frame in frame_list:
    frame_path = os.path.join(frame_dir, frame)
    save_path = os.path.join(save_dir, frame)
    src = cv2.imread(frame_path)
    # crop and canny
    src_crop=src[120:880,1680:] 
    img = src_crop.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst_img = cv2.Canny(gray_img, 50, 100)

    # find points
    h, w = dst_img.shape
    up_points = []          # [(x,y), ...]
    down_points = []
    for j in range(w):
        candidate_points = np.where(dst_img[:,j]==255)[0]
        if len(candidate_points)<2:
            continue
        up = candidate_points[-1]
        down = candidate_points[0]
        # if frame=='123.png':
        #     print(candidate_points)
        #     print(up)
        #     print(down)
        if up>550 or down<120:
            break

        up_points.append([j,up])
        down_points.append([j,down])
    up_points = np.array(up_points)
    down_points = np.array(down_points)

    # regress the line
    # print(up_points.shape)
    # print(down_points.shape)
    line_up = LineRegression(up_points)
    line_down = LineRegression(down_points)
    L1_coef, L1_intercept = line_up.coef_, line_up.intercept_
    L2_coef, L2_intercept = line_down.coef_, line_down.intercept_

    # 求出弧度值
    angR = abs(math.atan((L2_coef - L1_coef) / (1 + L1_coef * L2_coef)))
    # 弧度转换为角度
    angD = round(math.degrees(angR))
    print(frame, angD)

    # x_test = np.array(list(range(0,w)))[:,None]
    # y_test_up = LR_pred(x_test, line_up)
    # y_test_down = LR_pred(x_test, line_down)

    # plt.imshow(src)
    # plt.plot(x_test+1680 ,y_test_up+120 ,color='green',linewidth=2)
    # plt.plot(x_test+1680 ,y_test_down+120 ,color='green',linewidth=2)
    # plt.annotate(str(angD)+'°',(50,50), color='while')
    # # plt.show()
    # plt.savefig(save_path)
    # plt.clf()

    pt1 = [0+1680, int(0*line_up.coef_ + line_up.intercept_+120)]
    pt2 = [w+1680, int(w*line_up.coef_ + line_up.intercept_+120)]
    pt3 = [0+1680, int(0*line_down.coef_ + line_down.intercept_+120)]
    pt4 = [w+1680, int(w*line_down.coef_ + line_down.intercept_+120)]
    green = (0,255,0)
    cv2.line(src, pt1, pt2, green, 2) 
    cv2.line(src, pt3, pt4, green, 2)
    cv2.putText(src, str(angD), (1700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, green, 2)
    cv2.imwrite(save_path, src)
