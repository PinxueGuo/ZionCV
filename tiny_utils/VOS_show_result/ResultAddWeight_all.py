import cv2
import os

val_txt = 'dataset/val-davis.txt'
image_path = '/Users/pxguo/Documents/LocalData/DAVIS/JPEGImages/480p'
mask_path = '/Users/pxguo/Documents/LocalData/result_STCN/stcn_davis'
save_path = '/Users/pxguo/Documents/LocalData/result_STCN/stcn_davis_AddWeight'


val_list = open(val_txt).readlines()
val_list = [line.strip() for line in val_list]

for video in val_list:
    # print(video)
    image_dir = os.path.join(image_path, video)
    mask_dir = os.path.join(mask_path, video)
    save_dir = os.path.join(save_path, video)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in os.listdir(image_dir):
        image_filename = os.path.join(image_dir, i)
        mask_filename = os.path.join(mask_dir, i[:6])+'png'
        save_filename = os.path.join(save_dir, i)

        image = cv2.imread(image_filename)
        mask = cv2.imread(mask_filename)
        
        result = cv2.addWeighted(image, 1, mask, 1.5, 0)
        cv2.imwrite(save_filename, result)
    print(video, ': done!')
