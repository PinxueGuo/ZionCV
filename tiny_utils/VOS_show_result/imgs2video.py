import os
import cv2
import time

videos_dir = '/home/guopx/VOS/my_utils/result/Our_1'
videos_list = os.listdir(videos_dir)
fps = 5

for video in videos_list:
    images_dir = os.path.join(videos_dir, video)
    save_path = os.path.join(videos_dir, video+'.mp4')

    # images in this video
    images_list = os.listdir(images_dir)
    images_list.sort()
    shape = cv2.imread(os.path.join(images_dir, images_list[0])).shape[:2]      # eg: (480, 854) 
    size = (shape[1], shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, size)

    for image in images_list:
        if image.endswith('.jpg'):   #判断图片后缀是否是.jpg
            image = os.path.join(images_dir, image)
            img = cv2.imread(image) #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            # print(type(img))  # numpy.ndarray类型  
            videoWriter.write(img)        #把图片写进视频

    videoWriter.release() #释放