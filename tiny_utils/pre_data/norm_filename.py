import os

dataset_dir = 'MinVIS/UVO_demo_videos'

videos_list = os.listdir(dataset_dir)

for video in videos_list:
    video_path = os.path.join(dataset_dir, video)
    imgs_list = os.listdir(video_path)
    for img in imgs_list:
        old_name = os.path.join(video_path, img)
        img_id_int = int(img.split('.')[0])
        new_name = os.path.join(video_path, '{:05d}.jpg'.format(img_id_int))
        os.rename(old_name, new_name)