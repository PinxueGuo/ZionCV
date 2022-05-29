import os

path = '/mnt/share/et21-guopx/codes/PLR-MAMP/datasets/youtube_2018/train/JPEGImages'
dir_list = os.listdir(path)

dir_list = [item+'\n' for item in dir_list]

svae_path = 'tiny_utils/VOS_show_result/train-YTB18.txt'
with open(svae_path, 'w') as f:
    f.writelines(dir_list)