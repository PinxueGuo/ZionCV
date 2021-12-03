import os

path = '/home/guopx/VOS/DATA/YT18/valid/JPEGImages'
dir_list = os.listdir(path)

dir_list = [item+'\n' for item in dir_list]

svae_path = '/home/guopx/VOS/DATA/YT18/valid/val.txt'
with open(svae_path, 'w') as f:
    f.writelines(dir_list)