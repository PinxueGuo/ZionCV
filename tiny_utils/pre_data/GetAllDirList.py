import os

path = '/home/guopx/VOS/STCN_root/YouTube/valid/JPEGImages'
dir_list = os.listdir(path)

dir_list = [item+'\n' for item in dir_list]

svae_path = '../VOS_show_result/val-YTB19.txt'
with open(svae_path, 'w') as f:
    f.writelines(dir_list)