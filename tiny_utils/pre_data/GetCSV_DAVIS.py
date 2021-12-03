import os

data_path = '/home/guopx/VOS/DATA/DAVIS/JPEGImages/480p'
save_name = '/home/guopx/VOS/MAST/functional/feeder/dataset/DAVIS.csv'
txt = '/home/guopx/VOS/DATA/DAVIS/ImageSets/2017/train.txt'

dir_list = open(txt).readlines()

save_list = []

for vid_name in dir_list:
    frames_name = os.listdir(os.path.join(data_path, vid_name.strip()))
    frames_name.sort()

    '''-----------OxUvA------------'''
    start_int = int(frames_name[0][0:5])
    end_int = int(frames_name[-1][0:5])
    nframes = end_int-start_int+1
    query_1 = str(start_int)
    query_2 = str(nframes)
    save_list.append( vid_name.strip()+','+query_1+','+query_2+'\n')


with open(save_name, 'w') as f:
    f.writelines(save_list)
