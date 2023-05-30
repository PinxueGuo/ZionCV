import os

'''-----------OxUvA------------'''
data_path = '/home/guopx/VOS/DATA/OxUvA/images/dev'
save_name = '/home/guopx/VOS/MAST/functional/feeder/dataset/OxUvA.csv'

dir_list = os.listdir(data_path)
dir_list.sort()

save_list = []

for vid_name in dir_list:
    frames_name = os.listdir(os.path.join(data_path, vid_name))
    frames_name.sort()

    '''-----------OxUvA------------'''
    start_int = int(frames_name[0][0:6])
    end_int = int(frames_name[-1][0:6])
    nframes = end_int-start_int+1
    query_1 = str(start_int)
    query_2 = str(nframes)
    save_list.append( vid_name+','+query_1+','+query_2+'\n')


with open(save_name, 'w') as f:
    f.writelines(save_list)

