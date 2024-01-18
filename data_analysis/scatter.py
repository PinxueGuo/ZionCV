# scatter visualization for BiDecVOS olml frames classify (yes/no/optional).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel('classifier_data.xlsx')

label = df['label']
kl_pp = df['pp_kl']
kl_prev = df['prev_kl']
kl_curr = df['curr_kl']

 
label_0_kl_prev = []; label_0_kl_curr = []; # label_0_kl_pp = [];
label_1_kl_prev = []; label_1_kl_curr = []; # label_1_kl_pp = []; 
label_2_kl_prev = []; label_2_kl_curr = []; # label_2_kl_pp = []; 


for i in range(len(label)):
    if label[i] == 0:
        # label_0_kl_pp.append(kl_pp[i])
        label_0_kl_prev.append(kl_prev[i])
        label_0_kl_curr.append(kl_curr[i])
    if label[i] == 1:
        # label_1_kl_pp.append(kl_pp[i])
        label_1_kl_prev.append(kl_prev[i])
        label_1_kl_curr.append(kl_curr[i])
    if label[i] == 2:
        # label_2_kl_pp.append(kl_pp[i])
        label_2_kl_prev.append(kl_prev[i])
        label_2_kl_curr.append(kl_curr[i])


fig, ax = plt.subplots()
# ax.scatter(label_0_kl_prev, label_0_kl_curr, label_0_kl_pp, c='red', label='no')
# ax.scatter(label_1_kl_prev, label_1_kl_curr, label_1_kl_pp, c='yellow', label='just so so')
# ax.scatter(label_2_kl_prev, label_2_kl_curr, label_2_kl_pp, c='green', label='yes')
ax.scatter(label_0_kl_prev, label_0_kl_curr, c='red', label='no')
ax.scatter(label_1_kl_prev, label_1_kl_curr, c='yellow', label='just so so')
ax.scatter(label_2_kl_prev, label_2_kl_curr, c='green', label='yes')

ax.legend()
plt.savefig('show.jpg')