import numpy as np 
import pickle                               # 解包数据集
import torch as t                           # 仅为了转换tensor类型方便操作
from torchvision.utils import save_image    # 方便可视化


def unpickle(file):
    # 加载解码二进制数据集
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def Kmeans_a_echo():
# 使用切片与np函数 一次性快速计算出每个点与每个centre间的距离
    for i, cen in enumerate(centre):
        # L2 distance
        # dis[:,i] = np.linalg.norm(data-cen,axis=1,keepdims=True).reshape(l)   
        # L1 distance  
        dis[:,i] = np.linalg.norm(data-cen,axis=1,keepdims=True).reshape(l)  
    # 选取距离最近的聚类中心确定类别
    belong = np.argmin(dis,axis=1)                                              
    for i in range(k):
        # 更新聚类中心
        centre[i] = np.average(data[belong==i],axis=0)                          
    return belong

def DaviesBouldin(X, labels):
    # 评价指标CBI函数
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    #求S
    S = [np.mean([np.linalg.norm(p-centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    Ri = []
    for i in range(n_cluster):
        Rij = []
        #计算Rij
        for j in range(n_cluster):
            if j != i:
                r = (S[i] + S[j]) / np.linalg.norm(centroids[i]-centroids[j])
                Rij.append(r)
         #求Ri
        Ri.append(max(Rij)) 
    # 求dbi  
    dbi = np.mean(Ri)
    return dbi

file_name = "/home/guopx/ZionCV/DATA/cifar-10/data_batch_1"         # cifar-10文件
data_dict = unpickle(file_name)                                     # 加载、解码 数据
label = data_dict[b'labels']                                        # list, len=1000
data = data_dict[b'data']/255.0                                     # array, shape=(10000, 3072)，且归一化

l = data.shape[0]                       # 数据个数
k = 10                                  # 规定簇数量
epoch = 5                               # 设定训练轮数

data_show_list = []
for i in range(10):
    i_list = [data[j] for j in range(l) if label[j]==i]             # 遍历每个类别
    data_show_list.append(i_list[:5])                               # 每个类别五张
data_show = t.tensor(data_show_list).reshape(-1, 3, 32, 32)       # 转为tensor + reshape为目标维度 [B, 3, H, W]
save_image(data_show, 'data_src.png', nrow=5)                         # 可视化 原始图像

centre_num = np.random.randint(0,l,size=10)     # 随机初始聚类中心index
centre = data[centre_num]                       # 初始聚类中心
dis = np.zeros((l,k))                           # 距离矩阵

for i in range(epoch):
    belong = Kmeans_a_echo()                    # 一轮 K-means
    print(belong)
    print('--------------')

images_show_list = []
for i in range(k):
    # 每一类加五个图片在可视化列表里
    ki_list = [data[j] for j in range(l) if belong[j]==i]
    images_show_list.append(ki_list[:5])
    
# torch.Size([10, 5, 3072])
images_show_tensor = t.tensor(images_show_list).float()     
# 为save_image() 做reshape
images_show_tensor = images_show_tensor.reshape(10*5, 3, 32, 32)    
# 用torchvision中的save_image()把聚类结果可视化
save_image(images_show_tensor, 'kmeans_result_L1.jpg', nrow=5)      

print(DaviesBouldin(data, belong))      
# L1: 3.1303719191121546
# L2；3.088973239395308

