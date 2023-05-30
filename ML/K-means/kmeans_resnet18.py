import numpy as np 
import pickle                               # 解包数据集
import torch as t                           # 仅为了转换tensor类型方便操作
from torchvision.utils import save_image    # 方便可视化
from torchvision.models import resnet18     # 用于提取特征的CNN


def unpickle(file):
    # 加载解码二进制数据集
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def Kmeans_a_echo(feature, centre):
# 使用切片与np函数 一次性快速计算出每个点与每个centre间的距离
    for i, cen in enumerate(centre):
        dis[:,i] = np.linalg.norm(feature-cen,axis=1,keepdims=True).reshape(l)     # L2 distance
        # dis[:,i] = np.linalg.norm(feature-cen,axis=1,keepdims=True).reshape(l)     # L1 distance
    belong = np.argmin(dis,axis=1)                                              # 选取距离最近的聚类中心确定类别
    for i in range(k):
        centre[i] = np.average(feature[belong==i],axis=0)                        # 更新聚类中心
    return belong, centre

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


file_name = "/home/guopx/ZionCV/DATA/cifar-10/data_batch_1"             # cifar-10文件
data_dict = unpickle(file_name)                                         # 加载、解码 数据
label = data_dict[b'labels']                                            # lisr, len=1000
data = data_dict[b'data']/255.0                                         # array, shape=(10000, 3072)，且归一化

l = data.shape[0]   # 数据个数
k = 10              # 规定簇数量
epoch = 5           # 设定训练轮数

data_show_list = []
for i in range(10):
    i_list = [data[j] for j in range(l) if label[j]==i]             # 遍历每个类别
    data_show_list.append(i_list[:5])                               # 每个类别五张
data_show = t.tensor(data_show_list).reshape(-1, 3, 32, 32)         # 转为tensor + reshape为目标维度 [B, 3, H, W]
save_image(data_show, 'data_src.png', nrow=5)                       # 可视化 原始图像

with t.no_grad():
    data_tensor = t.tensor(data, dtype=t.float32).reshape(10000, 3, 32, 32)         # 把原始图像数据转为tensor类型，且reshape为(batch, ch, h, w)
    res18 = resnet18(pretrained=True, progress=False).eval()                        # 加载resnet18
    feature_extractor = t.nn.Sequential(res18.conv1, res18.bn1, res18.relu, res18.maxpool, res18.layer1)    # 只用resnet18的第一层特区特征
    feature = feature_extractor(data_tensor)                # 十分简单的卷积特征提取器
    feature_np = feature.reshape(10000,-1).numpy()

feature_show_list = []
for i in range(10):
    # 遍历每个类别
    i_list = [feature_np[j] for j in range(l) if label[j]==i]  
    # 每个类别五张          
    feature_show_list.append(i_list[:5])                               
data_show = t.tensor(feature_show_list).reshape(-1, 1, 8, 8)
# 可视化 特征
save_image(data_show[::64], 'feature.png', nrow=5)                       


centre_num = np.random.randint(0,l,size=10)     # 随机初始聚类中心index
centre = feature_np[centre_num]                 # 初始聚类中心
dis = np.zeros((l,k))                           # 距离矩阵

for i in range(epoch):
    belong, centre = Kmeans_a_echo(feature_np, centre)      # 一轮 K-means
    print(belong)
    print('--------------')

images_show_list = []
for i in range(k):
    # 每一类加五个图片在可视化列表里
    ki_list = [data[j] for j in range(l) if belong[j]==i]
    images_show_list.append(ki_list[:5])

images_show_tensor = t.tensor(images_show_list).float()         # torch.Size([10, 5, 3072])
images_show_tensor = images_show_tensor.reshape(10*5, 3, 32, 32)        # 为save_image() 做reshape
save_image(images_show_tensor, 'kmeans_result_res18_L2.jpg', nrow=5)       # 用torchvision中的save_image()把聚类结果可视化

print(DaviesBouldin(data, belong))      
# L1: 8.049914871146957
# L2: 6.389304211182785
