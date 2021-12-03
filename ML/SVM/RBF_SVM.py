import numpy as np                                      # 使用numpy处理array数据格式
import torch as t                                       # 借用pytorch处理矩阵数据
from torchvision import utils                           # 借用torchvision来可视化
from sklearn.datasets import load_digits                # sklearn的minist数据集接口
from sklearn.model_selection import train_test_split    # 分割train-set和test-set工具
from sklearn.preprocessing import StandardScaler        # 数据标准化
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report       # 定量结果分析
from sklearn.metrics import confusion_matrix            # 混淆矩阵计算接口
import matplotlib.pyplot as plt



# 加载数据集
digits = load_digits()

# 查看数据维度， 1797张 每张8*8大小
print(digits.data.shape)    # (1797, 64)

# 长度为64的一维向量
print(digits.data[0])       

# data, label 分别分成 train-set 和 test-set(总数据量的1/4)
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

# 查看 train-set的data形状
print(X_train.shape)  # (1347,64)
# 查看 test-set的label形状
print(Y_test.shape)  # (450,)
# 查看 test-set的label值
print(Y_test)

# 对train-set和test-set的图像数据做标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 创建SVM实例
# rbf kernel
svm = SVC(kernel='rbf')

# 训练SVM
svm.fit(X_train, Y_train)

# 用训好的SVM预测test的图像分类结果
Y_predict = svm.predict(X_test)

# 定量性能评价
print(classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str)))

# 计算分类结果的混淆矩阵
confusion = confusion_matrix(Y_test, Y_predict)
print(confusion)

for first_index in range(len(confusion)):    #第几行
    for second_index in range(len(confusion[first_index])):    #第几列
        plt.text(first_index, second_index, confusion[first_index][second_index])
        plt.imshow(confusion, interpolation='nearest', cmap='Blues')
# 
plt.savefig('confusion.png')

to_show = t.tensor(X_test).reshape(-1, 1, 8, 8)     # 转为tensor + reshape为目标维度 [B, 1, H, W]
utils.save_image(to_show, 'result.png')             # 用torchvision中的save_image函数把所有测试图同时展示出来
print(Y_predict)                                    # 打印预测结果
