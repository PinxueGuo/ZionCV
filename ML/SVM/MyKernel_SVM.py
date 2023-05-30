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

# myself kernel
def my_cosine_kernel(x1,x2):
    # 广义Jaccard相似度（Tanimoto系数） 作为kernel
    dot = np.dot(x1, x2.transpose())
    cos_similarity = dot / (np.linalg.norm(x1) + np.linalg.norm(x2) - dot)
    return cos_similarity
# 创建SVM实例
svm = SVC(kernel=my_cosine_kernel)     # kernrl: ['linear', 'rbf', 'poly']

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

# 输出结果
# (1797, 64)
# [ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
#  15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
#   0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
#   0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]
# (1347, 64)
# (450,)
# [1 3 7 3 2 4 6 1 4 0 4 7 8 9 2 8 3 6 7 0 6 0 8 3 0 6 2 3 0 9 0 2 0 6 9 1 1
#  5 8 0 6 1 5 8 9 5 1 6 2 6 6 7 6 7 7 2 7 8 0 7 3 6 3 9 6 6 5 5 4 2 9 3 7 6
#  5 7 2 8 1 2 2 8 1 1 6 3 5 0 0 1 6 7 5 8 9 7 0 0 9 8 0 8 2 3 6 1 9 9 1 7 8
#  9 8 8 5 9 5 1 1 9 9 3 3 2 8 1 3 8 6 4 0 0 0 7 1 5 5 1 8 5 1 8 8 6 9 9 4 5
#  7 5 2 1 2 3 8 7 7 5 1 9 1 9 8 0 6 1 2 1 3 3 8 9 6 8 4 1 0 0 9 8 7 2 8 6 4
#  8 9 4 2 6 1 8 5 6 7 5 1 9 2 8 3 2 9 4 8 5 5 6 2 4 3 2 6 4 8 5 8 0 8 8 6 3
#  2 3 0 5 7 1 3 9 3 2 1 6 6 5 1 9 7 2 4 5 2 1 3 1 1 2 1 7 0 1 2 2 1 2 4 9 6
#  6 3 9 2 8 1 5 5 1 8 6 2 5 6 0 1 4 2 1 8 9 4 3 0 6 8 3 3 2 0 2 5 6 5 6 6 4
#  6 1 8 3 4 1 3 5 1 4 9 8 7 5 1 1 3 7 8 8 3 7 4 0 7 2 9 7 1 9 4 5 3 5 2 5 1
#  3 0 5 8 4 7 6 9 9 3 3 4 8 6 4 7 0 6 8 2 3 3 4 5 3 3 5 2 0 9 7 1 5 5 8 4 4
#  3 6 2 5 1 0 6 1 5 8 4 9 6 4 3 8 0 3 0 1 2 8 0 5 4 5 2 8 9 6 9 8 0 8 8 2 4
#  6 5 6 4 3 9 8 9 7 1 7 9 4 1 5 9 5 9 8 6 8 2 5 1 4 2 6 3 7 9 8 7 4 3 7 1 8
#  8 9 5 3 6 6]
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00        35
#            1       0.83      0.98      0.90        54
#            2       1.00      1.00      1.00        44
#            3       0.98      0.91      0.94        46
#            4       1.00      0.91      0.96        35
#            5       0.94      0.94      0.94        48
#            6       0.98      0.94      0.96        51
#            7       0.85      1.00      0.92        35
#            8       0.92      0.79      0.85        58
#            9       0.86      0.86      0.86        44

#     accuracy                           0.93       450
#    macro avg       0.94      0.93      0.93       450
# weighted avg       0.93      0.93      0.93       450

# [[35  0  0  0  0  0  0  0  0  0]
#  [ 0 53  0  0  0  0  0  0  0  1]
#  [ 0  0 44  0  0  0  0  0  0  0]
#  [ 0  0  0 42  0  1  0  1  2  0]
#  [ 0  0  0  0 32  0  0  1  2  0]
#  [ 0  0  0  0  0 45  1  0  0  2]
#  [ 0  3  0  0  0  0 48  0  0  0]
#  [ 0  0  0  0  0  0  0 35  0  0]
#  [ 0  7  0  1  0  1  0  0 46  3]
#  [ 0  1  0  0  0  1  0  4  0 38]]
# [1 3 7 3 2 4 6 1 4 0 4 7 8 5 2 8 8 6 7 0 6 0 9 3 0 6 2 3 0 9 0 2 0 6 9 1 1
#  5 8 0 6 1 5 8 9 5 1 6 2 6 6 7 6 7 7 2 7 8 0 7 3 6 3 9 6 6 5 5 4 2 1 3 7 6
#  5 7 2 8 1 2 2 8 1 1 6 3 5 0 0 1 6 7 6 8 9 7 0 0 9 8 0 8 2 3 6 1 9 9 1 7 3
#  7 8 8 5 9 5 1 1 7 9 3 3 2 8 1 3 8 6 4 0 0 0 7 1 5 5 1 8 5 1 8 1 6 9 9 4 5
#  7 5 2 1 2 3 8 7 7 5 1 9 1 9 8 0 6 1 2 1 3 7 9 7 6 8 4 1 0 0 9 9 7 2 8 1 4
#  8 9 4 2 6 1 8 9 6 7 5 1 9 2 8 8 2 9 4 1 5 5 6 2 4 3 2 6 4 8 5 8 0 8 8 6 3
#  2 3 0 5 7 1 3 9 3 2 1 6 6 5 1 9 7 2 4 5 2 9 5 1 1 2 1 7 0 1 2 2 1 2 8 9 6
#  6 3 9 2 8 1 5 5 1 1 6 2 5 6 0 1 4 2 1 8 9 7 3 0 6 8 3 3 2 0 2 9 6 5 6 6 4
#  6 1 8 3 4 1 3 5 1 4 9 8 7 5 1 1 3 7 1 8 3 7 4 0 7 2 9 7 1 9 4 5 3 5 2 5 1
#  3 0 5 8 4 7 1 9 9 3 3 4 5 6 4 7 0 6 1 2 3 3 4 5 3 3 5 2 0 9 7 1 5 5 8 4 4
#  3 6 2 5 1 0 6 1 5 8 4 7 6 4 3 8 0 3 0 1 2 8 0 5 4 5 2 8 9 6 9 8 0 1 1 2 4
#  6 5 6 4 3 9 8 9 7 1 7 9 8 1 5 9 5 9 8 1 8 2 5 1 4 2 6 3 7 9 8 7 4 3 7 1 8
#  8 9 5 3 6 6]