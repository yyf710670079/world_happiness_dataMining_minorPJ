#   _*_ coding:utf-8 _*_
__author__ = 'yangyufeng'

"""
预处理数据：
先检查变量之间知否有大的相关性。
PCA算法
"""

import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.cluster import KMeans,DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 读取数据
data = []
country_name = []
line_count = 0
happiness = []
std_err = []
with open("2015.csv",'r',encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        if line_count == 0:
            var_name_list = line
        else:
            country_name.append(line[0])
            data.append([float(obj) for obj in line[5:-1]])
            happiness.append(float(line[3]))
            std_err.append(float(line[4]))
        line_count += 1

data = np.array(data)
happiness = np.array(happiness)
std_err = np.array(std_err)

'''
# 先查看6个变量之间的整体相关性多大
data = np.array(data)
pca1 = PCA(n_components=6)
pca1.fit(data)
print("pca.explained_variance_ratio:" + str(pca1.explained_variance_ratio_))


# 再来观察降到两维的可视化效果
pca2 = PCA(n_components=2)
pca2.fit(data)
new_data = pca2.fit_transform(data)
# print(new_data)

plt.figure(1)
plt.scatter(new_data[:, 0], new_data[:, 1])
plt.xlabel('x1_transform')
plt.ylabel('x2_transform')
plt.show()


# 经济与幸福指数一元线性回归
plt.figure(2)
plt.scatter(data[:, 0], happiness)
plt.xlabel('Economy')
plt.ylabel('Happiness score')

clf = linear_model.LinearRegression()
clf.fit(data[:, 0].reshape(-1, 1), happiness)
print("clf.coef_: ", clf.coef_)
print('clf.intercept_: ', clf.intercept_)

plt.scatter(data[:, 0], clf.predict(data[:, 0].reshape(-1, 1)), c='r')
plt.show()


# 一元回归显著性检验
x = np.array(data[:, 0])
y = np.array(happiness).T
n = len(happiness)
Sxx = sum(x**2) - (sum(x)**2)/n
Syy = sum(y**2) - (sum(y)**2)/n
Sxy = sum(x*y) -(sum(x))*(sum(y))/n
sigma2 = (Syy - 2.2182 * Sxy)/(n-2)
t = (2.2182/np.sqrt(sigma2))*np.sqrt(Sxx)
print('Sxx: ', Sxx)
print('Syy: ', Syy)
print('Sxy: ', Sxy)
print('sigma2: ', sigma2)
print('t检验的t: ', t)


# 各变量与幸福指数的散点图和一元回归
plt.figure(4)

plt.subplot(331)
plt.scatter(data[:, 0], happiness)
plt.xlabel('Economy')
plt.ylabel('Happiness score')
clf1 = linear_model.LinearRegression()
clf1.fit(data[:, 0].reshape(-1, 1), happiness)
plt.scatter(data[:, 0], clf1.predict(data[:, 0].reshape(-1, 1)), c='r')


plt.subplot(332)
plt.scatter(data[:, 1], happiness)
plt.xlabel('Family')
plt.ylabel('Happiness score')
clf2 = linear_model.LinearRegression()
clf2.fit(data[:, 1].reshape(-1, 1), happiness)
plt.scatter(data[:, 1], clf2.predict(data[:, 1].reshape(-1, 1)), c='r')

plt.subplot(333)
plt.scatter(data[:, 2], happiness)
plt.xlabel('Health')
plt.ylabel('Happiness score')
clf3 = linear_model.LinearRegression()
clf3.fit(data[:, 2].reshape(-1, 1), happiness)
plt.scatter(data[:, 2], clf3.predict(data[:, 2].reshape(-1, 1)), c='r')

plt.subplot(334)
plt.scatter(data[:, 3], happiness)
plt.xlabel('Freedom')
plt.ylabel('Happiness score')
clf4 = linear_model.LinearRegression()
clf4.fit(data[:, 3].reshape(-1, 1), happiness)
plt.scatter(data[:, 3], clf4.predict(data[:, 3].reshape(-1, 1)), c='r')

plt.subplot(335)
plt.scatter(data[:, 4], happiness)
plt.xlabel('Trust')
plt.ylabel('Happiness score')
clf5 = linear_model.LinearRegression()
clf5.fit(data[:, 4].reshape(-1, 1), happiness)
plt.scatter(data[:, 4], clf5.predict(data[:, 4].reshape(-1, 1)), c='r')

plt.subplot(336)
plt.scatter(data[:, 5], happiness)
plt.xlabel('Generosity')
plt.ylabel('Happiness score')
clf6 = linear_model.LinearRegression()
clf6.fit(data[:, 5].reshape(-1, 1), happiness)
plt.scatter(data[:, 5], clf6.predict(data[:, 5].reshape(-1, 1)), c='r')

plt.show()



# Trust and Economy
plt.figure(5)
plt.scatter(data[:, 0], data[:, 4], c=happiness)
plt.xlabel('Economy')
plt.ylabel('Trust')
plt.show()


# DBSCAN
X = np.array([data[:, 0], data[:, 4]]).T
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
dbscan = DBSCAN(eps=0.06, min_samples=2).fit(X)
plt.figure(6)
plt.scatter(data[:, 0], data[:, 4], c=dbscan.fit_predict(X))
plt.xlabel('Economy')
plt.ylabel('Trust')
plt.show()



# std_err
plt.figure(6)
plt.scatter(data[:, 3], std_err, c=happiness)
plt.xlabel('Economy')
plt.ylabel('Std_error')
plt.show()



plt.figure(7)
for i in range(6):
    plt.subplot(331+i)
    plt.scatter(data[:, i], std_err, c=happiness)
    plt.xlabel(var_name_list[5+i])
    plt.ylabel('Std_error')
plt.show()

'''
X = np.array([data[:, 0], data[:, 2], data[:, 3]]).T
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
plt.figure(8)
ax = plt.subplot(122, projection='3d')
ax.scatter(data[:, 0], data[:, 2], data[:, 3], c=kmeans.predict(X))
ax.set_zlabel('Freedom')  # 坐标轴
ax.set_ylabel('Health')
ax.set_xlabel('Economy')

ax = plt.subplot(121, projection='3d')
ax.scatter(data[:, 0], data[:, 2], data[:, 3], c=happiness)
ax.set_zlabel('Freedom')  # 坐标轴
ax.set_ylabel('Health')
ax.set_xlabel('Economy')

plt.show()