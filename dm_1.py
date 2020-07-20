import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 读取数据，并划分训练集和测试集
rate = pd.read_csv("C:\\Users\\asus\\Desktop\\大三下\\数据挖掘\\ml-latest-small\\ratings.csv")
# rate = rate[0:2000]
len = rate.shape[0]
testlen = int(len/10)
testmark = np.zeros((len, 1))
count = 0
for i in range(testlen):
    index = int(np.random.uniform(0, len))
    testmark[index, 0] = 1
    # print(index)
testrate = np.zeros((1, rate.shape[1]), dtype=int)
# print(rate)

# 建立索引
movielist = rate['movieId'].unique()
# print(movielist)
# print(movielist.shape[0])
userlist = rate['userId'].unique()
m = userlist.shape[0]
n = movielist.shape[0]


A = np.zeros((n, m))  # 建立二部图矩阵
f = np.zeros((n, m))  # 建立最终资源分配矩阵
mark = np.zeros((n, m))  # 标记看过的电影，无关评分
len = rate.shape[0]
kj = np.zeros((1, n))  # kj表示产品j的度
kl = np.zeros((1, m))  # kl表示用户l的度
count = 0
for ii in range(len):
    if testmark[ii, 0] == 0:  # 如果该行不是测试集中的行（是训练集中的行）
        jtemp = int(rate.iloc[ii].at['movieId'])  # 获取该行movieId
        ltemp = int(rate.iloc[ii].at['userId'])  # 获取该行userId
        for jjj in range(n):  # 将movieId转化为对应索引值
            if movielist[jjj] == jtemp:
                j = jjj
                break
        for iii in range(m):  # 将userId转化为对应索引值
            if userlist[iii] == ltemp:
                l = iii
                break
        mark[j, l] = 1  # 标记看过的电影
        if rate.iloc[ii].at['rating'] > 3:
            A[j][l] = 1  # 标记用户喜欢的电影
            kj[0, j] = kj[0, j] + 1  # 电影的度+1
            kl[0, l] = kl[0, l] + 1  # 用户的度+1
    if testmark[ii, 0] == 1:  # 如果该行是训练集中的行
        if count == 0:
            testrate[count, :] = rate.iloc[[ii]]  # 首行赋值
        else:
            testrate = np.concatenate((testrate, rate.iloc[[ii]]), axis=0,)  # 非首行在底部合并
        count += 1  # 统计测试集中的实际有效行数
        # print(ii)

# print(testrate)

w = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        wtemp = 0
        for l in range(m):
            if kl[0, l] != 0:
                wtemp += A[i, l] * A[j, l]/kl[0, l]
        if kj[0, j] != 0:
            w[i, j] = wtemp/kj[0, j]
f = np.dot(w, A)
for i in range(m):
    for j in range(n):
        if mark[j, i] != 0:
            f[j, i] = -1

L = np.zeros((1, m))
for i in range(m):
    for j in range(n):
        if A[j, i] == 0:
            L[0, i] += 1
R = np.zeros((m, n))  # 电影推荐程度排序中的位次
r = np.zeros((m, n))  # 相对位置
ff = np.zeros((n, m))  # f中推荐值在前列的进行最终的推荐，将推荐程度转化为是/否推荐
for i in range(m):  # 对于每个用户来说
    # print(f[:, i])
    sort_index = np.argsort(-f[:, i])  # 对每个用户对应的f中的列输出其从大到小排列时的位次
    for ii in range(sort_index.shape[0]):  # 对于每一个排列
        R[i, sort_index[ii]] = ii  # 将索引与值换一下存入R
        if ii <= (sort_index.shape[0]/3):
            ff[sort_index[ii], i] = 1  # 对f中推荐程度排在前1/3的电影做最终推荐
        # if L[0, i] != 0:
        #     r[i, sort_index[ii]] = R[i, sort_index[ii]]/L[0, i]

testA = np.zeros((n, m))
for ii in range(count):
    jtemp = int(testrate[ii, 1])
    ltemp = int(testrate[ii, 0])
    for jjj in range(n):
        if movielist[jjj] == jtemp:
            j = jjj
            break
    for iii in range(m):
        if userlist[iii] == ltemp:
            l = iii
            break
    if testrate[ii, 2] > 3:
        testA[j, l] = 1
        if L[0, i] != 0:
            r[l, j] = R[l, j]/L[0, l]  # 计算相对位置


rr = np.mean(r)
print(rr)

# print(ff)
# print(testA)

user = 0  # 指定一用户
fpr, tpr, threshold = roc_curve(testA[:, user], ff[:, user])  # 计算真正率和假正率
roc_auc = auc(fpr, tpr)
lw = 2
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # # # 假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
