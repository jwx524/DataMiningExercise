import numpy as np
import pandas as pd
import copy
import networkx as nx
import matplotlib.pyplot as plt

v = pd.read_csv("C:\\Users\\asus\\Desktop\\大三下\\数据挖掘\\karate\\karate-vertice.csv")
e = pd.read_csv("C:\\Users\\asus\\Desktop\\大三下\\数据挖掘\\karate\\karate-edge.csv")
vlen = v.shape[0]  # 点的个数
elen = e.shape[0]  # 边的个数

# 将v, e转化为列表
vertices = []
edges = []
for ii in range(vlen):
    vtemp = int(v.iloc[ii].at['Id'])
    vertices.append(vtemp)
for ii in range(elen):
    etemp1 = int(e.iloc[ii].at['Source'])
    etemp2 = int(e.iloc[ii].at['Target'])
    etemp = [etemp1, etemp2]
    edges.append(etemp)

# 建立邻接矩阵
A = np.zeros((vlen+1, vlen+1))
for ii in range(elen):
    etemp1 = int(e.iloc[ii].at['Source'])
    etemp2 = int(e.iloc[ii].at['Target'])
    A[etemp1, etemp2] = 1
# print(A)

N = [[]]  # 定义每个节点的邻居节点
for ii in range(vlen):
    i = ii + 1
    N.append([])
    for jj in range(vlen):
        j = jj + 1
        if A[i, j] == 1:  # 存在从i到j的边
            N[i].append(j)
        if A[j, i] == 1:  # 存在从j到i的边
            N[i].append(j)
S = np.zeros((vlen+1, vlen+1))  # 相似性度量指标
for i in range(vlen+1):
    for j in range(vlen+1):
        if len(set(N[i]) | set(N[j])) != 0:
            S[i, j] = len(set(N[i]) & set(N[j])) / len(set(N[i]) | set(N[j]))

clusters = []
for ii in range(vlen):
    i = ii + 1
    clusters.append([i])
clen = len(clusters)
total = 0
for ii in range(vlen+1):
    for jj in range(vlen+1):
        total += S[ii][jj]
# print(total)


def similarityIndicator(subCluster1, subCluster2, similarityMat, total):
    # 该函数用于计算两个子集Xp和Xq之间的相似度
    # 分子是分别位于子集Xp和Xq之间两两元素的相似性的和，分母是全集X中所有两两元素的相似性和
    nominator = 0
    for each1 in subCluster1:
        for each2 in subCluster2:
            nominator += similarityMat[each1][each2]
    return nominator / total


while True:
    maxDeltaQ = 0
    clusterToCombine1 = 0
    clusterToCombine2 = 0
    clustersCopy = copy.deepcopy(clusters)
    currentQ = 0
    for m in range(len(clustersCopy)):
        innerTotal = 0
        for n in range(len(clustersCopy)):
            innerTotal += similarityIndicator(clustersCopy[m], clustersCopy[n], S, total)
        innerTotal = innerTotal * innerTotal
        outer = similarityIndicator(clustersCopy[m], clustersCopy[m], S, total) - innerTotal
        currentQ += outer
    for i in range(clen):
        for j in range(i+1, clen):
            clustersCopy = copy.deepcopy(clusters)
            clustersCopy[i].extend(clustersCopy[j])
            clustersCopy.remove(clustersCopy[j])  # 得到假设合并后的新聚类
            outerTotal = 0
            for m in range(len(clustersCopy)):
                innerTotal = 0
                for n in range(len(clustersCopy)):
                    innerTotal += similarityIndicator(clustersCopy[m], clustersCopy[n], S, total)
                innerTotal = innerTotal * innerTotal
                outer = similarityIndicator(clustersCopy[m], clustersCopy[m], S, total) - innerTotal
                outerTotal += outer
            newDeltaQ = outerTotal - currentQ  # Q值相比原来增加了多少
            if newDeltaQ > maxDeltaQ:  # 记录Q值增加最多的一次假设合并
                maxDeltaQ = newDeltaQ  # 记录Q的增加值
                clusterToCombine1 = i  #记录假设合并中的两个被合并的子集
                clusterToCombine2 = j
    if maxDeltaQ == 0:  # 如果Q值不再变化，停止循环
        break
    clusters[clusterToCombine1].extend(clusters[clusterToCombine2])  # 如果Q值变化，则根据记录情况合并相应子集
    clusters.remove(clusters[clusterToCombine2])
    clen = len(clusters)

print('最终聚类结果：', clusters)
G = nx.Graph()
G.add_nodes_from(vertices)
G.add_edges_from(edges)
colorList = []
for eachVertice in vertices:
    if eachVertice in clusters[0]:
        colorList.append('red')
    else:
        colorList.append('blue')
nx.draw(G, with_labels=True, node_color=colorList, edge_color='black', node_size=400, alpha=0.5)
plt.show()
