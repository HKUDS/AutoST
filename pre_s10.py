
import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
from torch import nn
import networkx as nx
import numpy as np


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

region = load_data("../data/hy_vector_signal_trans_18.pickle")
check_vector = load_data("../data/ck_poi.pickle") 
hy = load_data("../data/hy_6.pickle") 


print(region.size())


print(check_vector[0].size())
print(hy.nodes())
print(hy)

hy_nodes_dict={}
for n,n_vec in zip(hy.nodes(),region):
	tp = n.split("_")[1]
	if tp not in hy_nodes_dict.keys():
		hy_nodes_dict[tp] = []
		hy_nodes_dict[tp].append(n_vec.tolist())
	else:
		hy_nodes_dict[tp].append(n_vec.tolist())

hy_com  = {}
for key,value in hy_nodes_dict.items():
    tmp = np.mean(value, axis=0).tolist()
    tmp_ = torch.tensor(tmp).tolist()
    hy_com[int(key)]  = tmp_


file=open(r"../data/hy_com_dict_trans.pickle","wb")
pickle.dump(hy_com,file) #storing_list
file.close()
print("---finish---")



println()
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np
import mglearn

# 读取数据，并划分训练集和测试集
X,y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
# 通过设置不同的alpha值建立三个lasso实例
lasso = Lasso().fit(X_train,y_train)
lasso001 =Lasso(alpha=0.01).fit(X_train,y_train)
lasso00001 = Lasso(alpha=0.0001).fit(X_train,y_train)
print('**********************************')
print("Lasso alpha=1")
print ("training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso.coef_!=0)))

print('**********************************')
print("Lasso alpha=0.01")
print ("training set score:{:.2f}".format(lasso001.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso001.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso001.coef_!=0)))

print('**********************************')
print("Lasso alpha=0.0001")
print ("training set score:{:.2f}".format(lasso00001.score(X_train,y_train)))
print ("test set score:{:.2f}".format(lasso00001.score(X_test,y_test)))
print ("Number of features used:{}".format(np.sum(lasso00001.coef_!=0)))














