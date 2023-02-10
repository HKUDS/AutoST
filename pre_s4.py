
import pickle
import pandas as pd
import numpy as np
import copy
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
import networkx as nx
import matplotlib.pyplot as pl




def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file


reg_vec_sort = load_data("../data/reg_poi_vec.pickle")
region_que = load_data("../data/reg_poi_idx.pickle")
region_attr_edges=[]


for idx in region_que:
    for idt in range(idx+1, len(reg_vec_sort)):
        # print("^^:",reg_vec_sort[idx].size())
        # print("**:",reg_vec_sort[idx+1].size())
        # pritnln()
        output = torch.cosine_similarity(torch.unsqueeze(reg_vec_sort[idx],0), torch.unsqueeze(reg_vec_sort[idt],0), eps=1e-08).mean()
        # print("output:", output.item())
        # pritnln()
        #0.87
        if output.item()>=0.9:
            tmp_1 = "r" + '_' + str(idx)+"_"+"p"
            tmp_2 = "r" + '_' + str(idt)+"_"+"p"
            # sim_dict[key] = [tmp_1, tmp_2, value]
            region_attr_edges.append([tmp_1, tmp_2, output.item()])
print(len(region_attr_edges))
# println()
G = nx.Graph()
# for edge in edges:
#     G.add_edge(edge[0],edge[1],weight= edge[2])

[G.add_edge(edge[0],edge[1],weight= edge[2], date = "1", start = edge[0], end = edge[1] ) for edge in region_attr_edges]
# print(len(G.adj))
# nx.draw(G, with_labels=True)
# plt.show()


file=open(r"../data/region_attr_graph_test.pickle","wb")
pickle.dump(G,file) #storing_list
file.close()

print("attr_region:", G)






























































